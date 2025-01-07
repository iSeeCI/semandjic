import logging
from typing import Dict, Any, List, Tuple, Optional, Type, Union, Set

from django import forms
from django.core.exceptions import ObjectDoesNotExist
from django.db import models
from django.forms import ModelForm
from django.http import QueryDict

from ..constants import (
    Onto,
)
from ..utils.introspection import (
    get_model_fields,
    get_model_calculated_fields,
    get_model_unique_fields,
    resolve_class_from_name,
)
from ..utils.traversal import (
    clean_model_dict,
)

logger = logging.getLogger(__name__)

def formfield_for_dbfield(db_field: models.Field, **kwargs) -> Optional[forms.Field]:
    """Custom formfield callback for empty defaults"""
    db_field.default = str  # Using str as empty default
    return db_field.formfield(**kwargs)

class NestedForms:
    """Handles nested form generation and processing for Django models"""

    @staticmethod
    def get_model_fields(model: Type[models.Model]) -> List[str]:
        """Extract visible form fields from model"""
        return get_model_fields(model)

    @classmethod
    def build_classmap_from_prefix_and_model_class(
            cls,
            prefix: str,
            model_class: str,
            related_fields: Optional[Set[str]] = None,
            visited: Optional[Set[Type[models.Model]]] = None
    ) -> Dict[str, Tuple[str, List[str]]]:
        """Build classmap recursively using model class names"""
        from ..utils.introspection import resolve_class_from_name

        visited = visited or set()
        model = resolve_class_from_name(model_class)

        if model in visited:
            return {}

        visited.add(model)
        classmap = {prefix: (model_class, cls.get_model_fields(model))}

        # Add related models - only forward relations
        for field in model._meta.get_fields():
            if field.is_relation and not field.auto_created and (related_fields is None or field.name in related_fields):
                related_prefix = f"{prefix}__{field.name}"
                related_model_class = f"{field.related_model._meta.app_label}.{field.related_model.__name__}"
                classmap.update(
                    cls.build_classmap_from_prefix_and_model_class(
                        related_prefix,
                        related_model_class,
                        related_fields,
                        visited
                    )
                )

        return classmap

    @classmethod
    def get_nested_forms_from_classmap(
            cls,
            classmap: Dict[str, Tuple[str, List[str]]],
            default_data: bool = True
    ) -> Dict[str, ModelForm]:
        """Generate nested forms from a classmap efficiently"""
        return {
            participant: cls.get_custom_form_from_classmap(classmap, participant, default_data)(prefix=participant)
            for participant in classmap
        }

    @classmethod
    def get_custom_form_from_classmap(
            cls,
            classmap: Dict[str, Tuple[str, List[str]]],
            participant: str,
            default_data: bool
    ) -> Type[Union[forms.Form, ModelForm]]:
        """Create a custom form class with proper widget handling"""
        try:
            kls, fields = classmap[participant]
            exc_fields = cls.get_excluded_fields(classmap, participant)

            if participant == Onto.UNSTRUCTURED:
                return type(participant, (forms.Form,), {
                    field: forms.CharField() for field in fields
                })

            model_class = resolve_class_from_name(kls)
            calc_fields = get_model_calculated_fields(model_class)
            widgets = {
                field.name: forms.DateInput(attrs={'type': 'date'})
                for field in model_class._meta.get_fields()
                if isinstance(field, models.DateField)
            }

            factory_kwargs = {
                'model': model_class,
                'fields': fields,
                'exclude': exc_fields + calc_fields,
                'widgets': widgets,
            }

            if not default_data:
                factory_kwargs['formfield_callback'] = formfield_for_dbfield

            return forms.modelform_factory(**factory_kwargs)

        except KeyError as e:
            logger.error(f"Invalid participant in classmap: {e}")
            raise ValueError(f"Participant {participant} not found in classmap")

    @classmethod
    def persist_nested_forms_and_objs(
            cls,
            classmap: Dict[str, Tuple[str, List[str]]],
            posted_data: Optional[Dict[str, Any]] = None,
            default_data: bool = True
    ) -> Tuple[Dict[str, ModelForm], bool, List[models.Model]]:
        """Persist nested forms and objects with optimized validation"""
        logger.info("Persisting nested forms and objects")
        available: Dict[str, models.Model] = {}
        forms: Dict[str, ModelForm] = {}
        objects: List[models.Model] = []
        is_valid = True

        try:
            for participant in reversed(classmap):
                form, template_fields = cls.get_object(classmap, participant, posted_data, default_data)
                forms[participant] = form

                if not form.is_valid():
                    logger.error(f"Form validation errors for {participant}: {form.errors}")
                    is_valid = False
                    continue

                instance = cls.override_unset_members(form, template_fields)

                if participant != Onto.DOCUMENT:
                    instance = cls.get_existing_or_create(instance, posted_data)

                cls.assign_relation_inplace(available, participant, instance)
                available[participant] = instance
                objects.append(instance)

            return forms, is_valid, objects

        except Exception as e:
            logger.error(f"Error in persist_nested_forms_and_objs: {e}", exc_info=True)
            raise

    @classmethod
    def persist_nested_forms_and_save(
            cls,
            classmap: Dict[str, Tuple[str, List[str]]],
            posted_data: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, ModelForm], bool, List[models.Model]]:
        """Persist and save nested forms in a single transaction"""
        forms, valid, objects = cls.persist_nested_forms_and_objs(classmap, posted_data)

        if valid:
            for obj in objects:
                obj.save()

        return forms, valid, objects

    @staticmethod
    def from_dict_to_prefix(entity_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Convert dictionary keys to form prefix format"""
        return {
            "-".join(k.rsplit(Onto.ENTITY_UNDER_SEP, 1)): str(v) if v is not None else ""
            for k, v in entity_dict.items()
        }

    @classmethod
    def get_object(
            cls,
            classmap: Dict[str, Tuple[str, List[str]]],
            participant: str,
            posted_data: Optional[Dict[str, Any]],
            default_data: bool
    ) -> Tuple[ModelForm, List[str]]:
        """Get form object and required fields with proper validation"""
        logger.info(f"Getting form for: {participant}")

        form = cls.get_custom_form_from_classmap(
            classmap,
            participant,
            default_data
        )(posted_data, prefix=participant)

        _, fields = classmap[participant]
        excluded_fields = cls.get_excluded_fields(classmap, participant)
        required_fields = list(fields) + excluded_fields
        return form, required_fields

    @classmethod
    def get_excluded_fields(
            cls,
            classmap: Dict[str, Tuple[str, List[str]]],
            current: str
    ) -> List[str]:
        """Get excluded fields for a given class in the map"""
        separator = Onto.ENTITY_UNDER_SEP
        excluded = {'created_at', 'updated_at', 'is_active'}

        # Add direct children fields
        prefix = f"{current}{separator}"
        child_fields = {
            key[len(prefix):]
            for key in classmap
            if key != current and key.startswith(prefix)
               and key[len(prefix):].count(separator) == 0
        }
        excluded.update(child_fields)

        return list(excluded)

    @classmethod
    def assign_relation_inplace(
            cls,
            available: Dict[str, models.Model],
            participant: str,
            instance: models.Model
    ) -> None:
        """Assign relations to model instance safely"""
        for avail_key, value in available.items():
            try:
                parts = avail_key.split(Onto.ENTITY_UNDER_SEP)
                field = parts[-1]
                parent_path = Onto.ENTITY_UNDER_SEP.join(parts[:-1]) if len(parts) > 1 else ""

                if participant == parent_path or not parent_path:
                    target_value = value
                    if participant == Onto.DOCUMENT:
                        current_value = getattr(instance, field) if field != Onto.DOCUMENT else instance
                        while id(target_value) == id(current_value):
                            target_value = value.parent

                    setattr(instance, field, target_value)
            except (IndexError, AttributeError) as e:
                logger.error(f"Failed to assign relation: {e}")
                continue

    @classmethod
    def get_recursive_instance(
            cls,
            instance: models.Model,
            participant: str
    ) -> models.Model:
        """Safely traverse nested instance attributes"""
        try:
            current = instance
            fields = participant.split(Onto.ENTITY_UNDER_SEP)

            for field in fields[1:]:
                current = getattr(current, field)
            return current
        except AttributeError as e:
            logger.error(f"Failed to get recursive instance: {e}")
            raise AttributeError(f"Invalid path {participant} for instance {instance}")

    @classmethod
    def override_unset_members(
            cls,
            form: ModelForm,
            template_fields: List[str]
    ) -> models.Model:
        """Override unset form members with None"""
        model_class = form.Meta.model
        instance = form.save(commit=False)

        for field in model_class._meta.get_fields():
            if (
                    field.name not in template_fields
                    and field.concrete
                    and not field.is_relation
                    and not any(x in field.model.__name__.lower() for x in ["auditedmodel"])
            ):
                setattr(instance, field.name, None)

        return instance

    @classmethod
    def get_existing_or_create(
            cls,
            instance: models.Model,
            posted_data: Optional[Dict[str, Any]]
    ) -> models.Model:
        """Get existing instance or create new one with proper error handling"""
        var_dict = clean_model_dict(instance)
        search_fields = get_model_unique_fields(instance.__class__)
        search_dict = {k: v for k, v in var_dict.items() if k in search_fields}

        should_update = (
                posted_data
                and "upsert" in posted_data
                and posted_data["upsert"] != "justget"
        )

        try:
            if search_dict:
                queryset = instance.__class__.objects.filter(**search_dict)
                if should_update:
                    existing_defaults = {
                        k: v for k, v in var_dict.items()
                        if k in (f.name for f in instance.__class__._meta.fields)
                    }
                    queryset.update(**existing_defaults)
                instance = queryset.get()
            else:
                instance = instance.__class__(**var_dict)

            return instance
        except ObjectDoesNotExist:
            return instance.__class__(**var_dict)
        except Exception as e:
            logger.error(f"Error in get_existing_or_create: {e}")
            raise

    @classmethod
    def get_post_data_from_forms_default(
            cls,
            forms: Dict[str, ModelForm]
    ) -> QueryDict:
        """Get post data from forms with proper defaults"""
        post_data = QueryDict('', mutable=True)
        post_data.update({
            form.add_prefix(field_name): field.initial if field.initial is not None else ''
            for form in forms.values()
            for field_name, field in form.fields.items()
        })
        return post_data


    @classmethod
    def build_form_tree(
            cls,
            forms_dict: Dict[str, ModelForm]
    ) -> Dict[str, Dict[str, Any]]:
        """Build hierarchical form tree structure efficiently"""
        tree = {}

        for key in sorted(forms_dict, key=len):
            current = tree
            *parts, last = key.split(Onto.ENTITY_UNDER_SEP)

            for part in parts:
                if part not in current:
                    current[part] = {
                        'form': None,
                        'children': {}
                    }
                current = current[part]['children']

            if last not in current:
                current[last] = {
                    'form': None,
                    'children': {}
                }
            current[last]['form'] = forms_dict[key]

        return tree