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
                'formfield_callback': None if default_data else formfield_for_dbfield
            }

            return forms.modelform_factory(**factory_kwargs)

        except KeyError as e:
            logger.error(f"Invalid participant in classmap: {e}")
            raise ValueError(f"Participant {participant} not found in classmap")

    @classmethod
    def get_custom_form_from_instance(
            cls,
            classmap: Dict[str, Tuple[str, List[str]]],
            instance: models.Model
    ) -> Dict[str, ModelForm]:
        """Generate forms from an existing model instance using the classmap structure."""
        logger.info("Getting custom forms from instance")
        logger.info(f"Classmap: {classmap}")
        logger.info(f"Instance: {instance}")

        forms = {}
        for participant in reversed(classmap.keys()):
            form_class = cls.get_custom_form_from_classmap(classmap, participant, default_data=False)
            recursive_instance = cls.get_recursive_instance(instance, participant)
            forms[participant] = form_class(instance=recursive_instance, prefix=participant)

        return forms

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
                # First try to get existing instance
                temp_form = cls.get_custom_form_from_classmap(classmap, participant, default_data)(
                    posted_data, prefix=participant
                )

                existing_instance = None
                if participant != Onto.DOCUMENT:
                    temp_instance = None
                    if temp_form.is_valid():
                        temp_instance = temp_form.save(commit=False)
                    else:
                        # Even if form is invalid (potentially due to unique constraints)
                        # Try to create a model instance from the form data
                        model_class = temp_form._meta.model
                        temp_data = {field: temp_form.data.get(temp_form.add_prefix(field))
                                     for field in temp_form.fields}
                        temp_instance = model_class(**temp_data)

                    if temp_instance:
                        existing_instance = cls.get_existing_or_create(temp_instance, posted_data)

                # Create the actual form with the instance if it exists
                form = cls.get_custom_form_from_classmap(classmap, participant, default_data)(
                    posted_data,
                    prefix=participant,
                    instance=existing_instance
                )
                forms[participant] = form

                if not form.is_valid():
                    logger.error(f"Final form validation errors for {participant}: {form.errors}")
                    is_valid = False
                    continue

                instance = form.save(commit=False)
                cls.assign_relation_inplace(available, participant, instance)
                available[participant] = instance
                objects.append(instance)

            return forms, is_valid, objects

        except Exception as e:
            logger.error(f"Error in persist_nested_forms_and_objs: {e}", exc_info=True)
            raise

        except Exception as e:
            logger.error(f"Error in persist_nested_forms_and_objs: {e}", exc_info=True)
            raise

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
    def get_existing_or_create(
            cls,
            instance: models.Model,
            posted_data: Optional[Dict[str, Any]]
    ) -> Optional[models.Model]:
        """Get existing instance or create new one with proper error handling"""
        var_dict = clean_model_dict(instance)
        search_fields = get_model_unique_fields(instance.__class__)
        search_dict = {k: v for k, v in var_dict.items() if k in search_fields}

        try:
            if search_dict:
                try:
                    existing = instance.__class__.objects.get(**search_dict)
                    if posted_data and "upsert" in posted_data and posted_data["upsert"] != "justget":
                        # Update existing instance with new values
                        for field_name, value in var_dict.items():
                            if field_name in (f.name for f in instance.__class__._meta.fields):
                                setattr(existing, field_name, value)
                    return existing
                except ObjectDoesNotExist:
                    if posted_data and "upsert" in posted_data:
                        return None
                    return instance

            # If no unique fields or all values are None
            if posted_data and "upsert" in posted_data:
                return None
            return instance

        except Exception as e:
            logger.error(f"Error in get_existing_or_create: {e}", exc_info=True)
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