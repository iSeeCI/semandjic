from typing import Dict, List, Type, Union

from django.apps import apps
from django.db import models

from ..constants import get_fields_to_exclude
from ..forms.fields import ModelFieldInfo


def resolve_class_from_name(class_name: str) -> Type[models.Model]:
    """Resolve a class from its string representation"""
    try:
        app_label, model_name = class_name.split('.')
        return apps.get_model(app_label, model_name)
    except (ValueError, LookupError) as e:
        raise ValueError(f"Could not resolve class name: {class_name}. Error: {e}")

def get_model_fields(model: Type[models.Model]) -> List[str]:
    """Extract visible form fields from model"""
    return [
        field.name for field in model._meta.get_fields()
        if not field.is_relation and field.name not in get_fields_to_exclude()
    ]

def get_model_calculated_fields(
        model_class: Type[models.Model],
        names_only: bool = True
) -> Union[List[str], Dict[str, List[str]]]:
    """Get calculated fields from model properties with dependency tracking"""
    calculated_fields: Dict[str, List[str]] = {}

    for name, member in vars(model_class).items():
        if isinstance(member, property):
            deps = []
            if member.__doc__:
                deps = [
                    dep.strip()
                    for line in member.__doc__.split('\n')
                    if line.strip().startswith('Depends on:')
                    for dep in line.split(':', 1)[1].split(',')
                ]
            calculated_fields[name] = deps

    return list(calculated_fields) if names_only else calculated_fields

def get_model_unique_fields(model_class: Type[models.Model]) -> List[str]:
    """Get unique fields for a model"""
    unique_fields = set()

    meta_unique = getattr(model_class._meta, 'unique_together', None)
    if meta_unique:
        unique_fields.update(field for constraint in meta_unique for field in constraint)

    unique_fields.update(
        field.name
        for field in model_class._meta.get_fields()
        if getattr(field, 'unique', False) and field.name != 'id'
    )

    return list(unique_fields)

def get_model_field_dict(model_class: Type[models.Model]) -> Dict[str, ModelFieldInfo]:
    """Get comprehensive field information dictionary"""
    return {
        field.name: ModelFieldInfo(
            null=getattr(field, 'null', True),
            related=getattr(field, 'related_model', None)
        )
        for field in model_class._meta.get_fields()
    }