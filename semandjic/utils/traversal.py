import logging
from typing import Dict, Any, List, Tuple, Optional, Set, Type

from django.db import models

from ..constants import (
    get_date_formatters,
    get_fields_to_exclude,
    get_max_depth,
    get_max_related
)

logger = logging.getLogger(__name__)

def clean_model_dict(
        instance: models.Model,
        prefix: str = '',
        excluded_fields: Optional[Set[str]] = None
) -> Dict[str, Any]:
    """Clean model instance dictionary with efficient formatting"""
    excluded = excluded_fields or get_fields_to_exclude()
    result = {}
    formatters = get_date_formatters()

    for key, value in instance.__dict__.items():
        if key.startswith('_') or key in excluded:
            continue

        if isinstance(value, tuple(formatters)):
            value = value.strftime(formatters[type(value)])

        new_key = f"{prefix}{key}" if prefix else key
        result[new_key] = value

    return result

def get_object_full_tree(
        instance: models.Model,
        visited_models: Optional[Set[Type[models.Model]]] = None,
        visited_paths: Optional[Set[Tuple[str, str]]] = None,
        depth: int = 0,
        max_depth: Optional[int] = None,
        current_path: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Build complete tree including forward and backward relations"""
    if visited_models is None:
        visited_models = set()
    if visited_paths is None:
        visited_paths = set()
    if current_path is None:
        current_path = []
    if max_depth is None:
        max_depth = get_max_depth()

    current_model = instance.__class__
    current_type = f"{current_model._meta.app_label}.{current_model.__name__}"

    # Check max depth
    if depth >= max_depth:
        return {
            'id': instance.pk,
            'type': current_type,
            'fields': {'note': f'Max depth reached for {current_type} #{instance.pk}'},
        }

    # Detect cycles in the path
    if current_type in current_path:
        return {
            'id': instance.pk,
            'type': current_type,
            'fields': {'note': f'Cycle detected for {current_type} #{instance.pk}'},
        }

    current_path = current_path + [current_type]
    data = {
        'id': instance.pk,
        'type': current_type,
        'fields': clean_model_dict(instance),
        'forward_relations': {},
        'backward_relations': {}
    }

    max_related = get_max_related()

    # Process all relations
    for field in current_model._meta.get_fields():
        if not field.is_relation:
            continue

        # Get the related model and direction
        if field.auto_created:  # Backward relation
            related_model = field.related_model
            relation_type = 'backward'
            name = field.get_accessor_name()
        else:  # Forward relation
            related_model = field.related_model if hasattr(field, 'related_model') else None
            relation_type = 'forward'
            name = field.name

        if not related_model:
            continue

        # Create path identifier
        path_key = tuple(sorted([current_model.__name__, related_model.__name__]))

        # Skip if we've seen this path before
        if path_key in visited_paths:
            continue

        visited_paths.add(path_key)

        try:
            if relation_type == 'forward':
                related_obj = getattr(instance, name)
                if related_obj is not None:
                    if isinstance(related_obj, models.Manager):
                        # Handle many-to-many
                        data['forward_relations'][name] = [
                            get_object_full_tree(
                                obj,
                                visited_models.copy(),
                                visited_paths.copy(),
                                depth + 1,
                                max_depth,
                                current_path
                            )
                            for obj in related_obj.all()[:max_related]
                        ]
                    else:
                        # Handle foreign key
                        data['forward_relations'][name] = get_object_full_tree(
                            related_obj,
                            visited_models.copy(),
                            visited_paths.copy(),
                            depth + 1,
                            max_depth,
                            current_path
                        )
            else:  # backward relation
                related_manager = getattr(instance, name)
                related_objects = related_manager.all()[:max_related]

                if related_objects:
                    data['backward_relations'][name] = [
                        get_object_full_tree(
                            obj,
                            visited_models.copy(),
                            visited_paths.copy(),
                            depth + 1,
                            max_depth,
                            current_path
                        )
                        for obj in related_objects
                    ]

        except Exception as e:
            logger.warning(f"Could not process relation {name}: {e}")
            continue

    return data