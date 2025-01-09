import datetime
from typing import Dict, Any

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured

SEMANDJIC_DEFAULTS = {
    'FIELDS_TO_EXCLUDE': {
        "id",
        "real_type_id",
        "created_at",
        "updated_at",
        "is_active"
    },
    'DATE_FORMATTERS': {
        datetime.date: '%Y-%m-%d',
        datetime.datetime: '%Y-%m-%dT%H:%M:%S',
        datetime.time: '%H:%M:%S'
    },
    'ENTITY_SEPARATOR': '__',
    'MAX_DEPTH': 10,
    'MAX_RELATED_OBJECTS': 10
}

class SemandjicSettings:
    """Settings handler for nested forms with dynamic override capabilities"""

    @classmethod
    def validate_setting(cls, setting: str, value: Any) -> Any:
        """Validate individual settings"""
        if setting == 'MAX_DEPTH':
            if not isinstance(value, int) or value < 1:
                raise ImproperlyConfigured(
                    f'SEMANDJIC["MAX_DEPTH"] must be a positive integer, got {value}'
                )
        elif setting == 'FIELDS_TO_EXCLUDE':
            if not isinstance(value, (set, list, tuple)):
                raise ImproperlyConfigured(
                    f'SEMANDJIC["FIELDS_TO_EXCLUDE"] must be a sequence, got {type(value)}'
                )
            value = set(value)  # Convert to set if it's list/tuple

        return value

    @classmethod
    def get(cls, setting: str, default: Any = None) -> Any:
        """Get a validated setting value"""
        value = getattr(settings, 'SEMANDJIC', {}).get(
            setting,
            default or SEMANDJIC_DEFAULTS.get(setting)
        )
        return cls.validate_setting(setting, value)

    @classmethod
    def get_all(cls) -> Dict[str, Any]:
        """Get all settings with overrides applied"""
        django_settings = getattr(settings, 'SEMANDJIC', {})
        all_settings = SEMANDJIC_DEFAULTS.copy()
        all_settings.update(django_settings)
        return all_settings