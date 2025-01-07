from dataclasses import dataclass
from typing import Optional, Type

from django.db import models


@dataclass
class ModelFieldInfo:
    """Dataclass to store model field information"""
    null: bool
    related: Optional[Type[models.Model]] = None