from datetime import date, datetime, time
from typing import TypedDict, Set, Dict, Union, Type


class NestedFormsSettings(TypedDict, total=False):
    APP_LABEL: str
    FIELDS_TO_EXCLUDE: Set[str]
    DATE_FORMATTERS: Dict[Union[Type[date], Type[datetime], Type[time]], str]
    ENTITY_SEPARATOR: str
    MAX_DEPTH: int
    MAX_RELATED_OBJECTS: int
    EXCLUDED_RELATIONS: Set[str]