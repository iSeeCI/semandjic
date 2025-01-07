from .settings import SemandjicSettings

class Onto:
    """Constants for ontology-like relationships"""
    ENTITY_UNDER_SEP = SemandjicSettings.get('ENTITY_SEPARATOR')
    DOCUMENT = "document"
    UNSTRUCTURED = "unstructured"

def get_settings():
    """Get current settings configuration"""
    return SemandjicSettings.get_all()

def get_fields_to_exclude():
    return SemandjicSettings.get('FIELDS_TO_EXCLUDE')

def get_date_formatters():
    return SemandjicSettings.get('DATE_FORMATTERS')

def get_max_depth():
    return SemandjicSettings.get('MAX_DEPTH')

def get_max_related():
    return SemandjicSettings.get('MAX_RELATED_OBJECTS')