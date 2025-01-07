# Semandjic

A Django app for handling nested forms and object trees with automatic relationship discovery.

## Features

- Automatic form generation for related models
- Handles nested relationships recursively
- Prevents circular references
- Configurable depth and relationship handling
- Support for forward and backward relations
- Customizable field exclusion
- Tree visualization of model relationships

## Installation

```bash
pip install semandjic
```

## Quick Start

1. Add to INSTALLED_APPS:

```python
INSTALLED_APPS = [
    ...,
    'semandjic',
]
```

2. Configure settings (optional):

```python
SEMANDJIC = {
    'FIELDS_TO_EXCLUDE': {'id', 'modified_at'},
    'MAX_DEPTH': 5,
    'MAX_RELATED_OBJECTS': 10,
}
```

3. Use the methods directly from the classes and write your own semantic data flows:

```python
from semandjic.forms import NestedForms

# Generate forms
classmap = NestedForms.build_classmap_from_prefix_and_model_class(
    prefix='someone',
    model_class='myapp.Person'
)

forms = NestedForms.get_nested_forms_from_classmap(classmap)

# Persist forms

forms, valid, objs = NestedForms.persist_nested_forms_and_objs(
    classmap,
    request.POST
)
if valid:
    for obj in objs:
        obj.save()
```

Also check for `test_default_data_workflow` in [test_forms.py](./semandjic/tests/test_forms.py) to see the simplicity of the API from schema to forms to related instances in a SQL Database using your own Applications Ontologies (models).

4. Use the included urls and views
```python
from django.urls import path

# including them
path('core/', include('semandjic.urls')),

# using them as you please
path('object/<str:model_class>/<int:pk>/', 
     ObjectTreeView.as_view(), 
     name='object-tree'),
```

## Documentation

For more detailed documentation, visit our [GitHub repository](https://github.com/iSeeCI/semandjic).

## Development

To contribute:

```bash
# Clone the repository
git clone https://github.com/iSeeCI/semandjic.git
cd semandjic

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install development dependencies
pip install -e ".[dev]"

# Run tests
python runtests.py
```

## License

MIT License