from django.http import JsonResponse
from django.views import View
from django.shortcuts import render, get_object_or_404, redirect
from django.views import View

from ..forms.nested import NestedForms
from ..utils.introspection import resolve_class_from_name
from ..utils.traversal import get_object_full_tree


class NestedFormView(View):
    template_name = 'semandjic/nested_forms.html'

    def get(self, request, model_class: str, prefix: str = None):
        prefix = prefix or 'something'
        classmap = NestedForms.build_classmap_from_prefix_and_model_class(prefix=prefix, model_class=model_class)
        forms = NestedForms.get_nested_forms_from_classmap(classmap, default_data=True)
        form_tree = NestedForms.build_form_tree(forms)
        return render(request, self.template_name, {'forms': form_tree})

    def post(self, request, model_class: str, prefix: str = None):
        prefix = prefix or 'something'
        classmap = NestedForms.build_classmap_from_prefix_and_model_class(prefix=prefix, model_class=model_class)
        forms, valid, objs = NestedForms.persist_nested_forms_and_objs(classmap, request.POST, default_data=True)
        form_tree = NestedForms.build_form_tree(forms)
        if valid:
            for o in objs:
                o.save()
            last_object = objs[-1]  # Get the last object
            model_name = f"{last_object._meta.app_label}.{last_object.__class__.__name__}"  # Get the model name in lowercase (or you can adjust it if needed)
            object_id = last_object.pk  # Get the primary key (ID) of the last object
            return redirect('object-tree', model_class=model_name, pk=object_id)
        else:
            return render(request, self.template_name, {'forms': form_tree})


class ObjectTreeView(View):
    template_name = 'semandjic/object_tree.html'

    def get(self, request, model_class: str, pk: int):
        """Get view handling both HTML and JSON responses"""
        model = resolve_class_from_name(model_class)
        instance = get_object_or_404(model, pk=pk)
        tree = get_object_full_tree(instance)

        # Return JSON if requested
        if 'application/json' in request.headers.get('Accept', ''):
            return JsonResponse(tree)

        return render(request, self.template_name, {
            'tree': tree,
            'model_name': model.__name__,
            'instance_id': pk
        })