from django.urls import path

from .views import NestedFormView, ObjectTreeView

app_name = 'semandjic'

urlpatterns = [
    path('form/<str:model_class>/', NestedFormView.as_view(), name='nested-form'),
    path('object/<str:model_class>/<int:pk>/', ObjectTreeView.as_view(), name='object-tree'),
]