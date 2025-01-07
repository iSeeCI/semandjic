from datetime import datetime
from unittest.mock import patch

import pytest
from django.forms import ModelForm
from django.test import Client
from django.test import TransactionTestCase
from django.urls import reverse

from .models import Person, Address
from ..forms.nested import NestedForms
from ..utils import get_model_unique_fields, clean_model_dict, get_model_calculated_fields, get_model_field_dict


class TestNestedForms(TransactionTestCase):
    databases = ["default"]
    classmap = {
        'someone': ('tests.Person', ['first_name', 'last_name', 'email', 'birth_date']),
        'someone__address': ('tests.Address', ['street', 'city', 'state', 'postal_code', 'country'])
    }

    def setUp(self):
        """Set up test data for each test method"""
        self.address = Address.objects.create(
            street='123 Test St',
            city='Test City',
            state='Test State',
            postal_code='12345',
            country='Test Country'
        )

        self.person = Person.objects.create(
            first_name='John',
            last_name='Doe',
            email='john@example.com',
            birth_date=datetime.strptime('1990-01-01', '%Y-%m-%d').date(),
            address=self.address
        )

    def test_get_nested_forms_from_classmap(self):
        """Test generation of nested forms from classmap"""
        forms = NestedForms.get_nested_forms_from_classmap(self.classmap)

        self.assertTrue(all(isinstance(form, ModelForm) for form in forms.values()))
        self.assertEqual(len(forms), len(self.classmap))

        # Verify form fields
        person_form = forms['someone']
        self.assertIn('first_name', person_form.fields)
        self.assertIn('email', person_form.fields)

        address_form = forms['someone__address']
        self.assertIn('street', address_form.fields)
        self.assertIn('postal_code', address_form.fields)

    def test_get_model_calculated_fields(self):
        """Test extraction of calculated fields from model"""
        fields = get_model_calculated_fields(Person)

        self.assertIn('full_name', fields)
        self.assertIn('age', fields)

        # Test with names_only=False
        fields_dict = get_model_calculated_fields(Person, names_only=False)
        self.assertIsInstance(fields_dict, dict)
        self.assertIn('full_name', fields_dict)
        self.assertIn('age', fields_dict)

        # Verify dependencies are correctly parsed
        self.assertEqual(sorted(fields_dict['full_name']), sorted(['first_name', 'last_name']))
        self.assertEqual(fields_dict['age'], ['birth_date'])

    def test_get_excluded_fields(self):
        """Test generation of excluded fields list"""
        excluded = NestedForms.get_excluded_fields(self.classmap, 'someone')

        self.assertIn('created_at', excluded)
        self.assertIn('updated_at', excluded)
        self.assertIn('is_active', excluded)
        self.assertIn('address', excluded)  # Should exclude ForeignKey field

    def test_get_recursive_instance(self):
        """Test traversal of nested instance attributes"""
        instance = self.person
        address = NestedForms.get_recursive_instance(instance, 'someone__address')

        self.assertEqual(address, self.address)

    def test_get_recursive_instance_invalid_path(self):
        """Test handling of invalid path in recursive instance traversal"""
        with self.assertRaises(AttributeError):
            NestedForms.get_recursive_instance(self.person, 'someone__invalid')

    def test_clean_model_dict(self):
        """Test cleaning of model instance dictionary"""
        cleaned = clean_model_dict(self.person)

        self.assertNotIn('id', cleaned)
        self.assertNotIn('created_at', cleaned)
        self.assertNotIn('updated_at', cleaned)
        self.assertNotIn('is_active', cleaned)
        self.assertIn('first_name', cleaned)
        self.assertIn('last_name', cleaned)
        self.assertIn('email', cleaned)

        # Test with prefix
        cleaned_prefix = clean_model_dict(self.person, prefix='person_')
        self.assertIn('person_first_name', cleaned_prefix)
        self.assertIn('person_email', cleaned_prefix)

    def test_build_classmap_from_prefix_and_model_class(self):
        """Test building classmap from model class"""
        classmap = NestedForms.build_classmap_from_prefix_and_model_class(
            prefix='someone',
            model_class='tests.Person'
        )

        self.assertIn('someone', classmap)
        self.assertIn('someone__address', classmap)

        # Verify field lists
        self.assertTrue(all(
            field in classmap['someone'][1]
            for field in ['first_name', 'last_name', 'email', 'birth_date']
        ))

    def test_get_model_unique_fields(self):
        """Test extraction of unique fields from model"""
        # Test Person model unique fields
        person_unique_fields = get_model_unique_fields(Person)
        self.assertIn('email', person_unique_fields)
        self.assertNotIn('id', person_unique_fields)

        # Test Address model unique fields (unique_together)
        address_unique_fields = get_model_unique_fields(Address)
        self.assertTrue(all(
            field in address_unique_fields
            for field in ['street', 'city', 'postal_code']
        ))

    @patch('django.forms.ModelForm.is_valid')
    def test_persist_nested_forms_and_objs(self, mock_is_valid):
        """Test persistence of nested forms and objects"""
        mock_is_valid.return_value = True

        posted_data = {
            'someone-first_name': 'Jane',
            'someone-last_name': 'Doe',
            'someone-email': 'jane@example.com',
            'someone-birth_date': '1995-01-01',
            'someone__address-street': '456 Test Ave',
            'someone__address-city': 'New City',
            'someone__address-state': 'New State',
            'someone__address-postal_code': '54321',
            'someone__address-country': 'New Country'
        }

        forms, is_valid, objects = NestedForms.persist_nested_forms_and_objs(
            self.classmap,
            posted_data
        )

        self.assertTrue(is_valid)
        self.assertEqual(len(objects), 2)  # Should have person and address objects
        self.assertTrue(all(isinstance(form, ModelForm) for form in forms.values()))

    def test_get_model_field_dict(self):
        """Test generation of field dictionary with metadata"""
        field_dict = get_model_field_dict(Person)

        self.assertIn('first_name', field_dict)
        self.assertIn('address', field_dict)
        self.assertTrue(field_dict['address'].related == Address)
        self.assertFalse(field_dict['email'].null)
        self.assertTrue(field_dict['birth_date'].null)

    def test_from_dict_to_prefix(self):
        """Test conversion of dictionary keys to form prefix format"""
        entity_dict = {
            'someone__address': '123 Test St',
            'someone__first_name': 'John'
        }

        prefixed = NestedForms.from_dict_to_prefix(entity_dict)

        self.assertEqual(prefixed['someone-address'], '123 Test St')
        self.assertEqual(prefixed['someone-first_name'], 'John')

        # Test handling of None values
        entity_dict['someone__middle_name'] = None
        prefixed = NestedForms.from_dict_to_prefix(entity_dict)
        self.assertEqual(prefixed['someone-middle_name'], '')

    def test_default_data_workflow(self):
        """Test full workflow with default data"""
        # Test with explicit classmap
        forms1 = NestedForms.get_nested_forms_from_classmap(self.classmap, default_data=True)
        post_data = NestedForms.get_post_data_from_forms_default(forms1)
        forms2, valid, objs = NestedForms.persist_nested_forms_and_objs(
            self.classmap,
            post_data,
            default_data=True
        )

        self.assertTrue(valid)

        for obj in objs:
            obj.save()
            self.assertTrue(obj.pk)  # Verify objects were saved

    @pytest.mark.django_db
    def test_nested_form_view_get_html(cls):
        client = Client()
        model_class = "tests.Person"  # Replace with your actual model class name
        url = reverse('nested-form', kwargs={'model_class': model_class})
        response = client.get(url)

    @pytest.mark.django_db
    def test_object_tree_view_get_html(cls):
        client = Client()
        model_class = "tests.Person"  # Replace with your actual model class name
        url = reverse('nested-form', kwargs={'model_class': model_class})
        response = client.get(url)
