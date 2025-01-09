from datetime import datetime

from django.db import models


class Address(models.Model):
    street = models.CharField(max_length=255, default="street")
    city = models.CharField(max_length=100, default="city")
    state = models.CharField(max_length=50, default="state")
    postal_code = models.CharField(max_length=10, null=True, blank=True)
    country = models.CharField(max_length=100, default="country")
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = [['street', 'city', 'postal_code']]

class Person(models.Model):
    first_name = models.CharField(max_length=100, default="fn")
    last_name = models.CharField(max_length=100, default="ln")
    email = models.EmailField(unique=True, default="d@em.co")
    birth_date = models.DateField(null=True, blank=True)
    address = models.ForeignKey(Address, on_delete=models.CASCADE)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    @property
    def full_name(self) -> str:
        """
        Get person's full name.
        Depends on: first_name,last_name
        """
        return f"{self.first_name} {self.last_name}"

    @property
    def age(self) -> int:
        """
        Calculate person's age.
        Depends on: birth_date
        """
        if not self.birth_date:
            return 0
        today = datetime.now()
        return today.year - self.birth_date.year - ((today.month, today.day) < (self.birth_date.month, self.birth_date.day))

class Contact(models.Model):
    person = models.ForeignKey(Person, related_name='contacts', on_delete=models.CASCADE)
    contact_type = models.CharField(max_length=50, default="default")
    value = models.CharField(max_length=100, default="default_value")
    is_primary = models.BooleanField(default=False)