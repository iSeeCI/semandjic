#!/usr/bin/env python
import os
import sys
import django
import pytest

def run_tests():
    os.environ['DJANGO_SETTINGS_MODULE'] = 'semandjic.tests.settings'
    django.setup()

    # Pass command-line arguments to pytest
    exit_code = pytest.main(sys.argv[1:])

    sys.exit(exit_code)

if __name__ == "__main__":
    run_tests()
