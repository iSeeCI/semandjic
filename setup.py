# setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="semandjic",
    version="0.3.0",
    author="Andres Fernandez",
    author_email="andres.fernandez@iseeci.com",
    description="A Django app for semantic model relationships and nested forms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/semandjic",
    project_urls={
        "Bug Tracker": "https://github.com/iSeeCI/semandjic/issues",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Web Environment",
        "Framework :: Django",
        "Framework :: Django :: 3.2",
        "Framework :: Django :: 4.0",
        "Framework :: Django :: 4.1",
        "Framework :: Django :: 4.2",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Internet :: WWW/HTTP",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    packages=find_packages(exclude=["tests*", "docs*"]),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "Django>=3.2",
    ],
    extras_require={
        'test': [
            'pytest>=7.0.0',
            'pytest-django>=4.5.0',
            'pytest-cov>=4.0.0',
            'coverage>=7.0.0',
        ],
        'dev': [
            'black',
            'isort',
            'flake8',
            'mypy',
            'twine',
            'build',
        ],
    },
    zip_safe=False,
)