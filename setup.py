"""
Set up YAMLChem package.
"""

from setuptools import find_packages, setup
from urllib.parse import urljoin


PACKAGE_URL = 'http://github.com/sanjaradylov/yamlchem/'


setup(
    name='yamlchem',
    description='Yet Another Machine Learning package for Chemistry',
    version='0.1',
    author='Sanjar Ad[iy]lov',
    url=PACKAGE_URL,
    project_urls={
        # 'Documentation': urljoin(PACKAGE_URL, 'wiki'),
        'Source Code': urljoin(PACKAGE_URL, 'tree/master/moleculegen'),
    },
    packages=find_packages(exclude=['*tests']),
    include_package_data=False,
    install_requires=[
        'mxnet',
    ],
)
