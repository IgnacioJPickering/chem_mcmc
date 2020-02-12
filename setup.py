from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup_attrs = {
    'name': 'chem_mcmc', 
    'description': 'python implementation of MCMC code', 
    'long_description': long_description,
    'long_description_content_type': "text/markdown",
    'url': 'https://github.com/IgnacioJPickering/chem_mcmc',
    'author': 'Ignacio Pickering',
    'author_email': 'ign.pickering@gmail.com',
    'license': 'MIT',
    'packages': find_packages(),
    'include_package_data': True,
    'use_scm_version': True,
    'setup_requires': ['setuptools_scm'],
    'install_requires': [
        'numpy',
        'matplotlib',
    ],
}

setup(**setup_attrs)
