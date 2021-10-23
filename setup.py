# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='te_algorithms',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='1.0.9',

    description='A python package to facilitate analyzing remotely-sensed datasets from GEE in support of monitoring land degradation.',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/ConservationInternational/trends.earth-algorithms',

    # Author details
    author='Conservation International',
    author_email='trends.earth@conservation.org',

    # Choose your license
    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: GIS',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],

    # What does your project relate to?
    keywords='land degradation LDN SDG sustainable development goals',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=['te_algorithms'],
    package_dir={'te_algorithms': 'te_algorithms'},
    package_data={'te_algorithms': ['version.json']},
    include_package_data=True,

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=[
        'marshmallow>=3.14.0',
        'marshmallow-dataclass>=8.5.3',
        'te_schemas @ git+https://github.com/ConservationInternational/trends.earth-schemas.git@develop'
    ],
    
    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={
        'gee': [
            'earthengine-api==0.1.232'
        ],
        'gdal': [
            'GDAL>=3.0.0',
            'numpy>=3.0.0'
        ],
        'dev': ['check-manifest'],
        'test': ['coverage'],
    }
)
