from pathlib import Path

from setuptools import setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="te_algorithms",
    version="2.1.17",
    description="Library supporting analysis of land degradation.",
    long_description=long_description,
    url="https://github.com/ConservationInternational/trends.earth-algorithms",
    author="Conservation International",
    author_email="trends.earth@conservation.org",
    license="MIT",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "OSI Approved :: GNU General Public License (GPL)"
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    keywords="land degradation LDN SDG sustainable development goals",
    packages=[
        "te_algorithms",
        "te_algorithms.common",
        "te_algorithms.gdal.land_deg",
        "te_algorithms.gdal",
        "te_algorithms.gee",
        "te_algorithms.api",
    ],
    package_dir={"te_algorithms": "te_algorithms"},
    package_data={"te_algorithms": ["version.json", "data/*"]},
    include_package_data=True,
    install_requires=[
        "openpyxl>=3.1.3",
        "backoff>=2.1.0",
        "marshmallow>=3.21.3,<4.0.0",
        "marshmallow-dataclass>=8.7.1",
        "defusedxml>=0.7.1",
        "te_schemas @ git+https://github.com/ConservationInternational/trends.earth-schemas.git@master",
    ],
    extras_require={
        "api": ["boto3>=1.16", "GDAL>=3.0.0"],
        "gee": ["earthengine-api==0.1.232"],
        "gdal": ["GDAL>=3.0.0", "numpy>=1.17.0"],
        "numba": ["numba>=0.54.1"],
        "dev": ["check-manifest"],
        "test": ["coverage", "pytest"],
        "docs": ["sphinx", "sphinx_rtd_theme"],
    },
)
