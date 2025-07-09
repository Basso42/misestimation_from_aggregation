"""Setup script for misestimation_from_aggregation package."""

from setuptools import setup, find_packages, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
import numpy

# Define C++ extension modules
cpp_extensions = [
    Pybind11Extension(
        "misestimation_from_aggregation._cpp_core",
        [
            "misestimation_from_aggregation/cpp/similarity_ops.cpp",
            "misestimation_from_aggregation/cpp/matrix_ops.cpp", 
            "misestimation_from_aggregation/cpp/network_ops.cpp",
            "misestimation_from_aggregation/cpp/python_bindings.cpp",
        ],
        include_dirs=[
            # Path to pybind11 headers
            pybind11.get_include(),
            # Path to numpy headers
            numpy.get_include(),
            # Local include directory
            "misestimation_from_aggregation/cpp/include",
        ],
        language='c++',
        cxx_std=17,
    ),
]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="misestimation_from_aggregation",
    version="0.1.0",
    author="Christoph Diem, András Borsos, Tobias Reisch, János Kertész, Stefan Thurner",
    author_email="",
    description="Estimating the loss of economic predictability from aggregating firm-level production networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Basso42/misestimation_from_aggregation",
    packages=find_packages(),
    ext_modules=cpp_extensions,
    cmdclass={"build_ext": build_ext},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "networkx>=2.6",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "jupyter>=1.0.0",
        "numba>=0.56.0",
        "pybind11>=2.6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.12",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.910",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)