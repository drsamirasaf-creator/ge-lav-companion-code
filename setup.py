"""Setup script for the gelav companion code package."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="gelav",
    version="0.1.0",
    author="Samir Asaf",
    author_email="drsamirasaf@gmail.com",
    description="Companion code for the GE-LAV graduate finance course",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/drsamirasaf-creator/ge-lav-companion-code",
    packages=find_packages(exclude=["tests", "notebooks"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Intended Audience :: Education",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24,<2.1",
        "scipy>=1.11,<1.14",
        "pandas>=2.0,<2.3",
        "matplotlib>=3.7,<3.10",
    ],
)
