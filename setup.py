"""
Setup file for the Transformative Media Recommendation System
"""

from setuptools import setup, find_packages

with open("PROJECT_README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="recom-system",
    version="1.0.0",
    author="Transformative Media Project",
    description="Comprehensive recommendation system for transformative media",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "networkx>=3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=23.0",
            "flake8>=6.0",
        ],
        "ml": [
            "scikit-learn>=1.3.0",
            "scipy>=1.10.0",
        ],
        "data": [
            "pandas>=2.0.0",
        ]
    },
)
