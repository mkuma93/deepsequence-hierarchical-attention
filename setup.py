"""
DeepSequence Hierarchical Attention - Setup Configuration
"""

from setuptools import setup, find_packages
import os

# Read README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="deepsequence-hierarchical-attention",
    version="1.0.0",
    author="Mritunjay Kumar",
    author_email="mritunjay.kmr1@gmail.com",
    description="Hierarchical attention framework for time series forecasting with TabNet and intermittent demand support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mkuma93/deepsequence-hierarchical-attention",
    project_urls={
        "Bug Tracker": "https://github.com/mkuma93/deepsequence-hierarchical-attention/issues",
        "Documentation": "https://github.com/mkuma93/deepsequence-hierarchical-attention",
        "Source Code": "https://github.com/mkuma93/deepsequence-hierarchical-attention",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
            "jupyter>=1.0",
        ],
    },
    keywords="time-series forecasting deep-learning attention tabnet intermittent-demand hierarchical-attention",
    include_package_data=True,
    zip_safe=False,
)
