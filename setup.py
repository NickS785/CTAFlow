"""
Setup script for CTAFlow package.
"""
import os
from setuptools import setup, find_packages

# Read the README file
def read_readme():
    """Read README.md file if it exists, otherwise return basic description"""
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "CTA positioning prediction system using COT data and technical analysis"

# Read requirements
def read_requirements():
    """Read requirements.txt file"""
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="CTAFlow",
    version="1.0.0",
    author="CTA Research Team",
    author_email="research@example.com",
    description="CTA positioning prediction system using COT data and technical analysis",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    py_modules=["config", "main"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.900",
        ],
    },
    include_package_data=True,
    package_data={
        '': ['*.toml', '*.md'],
    },
    entry_points={
        'console_scripts': [
            'ctaflow=main:main',
        ],
    },
    keywords="cta commodity trading advisor machine learning finance forecasting",
    project_urls={
        "Source": "https://github.com/yourusername/CTAFlow",
        "Documentation": "https://ctaflow.readthedocs.io/",
        "Bug Reports": "https://github.com/yourusername/CTAFlow/issues",
    },
)