"""Package setup for OSL-SignX."""
from setuptools import setup, find_packages

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="osl-signx",
    version="0.1.0",
    description="SignX architecture for Omani Sign Language continuous recognition",
    author="OSL-SignX Team",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=requirements,
)
