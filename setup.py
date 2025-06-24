from setuptools import setup, find_packages

setup(
    name="pyspark_common_utilities",
    version="0.1.1",
    packages=find_packages(),
    install_requires=[
        "pyspark>=3.0.0",
        "pytest"
    ],
    author="Hakim Pocketwalla",
    author_email="hakim.pocketwalla@nearform.com",
    description="Reusable PySpark utility functions for data pipelines",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/nearform/pyspark-utilities",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
