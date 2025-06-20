from setuptools import setup, find_packages

setup(
    name='pyspark_utilities',
    version='0.1.0',
    description='Reusable PySpark utility functions for data pipelines',
    author='Hakim Pocketwalla',
    packages=find_packages(),
    install_requires=[
        'pyspark>=3.0.0',
        'pytest'
    ],
    python_requires='>=3.8',
)
