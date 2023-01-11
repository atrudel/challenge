from setuptools import setup, find_packages

with open("requirements.txt") as requirement_file:
    requirements = requirement_file.read().split()

setup(
    name='challenge',
    version='0.1.0',
    description='Code for challenge',
    author='Amric Trudel',
    url='https://github.com/atrudel/challenge',
    python_requires='>=3.7',
    install_requires=requirements,
    packages=find_packages()
)