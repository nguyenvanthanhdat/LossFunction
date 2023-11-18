from setuptools import setup, find_packages
import sys

with open('README.md', encoding='utf-8') as f:
    readme = f.read()

# with open('LICENSE', encoding='utf-8') as f:
#     license = f.read()

with open('requirements.txt', encoding='utf-8') as f:
    reqs = f.readlines()[1:]

setup(
    name = 'loss_nli',
    version='0.1.0',
    description='Empirical study of loss functions for inference methods based on language models',
    long_description='readme',
    python_requires='>=3.9',
    package_dir={"": "src"},
    packages=find_packages(exclude=('data')),
    # install_requires=reqs.strip().split('\n'),
    install_requires=reqs,
)