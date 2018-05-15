from setuptools import setup, find_packages

setup_requires = []

install_requires = []

dependency_links =[]

setup(
        name = 'CNNpeaks',
        version='0.1',
        author='dongpin',
        author_email='dhehdqls@gmail.com',
        packages=find_packages(),
        install_requires=install_requires,
        dependency_links=dependency_links,
        scripts=['CNNpeaks'],
        )
