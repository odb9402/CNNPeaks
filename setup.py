from setuptools import setup, find_packages

setup_requires = ['samtools', 'bamtools']

install_requires = ['pysam','scipy','numpy','sklearn','tensorflow-gpu','pandas','progressbar2']

dependency_links =[]

setup(
        name = 'CNNpeaks',
        version='1.0',
        author='dongpin',
        author_email='dhehdqls@gmail.com',
        packages=find_packages(),
        install_requires=install_requires,
        dependency_links=dependency_links,
        scripts=['CNNpeaks'],
        )
