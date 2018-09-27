from distutils.core import setup, Extension
import os

module1 = Extension('bamread', 
    sources = ['bamread.c'],
    #libraries=['htslib'],
    #ibrary_dirs=['{}/htslib-1.8'.format(os.getcwd())]
    include_dirs=['{}/samtools-1.8'.format(os.getcwd())]
    )

setup(name="bamRead",
    version = '1.0',
    description = '',
    ext_modules = [module1])
