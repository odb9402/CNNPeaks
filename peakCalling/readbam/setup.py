from distutils.core import setup, Extension

module1 = Extension('bamread', sources = ['bamread.c'])

setup(name="bamReadCount",
    version = '1.0',
    description = '',
    ext_modules = [module1])