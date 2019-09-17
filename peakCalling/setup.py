from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

cython_ext_modules = [Extension("readCounts",["readCounts.pyx"],
                                include_dirs=[numpy.get_include()])
                      ,
            Extension("bedGen",["genBedRows.pyx"],
                      include_dirs=[numpy.get_include()])]

c_ext_modules =[Extension("readbam",
                          sources = ["./bamdepth/readbam.c","./bamdepth/htslib/libhts.a"],
                        include_dirs=["./bamdepth/htslib","./bamdepth/htslib/htslib",numpy.get_include()],
                         library_dirs=["./bamdepth/htslib","./bamdepth/htslib/htslib"],
                         libraries=["z","m","pthread"])]


setup(
        ext_modules = cythonize(cython_ext_modules)
    )
