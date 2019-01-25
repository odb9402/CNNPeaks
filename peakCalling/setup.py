from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

cython_ext_modules = [Extension("readCounts",["readCounts.pyx"]),
            Extension("bedGen",["genBedRows.pyx"])]

c_ext_modules =[Extension("readbam",
                          sources = ["./bamdepth/readbam.c","./bamdepth/htslib/libhts.a"],
                        include_dirs=["./bamdepth/htslib","./bamdepth/htslib/htslib"],
                         library_dirs=["./bamdepth/htslib","./bamdepth/htslib/htslib"],
                         libraries=["z","m","pthread"])]


setup(
        ext_modules = cythonize(cython_ext_modules)
    )
