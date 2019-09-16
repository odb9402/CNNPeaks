from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extension = Extension(
    name = "readbam",
    sources=['read_bam.pyx'],
    libraries=["readbam","hts"],
    library_dirs=[".","/usr/local/lib"]#,"./htslib"],
    include_dirs=[".","./htslib",numpy.get_include()]
)


setup(
    ext_modules=cythonize([extension])
)