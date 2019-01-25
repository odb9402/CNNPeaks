from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extension = Extension(
    name = "readbam",
    sources=['read_bam.pyx'],
    libraries=["readbam","hts"],
    library_dirs=[".","/usr/local/lib"]#,"./htslib"],
#   include_dirs=[".","./htslib"]
)

setup(
    ext_modules=cythonize([extension])
)