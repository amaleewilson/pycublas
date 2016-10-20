from distutils.core import setup, Extension
import numpy.distutils.misc_util

cuda_ext = Extension("_cusgemm",
                     sources=["_cusgemm.c", "cusgemm.c"],
                     libraries=['cublas', 'cudart'])


setup(
    ext_modules=[cuda_ext],
    include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
)
