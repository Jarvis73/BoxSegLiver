import sys
sys.argv.extend("build_ext --inplace".split())

from distutils.core import setup
from Cython.Distutils import build_ext
from distutils.extension import Extension
import numpy as np


# Obtain the numpy include directory.  This logic works across numpy versions.
numpy_include = np.get_include()

ext_modules = [
    Extension(
        "Networks.layers.cython_bbox",
        ["Networks/layers/bbox.pyx"],
        extra_compile_args={'gcc': ["-Wno-cpp", "-Wno-unused-function"]},
        include_dirs=[numpy_include]
    )
]

setup(
    name='MedicalImageSegmentation',
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
)
