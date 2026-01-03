import os
from glob import glob

from setuptools import Extension
from setuptools import setup

amd_source_dir = os.path.join("SuiteSparse", "AMD", "Source")
amd_include_dir = os.path.join("SuiteSparse", "AMD", "Include")

amd_config_dir = os.path.join("SuiteSparse", "SuiteSparse_config")

amd_sources = glob(os.path.join(amd_source_dir, "*.c"))
amd_config_sources = glob(os.path.join(amd_config_dir, "*.c"))

setup(version="0.1.0",
      ext_modules=[Extension(name="suitesparse_amd._amd",
                             sources=['src/suitesparse_amd/_amd.c'] + amd_sources + amd_config_sources,
                             include_dirs=[amd_include_dir, amd_config_dir],
                             language="c",
                             extra_compile_args=["-O3"], )])
