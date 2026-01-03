import os
from glob import glob

from setuptools import Extension
from setuptools import setup

amd_source_dir = os.path.join("SuiteSparse", "AMD", "Source")
amd_include_dir = os.path.join("SuiteSparse", "AMD", "Include")

amd_config_dir = os.path.join("SuiteSparse", "SuiteSparse_config")

amd_sources = glob(os.path.join(amd_source_dir, "*.c"))
amd_config_sources = glob(os.path.join(amd_config_dir, "*.c"))

setup(name="amd-lib",
      version="0.1",
      ext_modules=[Extension(name="_amd",
                             sources=['src/_amd.c'] + amd_sources + amd_config_sources,
                             include_dirs=[amd_include_dir, amd_config_dir])])
