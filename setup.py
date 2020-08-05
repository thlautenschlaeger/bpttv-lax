import os
from distutils.core import setup
from distutils.extension import Extension

import numpy as np


setup(name='lax',
      version='0.0.1',
      description='Backpropagation through the void LAX',
      author='Thomas Lautenschlaeger',
      author_email='th.la@me.com',
      install_requires=['numpy', 'scipy', 'matplotlib', 'sklearn', 'pytorch'],
      packages=['lax'],
      include_dirs=[np.get_include()],)
