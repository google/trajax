# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from setuptools import find_packages
from setuptools import setup


root_path = os.path.dirname(__file__)
version_path = os.path.join(root_path, 'trajax', 'version.py')

_dct = {}
with open(version_path) as f:
  exec(f.read(), _dct)
__version__ = _dct['__version__']

req_path = os.path.join(root_path, 'requirements.txt')
install_requires = []
if os.path.exists(req_path):
  with open(req_path) as fp:
    install_requires = [line.strip() for line in fp]

setup(
    name='trajax',
    version=__version__,
    description='Accelerated, batchable, differentiable trajectory optimization with JAX.',
    author='Trajax authors',
    author_email='no-reply@google.com',
    url='https://github.com/google/trajax',
    license='Apache 2.0',
    packages=find_packages(),
    package_data={},
    install_requires=install_requires,
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    keywords='optimization, control, trajectory optimization, automatic differentiation, jax',
    requires_python='>=3.7',
)
