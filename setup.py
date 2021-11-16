from distutils.core import setup
from setuptools import Extension,find_packages

setup(name='FEniCS_brainSolver_meshes',
      version='1.0',
      author='Åsmund',  
      author_email='aasmunar@stud.ntnu.no',
      packages = find_packages(),
      package_data={'FEniCS_brainSolver_meshes': ['*.h5', 'meshes/*.h5']},
      include_package_data = True,
      )




