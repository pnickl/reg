from setuptools import setup

setup(name='reg',
      version='0.0.1',
      description='Regression',
      author='Hany Abdulsamad',
      author_email='hany@robot-learning.de',
      install_requires=['numpy', 'scipy', 'matplotlib',
                        'torch', 'sklearn', 'autograd',
                        'mimo', 'gpytorch'],
      packages=['reg'],
      zip_safe=False,
)