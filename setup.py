from setuptools import setup

setup(
    name='SortingHat Library',
    url='https://github.com/pvn25/SortingHatLib',
    author='Vraj Shah, Tanay Karve',
    author_email='pvn251@gmail.com, tanaykarve@gmail.com',
    packages=['sortinghat'],
    install_requires=['numpy','gdown'],
    version='0.5',
    license='MIT',
    description='An example of a python package from pre-existing code',
    scripts=['scripts/hello.py'],
)
