from setuptools import setup

setup(
    # Needed to silence warnings
    name='Measurements',
    url='https://github.com/pvn25/SortingHatLib',
    author='Vraj Shah',
    author_email='pvn251@gmail.cin',
    # Needed to actually package something
    packages=['measure'],
    # Needed for dependencies
    install_requires=['numpy'],
    # *strongly* suggested for sharing
    version='0.5',
    license='MIT',
    description='An example of a python package from pre-existing code',
    # We will also need a readme eventually (there will be a warning)
#     long_description=open('README.rst').read(),
    # if there are any scripts
    scripts=['scripts/hello.py'],
)
