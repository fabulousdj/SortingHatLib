from setuptools import setup

setup(
    # Needed to silence warnings
    name='Measurements',
    url='https://github.com/pvn25/ML-Data-Prep-Zoo/tree/master/ML%20Schema%20Inference',
    author='Vraj Shah',
    author_email='vps002@eng.ucsd.edu',
    # Needed to actually package something
    packages=['sortinghat'],
    # Needed for dependencies
    install_requires=['numpy'],
    # *strongly* suggested for sharing
    version='0.5',
    license='MIT',
    # We will also need a readme eventually (there will be a warning)
    long_description=open('README.rst').read(),
    # if there are any scripts
    scripts=['scripts/test.py'],
)