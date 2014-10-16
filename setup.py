from setuptools import setup, find_packages

setup(
    name='simdb',
    version='0.1.dev1',
    packages=find_packages(),
    description='A simple Python package for management of simulation datasets',
    #long_description=long_description,
    url='https://github.com/sdrave/simdb',
    author='Stephan Rave',
    author_email='mail@stephanrave.de',
    license='BSD License',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries',
        'Topic :: Database',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
    ],
    keywords='simulation data numerics reproducibility',
    install_requires = ['pyyaml', 'sh'],
)
