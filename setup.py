from setuptools import setup, find_packages

setup(
    name='simdb',
    version='0.3.0',
    packages=find_packages(),
    description='A simple Python package for management of simulation datasets',
    #long_description=long_description,
    url='https://github.com/sdrave/simdb',
    author='Stephan Rave',
    author_email='mail@stephanrave.de',
    license='BSD License',
    classifiers=[
        'Development Status :: 4 - Beta'
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries',
        'Topic :: Database',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    keywords='simulation data numerics reproducibility',
    install_requires = ['pyyaml', 'numpy'],
)
