#! /usr/bin/env python
# coding=utf-8

from distutils.core import setup
import os

def read(*parts):
    return open(os.path.join(os.path.dirname(__file__), *parts)).read()

def get_version():
    f = open('Goulib/__init__.py')
    try:
        for line in f:
            if line.startswith('__version__'):
                return eval(line.split('=')[-1])
    finally:
        f.close()

setup(
    name='Goulib',
    version=get_version(),
    description="My Python Library",
    long_description=read('README.rst'),
    keywords='plot, graph, nvd3, d3',
    author='Philippe Guglielmetti',
    author_email='goulib@goulu.net',
    url='http://github.com/goulu/goulib',
    license='LGPL',
    packages=['Goulib'],
    scripts=[],
    install_requires=[
        'setuptools',
    ],
    extras_require = {
        'Excel':  ['xlrd'],
        'dxf2img': ['PIL','dxfgrabber'],
    },
    test_suite="nose.collector",       
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Multimedia :: Graphics :: Graphics Conversion',
    ],
)