from __future__ import absolute_import, division, print_function
from os.path import join as pjoin

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 1
_version_micro = ''  # use '' for first of series, number for 1 and above
# _version_extra = 'dev'
_version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering",
               "Topic :: Multimedia :: Sound/Audio :: Analysis",
               "Topic :: Multimedia :: Sound/Audio"]

# Description should be a one-liner:
description = "loopextractor: an implementation of a source separation technique for extracting loops from music"
# Long description will go up on the pypi page
long_description = """

loopextractor
========
loopextractor is an implementation of a source separation technique that uses
the non-negative Tucker decomposition to model an audio file that has been
re-shaped into a 3D spectrum.

It is based on research led by Jordan B. L. Smith that was published at ICASSP
in 2018: https://github.com/jblsmith/icassp2018

Although the code for loopextractor follows the same steps described in that
ICASSP paper, this code was written from scratch in December 2019 by Jordan
Smith alone.

Outside of this, the code for loopextractor has no relationship or connection
to work done at AIST, nor to the code that powers the Unmixer website.
"""

NAME = "loopextractor"
MAINTAINER = "Jordan B. L. Smith"
MAINTAINER_EMAIL = "jblsmith@gmail.com"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "http://github.com/jblsmith/loopextractor"
DOWNLOAD_URL = ""
LICENSE = "LGPLv3"
AUTHOR = "Jordan B. L. Smith"
AUTHOR_EMAIL = "jblsmith@gmail.com"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGE_DATA = {'loopextractor': [pjoin('data', '*')]}
REQUIRES = ["numpy", 'madmom', 'librosa', 'tensorly']
