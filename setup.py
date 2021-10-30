"""
Created on 10-29-21 by Mojtaba Heydari
"""


# Local imports
# None.

# Third party imports
# None.

# Python standard library imports
import setuptools
from setuptools import find_packages
import distutils.cmd


# Required packages
REQUIRED_PACKAGES = [
    'numpy',
    'Cython',
    'librosa>=0.8.0',
    'numba==0.48.0', # Manually specified here as librosa incorrectly states that it is compatible with the latest version of numba although 0.50.0 is not compatible. 
    'scipy',
    'mido>=1.2.6',
    'pytest',
    #'pyaudio',
    ##'pyfftw',
    'torch',
    'Matplotlib',
    'BeatNet>=0.0.4',
    'madmom',
]


class MakeReqsCommand(distutils.cmd.Command):
  """A custom command to export requirements to a requirements.txt file."""

  description = 'Export requirements to a requirements.txt file.'
  user_options = []

  def initialize_options(self):
    """Set default values for options."""
    pass

  def finalize_options(self):
    """Post-process options."""
    pass

  def run(self):
    """Run command."""
    with open('./requirements.txt', 'w') as f:
        for req in REQUIRED_PACKAGES:
            f.write(req)
            f.write('\n')



setuptools.setup(
    cmdclass={
        'make_reqs': MakeReqsCommand
    },

    # Package details
    name="jump_reward_inference",
    version="0.0.3",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    # packages=find_packages(),
    include_package_data=True,
    install_requires=REQUIRED_PACKAGES,

    # Metadata to display on PyPI
    author="Mojtaba Heydari",
    author_email="mhydari@ur.rochester.edu",
    description="A package for online music joint rhythmic parameters tracking including beats, downbeats, tempo and meter using the BeatNet AI, a super compact 1D state space and the jump back reward technique",
    keywords="Beat tracking, Downbeat tracking, meter detection, tempo tracking, 1D state space, jump reward technique, efficient state space, ",
    url="https://github.com/mjhydri/1D-StateSpace"


    # CLI - not developed yet
    #entry_points = {
    #    'console_scripts': ['beatnet=beatnet.cli:main']
    #}
)
