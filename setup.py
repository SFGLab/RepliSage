from setuptools import setup, find_packages
from pathlib import Path

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pyRepliSage',  # Package name
    version='0.0.8',  # Version of the software
    description='A stochastic model for the modeling of DNA replication and cell-cycle dynamics.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Sebastian Korsak',
    author_email='s.korsak@datascience.edu.pl',
    url='https://github.com/SFGLab/RepliSage',  # GitHub repository URL
    license='GNU General Public License v3.0',
    packages=find_packages(include=['RepliSage', 'RepliSage.*']),
    include_package_data=True,
    package_data={
    'RepliSage': ['forcefields/*'],
    },
    install_requires=[  # List your package dependencies here
        'scipy>=1.11.4',
        'mdtraj>=1.9.9',
        'seaborn>=0.13.0',
        'statsmodels>=0.14.0',
        'matplotlib>=3.8.2',
        'numpy==1.26.2',
        'pandas>=2.1.3',
        'OpenMM>=8.1.1',
        'openmm-cuda>=8.1.1',
        'scikit-learn>=1.5.2',
        'scikit-image>=0.24.0',
        'networkx>=3.4.2',
        'numba>=0.60.0',
        'mat73>=0.65',
        'hilbertcurve>=2.0.5',
        'matplotlib-venn>=1.1.1',
        'imageio>=2.36.0',
        'imageio-ffmpeg>=0.5.1',
        'jupyterlab>=4.2.5',
        'jupyter_core>=5.7.2',
        'tqdm',
    ],
    entry_points={
        'console_scripts': [
            'replisage=RepliSage.run:main',  # loopsage command points to run.py's main function
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: POSIX :: Linux',  # General OS classifier
    ],
    python_requires='>=3.10',  # Specify Python version compatibility
)