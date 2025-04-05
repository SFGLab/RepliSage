from setuptools import setup, find_packages
from pathlib import Path

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pyRepliSage',  # Package name
    version='0.0.1',  # Version of the software
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
        'scipy',
        'mdtraj',
        'seaborn',
        'statsmodels',
        'matplotlib',
        'numpy==1.26.2',
        'pandas',
        'OpenMM',
        'openmm-cuda'
        'scikit-learn',
        'scikit-image',
        'networkx',
        'numba',
        'mat73',
        'hilbertcurve',
        'matplotlib-venn',
        'imageio',
        'imageio-ffmpeg',
        'jupyterlab',
        'jupyter_core',
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