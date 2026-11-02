import re
from pathlib import Path
from setuptools import setup, find_packages

here = Path(__file__).parent
long_description = (here / 'README.md').read_text(encoding='utf-8')
version = re.search(r'__version__ = "(.+?)"', (here / 'src' / '_version.py').read_text()).group(1)

# the code lives in src/ and is installed as the package `RepliSage`
subpackages = find_packages('src')

setup(
    name='pyRepliSage',
    version=version,
    description='A stochastic model for the modeling of DNA replication and cell-cycle dynamics.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Sebastian Korsak',
    author_email='s.korsak@datascience.edu.pl',
    url='https://github.com/SFGLab/RepliSage',
    license='GNU General Public License v3.0',
    package_dir={'RepliSage': 'src'},
    packages=['RepliSage'] + [f'RepliSage.{p}' for p in subpackages],
    include_package_data=True,
    package_data={'RepliSage': ['forcefields/*', 'data/*', 'config.ini', 'viz3d/assets/*', 'viz3d/vendor/*']},
    install_requires=[
        'scipy>=1.11.4', 'mdtraj>=1.9.9', 'seaborn>=0.13.0', 'statsmodels>=0.14.0', 'matplotlib>=3.8.2',
        'numpy>1.2,<2.0', 'pandas>=2.1.3', 'OpenMM>=8.1.1', 'openmm-cuda>=8.1.1', 'scikit-learn>=1.5.2',
        'scikit-image>=0.24.0', 'networkx>=3.4.2', 'numba>=0.60.0', 'hilbertcurve>=2.0.5',
        'matplotlib-venn>=1.1.1', 'imageio>=2.36.0', 'imageio-ffmpeg>=0.5.1', 'jupyterlab>=4.2.5',
        'jupyter_core>=5.7.2', 'tqdm', 'pyarrow', 'fastparquet', 'pyBigWig>=0.3.19',
    ],
    extras_require={'viz': ['pyvista'], 'test': ['pytest'], 'docs': ['mkdocs-material>=9.5']},
    project_urls={'Documentation': 'https://sfglab.github.io/RepliSage/', 'Source': 'https://github.com/SFGLab/RepliSage'},
    entry_points={'console_scripts': ['replisage=RepliSage.run:main']},
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: POSIX :: Linux',
    ],
    python_requires='>=3.10',
)
