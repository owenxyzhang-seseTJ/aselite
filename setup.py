"""ASE-lite 安装脚本"""

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent
long_description = (ROOT / 'README.md').read_text(encoding='utf-8')

setup(
    name='aselite',
    version='0.1.0',
    description='轻量原子模拟环境库：容错 CIF 解析、向量化几何计算，仅依赖 NumPy',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='owenxyzhang-seseTJ',
    url='https://github.com/owenxyzhang-seseTJ/aselite',
    project_urls={
        'Source': 'https://github.com/owenxyzhang-seseTJ/aselite',
        'Issues': 'https://github.com/owenxyzhang-seseTJ/aselite/issues',
    },
    license='MIT',
    license_files=['LICENSE'],
    packages=find_packages(include=['aselite', 'aselite.*']),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
    install_requires=['numpy>=1.20'],
    extras_require={'dev': ['pytest', 'build', 'twine']},
    keywords='atoms cif molecular-structure crystallography ase numpy',
)
