from setuptools import setup, find_packages

setup(
    name='FlashSRInfer',
    version='0.1',
    package_dir={'': './'},
    packages=find_packages( where = './'),
    install_requires=[
        'numpy',
        'tqdm',
        'psutil',
        'pyyaml',
        'matplotlib',
        'librosa',
        'wandb',
        'tensorboardX',
        'einops'
    ],
    # additional metadata about your project
    author='Jaekwon Im',
    author_email='jakeoneijk@kaist.ac.kr',
    description='',
    license='',
    keywords='',
)