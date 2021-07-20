#nsml: nsml/ml:cuda10.1-cudnn7-pytorch1.3keras2.3
from distutils.core import setup
setup(
    name='ladder_networks',
    version='1.0',
    install_requires=[
        'matplotlib',
        'pillow',
        'fastprogress',
        'attrdict==2.0.1',
        'tokenizers==0.7.0',
        'tqdm==4.46.1',
        'transformers==2.11.0'
]
)