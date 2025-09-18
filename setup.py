# Use setuptools to install the package
from setuptools import setup, find_packages

setup(
    name='dpplgngr',
    packages=find_packages(),
    version='0.2.0',
    description='Deep-learning automated twinning',
    author='Sean Benson',
    license='MIT',
    author_email='s.h.benson@amsterdamumc.nl',  # Add your email
    url='https://github.com/tevien/Doppelganger',  # Add your GitHub repo URL
    install_requires=[],  # Add your dependencies here
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    python_requires='>=3.6',
)
