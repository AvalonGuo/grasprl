from setuptools import setup, find_packages
setup(
    name='grasprl',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'mujoco',
        'gymnasium',
        'dm-control',
        'opencv-python',
        'torch',
        'torchvision',
        'torchaudio'
    ],
    # Other information like author, author_email, description, etc.
)