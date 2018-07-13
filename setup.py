import setuptools

with open("README.md", 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name="apriltag_pose",
    version="0.0.1",
    author="Jake Olkin",
    description="A package to reconstruct the pose of a camera relative to an apriltag",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages()

)