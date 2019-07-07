from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["pandas", "visdom", "torch", "torchvision", "tqdm",
        "matplotlib", "opencv-python"]

setup(
    name="AlphaPose",
    version="0.1",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/MVIG-SJTU/AlphaPose",
    # packages=['PoseFlow', 'SPPE','yolo', 'AlphaPose'],
    packages = find_packages(),
    install_requires=requirements,
)
