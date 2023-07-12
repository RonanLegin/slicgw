from setuptools import setup, find_packages

setup(
    name="slicgw",
    version="0.1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    author='Ronan Legin, Kaze Wong, Maximiliano Isi',
    description='Score based likelihood characterization (SLIC) for gravitational waves.',
)
