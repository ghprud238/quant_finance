from setuptools import setup, find_packages

setup(
    name="nexus-quant",
    version="1.0.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    package_data={"nexus_quant": ["py.typed"]},
    install_requires=[
        "numpy>=1.23.0",
        "pandas>=1.5.0",
        "scipy>=1.9.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
    ],
)
