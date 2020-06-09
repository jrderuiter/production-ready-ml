import setuptools


setuptools.setup(
    name="titanic",
    version="0.1",
    description="My titanic model.",
    url="https://github.com/pypa/sampleproject",
    author="A. Random Developer",
    author_email="author@example.com",
    package_dir={"": "src"},
    packages=setuptools.find_packages("src"),
    python_requires=">=3.5, <4",
    install_requires=[
        "pandas>=0.23.1",
        "scikit-learn>=0.21.0",
        "joblib",
        "click",
        "flask",
    ],
    extras_require={"dev": ["pep517", "pytest", "pylint", "black"],},
    entry_points={"console_scripts": ["titanic=titanic.cli:cli"]},
)
