import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="featureExplorePlus",
    version="0.0.5.0",
    author="Zhong Xiaolin",
    author_email="xlzhong123@163.com",
    description="Feature exploration for supervised learning. This version based on Abhay's featexp, "
                "add user define bins which base on decision tree,"
                "and add the origin distribution plot function based on seaborn histplot"
                "add psi calculate function"
                "and add some useful pandas tools",
    long_description="FeatureExplorePlus helps with feature understanding, feature debugging, leakage detection, "
                     "finding noisy features and model monitoring",
    long_description_content_type="text/markdown",
    url="https://github.com/XiaolinZHONG/FeatureExplorePlus.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['pandas>=0.23.4', 'numpy>=1.15.4', 'matplotlib>=3.0.2', 'scikit-learn>=0.19.0', 'seaborn>=0.8']

)
