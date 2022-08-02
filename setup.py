import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='mltoolbox',
    version='0.0.56',
    author='Faisal Alshargi',
    author_email='falsharg@amazon.com',
    description='Testing installation of Package',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/alshargi/mltoolbox',
    project_urls = {
        "Bug Tracker": "https://github.com/alshargi/mltoolbox/issues"
    },
    license='MIT',
    packages=['mltoolbox'],
    install_requires=['requests'],
)
