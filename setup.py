import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='lextoolbox',
    version='0.0.1',
    author='Faisal Alshargi',
    author_email='falsharg@amazon.com',
    description='Testing installation of Package',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/Muls/toolbox',
    project_urls = {
        "Bug Tracker": "https://github.com/Muls/toolbox/issues"
    },
    license='MIT',
    packages=['lextoolbox'],
    install_requires=['requests'],
)


for i in range(100):
    print("woow", i)

