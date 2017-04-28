from setuptools import setup

setup(
    name='actlearn',
    version='0.1',
    description='Implements learning methods.',
    url='https://github.com/bnjmacdonald/active-learner',
    author='Bobbie NJ Macdonald',
    author_email='bnjmacdonald@gmail.com',
    license='MIT',
    packages=['actlearn'],
    install_requires=[
        'sklearn',
        'numpy',
      ],
    zip_safe=False
)