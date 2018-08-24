from setuptools import setup

setup(name='cochelp',
      version='0.1',
      description='Utilities for cochlea events reading, localization and separation',
      url='https://github.com/SensorsAudioINI/cochlea_utils',
      author='Jithendar Anumula (UZH/ETH Zurich)',
      author_email='anumulaj@ini.uzh.ch',
      license='MIT',
      packages=['cochelp'],
      install_requires=[
            'scipy',
            'matplotlib',
            'progressbar'
      ],
      extras_require={
      },
      zip_safe=False,
      )
