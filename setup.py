from setuptools import setup, find_packages

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='vimg',
      version='0.4.0',
      description='An image viewer for the command line',
      long_description=readme(),
      keywords='cli terminal console image picture graphics viewer preview',
      url='http://github.com/jnhansen/vimg',
      author='Johannes Hansen',
      author_email='johannes.niklas.hansen@gmail.com',
      license='MIT',
      classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Environment :: Console :: Curses',
        'Environment :: MacOS X',
        'Intended Audience :: Developers',
        'Intended Audience :: System Administrators',
        'License :: OSI Approved :: MIT License',
        'Topic :: Multimedia :: Graphics',
        'Topic :: Multimedia :: Graphics :: Viewers',
        'Topic :: Utilities'
      ],
      packages=find_packages(),
      install_requires=[
        'numpy',
        'scipy',
        'opencv-python',
      ],
      entry_points={
          'console_scripts': ['vimg=vimg.main:main'],
      },
      include_package_data=True,
      zip_safe=False)
