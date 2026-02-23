from setuptools import setup, find_packages


setup(name='PTCML',
      version='1.0.0',
      description='Patch-based Tendency Camera Multi-Constraint Learning for Unsupervised Person Re-identification',
      author='',
      author_email='',
      url='',
      install_requires=[
          'numpy', 'torch', 'torchvision',
          'six', 'h5py', 'Pillow', 'scipy',
          'scikit-learn', 'metric-learn', 'faiss_gpu==1.6.3'],
      packages=find_packages()
      )
