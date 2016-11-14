==================================================================
abnet2: Siamese Neural Network for speech
==================================================================

Free software: GPLv3 license

Python package for using siamese neural network (ABnet) in speech processing.

..
   This is a "long description" file for the package that you are creating.
   If you submit your package to PyPi, this text will be presented on the `public page <http://pypi.python.org/pypi/python_package_boilerplate>`_ of your package.

   Note: This README has to be written using `reStructured Text <http://docutils.sourceforge.net/rst.html>`_, otherwise PyPi won't format it properly.

Installation
------------

We assume you are using a conda virtual environment. First clone the ``abnet2`` package from github::

  git clone git@github.com:bootphon/abnet2.git
  cd abnet2

Then install its dependencies (here using conda, but it works with pip as well)::

  conda install numpy theano lasagne cython

Finally install the ``abnet2`` package itself::

  python setup.py build
  python setup.py install


Usage
-----

See tests/test_neuralnet.py
