Welcome to whimstan's documentation!
======================================
.. image:: /media/logo_sq.png

whimstan is a package for both fitting GRB afterglow spectra with a hierarchical
Bayesian model and for including a homogeneous WHIM component in that fit. It
implements x-ray gas absorption models in Stan which can be used for purposes
outside of the original intent of searching for the WHIM and modeling the
population of GRB afterglows.

.. image:: /media/spec.png

Additionally, there are sophisticated simulation routines that allow one to
specify population parameters for GRB afterglow and generate synthetic Swift-XRT
data. This is both for validating the included models as well as experimenting
with what different parameters would imply for observations.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   notebooks/simulations.ipynb
   api/API.rst
	     
