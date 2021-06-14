XICSRT: Contributed Modules
===========================

Git Repository: https://bitbucket.org/amicitas/xicsrt_contrib
Git Mirror: https://github.com/PrincetonUniversity/xicsrt_contrib

| **Git Repository:** `bitbucket.org/amicitas/xicsrt_contrib`_
| **Git Mirror:** `github.com/PrincetonUniversity/xicsrt_contrib`_

Purpose
-------
A collection of community contributed modules for extension of the XICSRT
raytracing code.

The optic, source and filter objects in this repository are meant for use with
XICSRT. These are extra objects that may be useful to an XICSRT user but have
not been included as built-in objects for one of the following reasons:

- Usage is too specific for general use.
- Non-standard external dependencies.
- Performance and stability not at production quality.

Some of these objects may eventually be moved into the main repository as their
development advances.


Installation
------------
The XICSRT contributed modules can be simply installed using `pip`

.. code:: bash

    pip install xicsrt_contrib

Alternatively it is possible to install from source using `setuptools`

.. code:: bash

    python setup.py install

After installation xicsrt should natively see all available contributed objects
in the same way as built-in objects.


Usage
-----
Use `xicsrt` normally; after installation the contributed objects are natively available.


.. _bitbucket.org/amicitas/xicsrt_contrib: https://bitbucket.org/amicitas/xicsrt_contrib
.. _github.com/PrincetonUniversity/xicsrt_contrib: https://github.com/PrincetonUniversity/xicsrt_contrib


License
-------
.. include:: ../LICENSE

.. toctree::
    :maxdepth: 1
    :caption: Contents:

    apidoc/xicsrt_contrib

Indices and tables:
-------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _xicsrt_contrib.readthedocs.org: https://xicsrt_contrib.readthedocs.org
.. _bitbucket.org/amicitas/xicsrt: https://bitbucket.org/amicitas/xicsrt
.. _github.com/PrincetonUniversity/xicsrt: https://github.com/PrincetonUniversity/xicsrt
.. _jupyter: https://jupyter.org/
.. _numpy: https://numpy.org/
.. _matplotlib: https://matplotlib.org/
.. _plotly: https://github.com/plotly
