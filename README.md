XICSRT: Contributed Modules
===========================
 
Git Repository: https://bitbucket.org/amicitas/xicsrt_contrib  
Git Mirror: https://github.com/PrincetonUniversity/xicsrt_contrib

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

    pip install xicsrt_contrib

Alternatively it is possible to install from source using `setuptools`

    python setup.py install

After installation xicsrt should natively see all available contributed objects
in the same way as built-in objects.

Usage
-----

Use xicsrt normally, but now these contributed objects are also available.
