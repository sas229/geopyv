:py:mod:`geopyv.io`
===================

.. py:module:: geopyv.io

.. autoapi-nested-parse::

   IO module for geopyv.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   geopyv.io.load
   geopyv.io.save



.. py:function:: load(filename=None)

   Function to load a geopyv data object into the workspace. If no filename
    is provided, the host OS default file browser will be used to allow the
     user to select a geopyv data file with .pyv extension.


   :param filename: Name of file containing a geopy data object.
   :type filename: str, optional

   :returns: **object** -- The geopyv data object loaded.
   :rtype: geopyv.object

   .. note::
       * Any .pyv object can be loaded with this function.
       * The data object will be loaded into a `ObjectResults` instance where
         `Object` represents the instance type that generated the data. For example,
         data from a `Subset` instance will be loaded into a `SubsetResults` instance.



.. py:function:: save(object, filename=None)

   Function to save data from a geopyv object. If no filename is
   provided, the host OS default file browser will be used to allow
   the user to choose a filename and storage location.


   :param object: The object to be saved.
   :type object: geopyv.object
   :param filename: The filename to give to the saved data file.
   :type filename: str, optional

   .. note::
       * Any geopyv object can be passed to this function.
       * Do not include the .pyv extension in the `filename` argument.



