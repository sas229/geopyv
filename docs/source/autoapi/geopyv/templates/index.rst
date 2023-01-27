:orphan:

:py:mod:`geopyv.templates`
==========================

.. py:module:: geopyv.templates


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   geopyv.templates.Template
   geopyv.templates.Circle
   geopyv.templates.Square




.. py:class:: Template(size)

   Initialisation of geopyv subset template.

   .. py:method:: mask(centre, mask)

      Method to mask subset based on binary mask from mesh.



.. py:class:: Circle(radius=25)

   Bases: :py:obj:`Template`

   Initialisation of geopyv circular subset template.


.. py:class:: Square(length=50)

   Bases: :py:obj:`Template`

   Initialisation of geopyv square subset template.


