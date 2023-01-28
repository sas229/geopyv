:py:mod:`geopyv.log`
====================

.. py:module:: geopyv.log


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   geopyv.log.CustomFormatter



Functions
~~~~~~~~~

.. autoapisummary::

   geopyv.log.initialise
   geopyv.log.set_level



.. py:class:: CustomFormatter(fmt, fmt_INFO)

   Bases: :py:obj:`logging.Formatter`

   Override to format the std log output such that INFO logs provide
   just the log message and other levels also provide the level type.

   :param fmt: Format of general log message.
   :type fmt: str
   :param fmt_INFO: Modified format of logging.INFO message.
   :type fmt_INFO: str

   .. py:method:: format(record)

      Override to provide a custom log format.

      :param record: Log record object.
      :type record: LogRecord



.. py:function:: initialise(level)

   Function to initialise the log file.

   :param level: Log level. Options include logging.VERBOSE, logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR and logging.FATAL. Defaults to logging.INFO.
   :type level: logging.level


.. py:function:: set_level(level)

   Function to set the log level after initialisation.

   :param level: Log level. Options include logging.VERBOSE, logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR and logging.FATAL. Defaults to logging.INFO.
   :type level: logging.level


