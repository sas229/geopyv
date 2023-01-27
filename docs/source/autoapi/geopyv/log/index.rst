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

   Initialize the formatter with specified format strings.

   Initialize the formatter either with the specified format string, or a
   default as described above. Allow for specialized date formatting with
   the optional datefmt argument. If datefmt is omitted, you get an
   ISO8601-like (or RFC 3339-like) format.

   Use a style parameter of '%', '{' or '$' to specify that you want to
   use one of %-formatting, :meth:`str.format` (``{}``) formatting or
   :class:`string.Template` formatting in your format string.

   .. versionchanged:: 3.2
      Added the ``style`` parameter.

   .. py:method:: format(record)

      Format the specified record as text.

      The record's attribute dictionary is used as the operand to a
      string formatting operation which yields the returned string.
      Before formatting the dictionary, a couple of preparatory steps
      are carried out. The message attribute of the record is computed
      using LogRecord.getMessage(). If the formatting string uses the
      time (as determined by a call to usesTime(), formatTime() is
      called to format the event time. If there is exception information,
      it is formatted using formatException() and appended to the message.


   .. py:attribute:: white
      :value: '\x1b[37m'

      

   .. py:attribute:: yellow
      :value: '\x1b[38;5;226m'

      

   .. py:attribute:: red
      :value: '\x1b[38;5;196m'

      

   .. py:attribute:: bold_red
      :value: '\x1b[31;1m'

      

   .. py:attribute:: reset
      :value: '\x1b[0m'

      


.. py:function:: initialise(level)

   Function to initialise the log file.


.. py:function:: set_level(level)

   Function to set the log level after initialisation.


