import logging
import numpy as np
import os

log = logging.getLogger(__name__)


def _check_type(parameter, name, types):
    """

    Standard function for checking types of input parameter.

    Parameters
    ----------
    parameter :
        The input to check.
    types:
        The types expected.

    Returns
    -------
    msg :

    """

    if type(parameter) not in types:
        if len(types) > 1:
            type_str = ", ".join([val.__name__ for val in types])
            splitpoint = type_str.rfind(",")
            type_str = type_str[:splitpoint] + " or" + type_str[splitpoint + 1 :]
        else:
            type_str = types[0]
        return (
            "`{name}` kwarg type invalid. "
            "Expected a {types}, but got a {parameter_type}."
        ).format(
            name=name,
            types=type_str,
            parameter_type=type(parameter).__name__,
        )


def _check_index(index, name, axis, array):
    """

    Private function for checking input index against array bounds.

    Parameters
    ----------
    index: int
        The input index.
    name:
        The parameter name.
    axis:
        The axis to check for.
    array:
        The array to check for.

    Returns:
    None if check passed.
    Error message if check failed.
    """

    if index < 0 or index >= np.shape(array)[axis]:
        return (
            "`{name}` {value} is out of bounds " "for axis {axis} with size {maximum}"
        ).format(
            name=name,
            value=index,
            axis=axis,
            maximum=np.shape(array)[axis],
        )


def _check_value(parameter, name, values):
    """

    Private function for checking parameter value.

    Parameters
    ----------
    parameter :
    name :
    values :
    """

    if parameter not in values:
        if len(values) > 1:
            value_str = ", ".join(values)
            splitpoint = value_str.rfind(",")
            value_str = value_str[:splitpoint] + " or" + value_str[splitpoint + 1 :]
        else:
            value_str = values[0]
        return (
            "`{name}` kwarg value invalid. "
            "Expected a value from {values}, but got {value}."
        ).format(
            name=name,
            values=value_str,
            value=parameter,
        )


def _check_range(parameter, name, lb, ub=None):
    """

    Private function for checking parameter value against a specified range.

    Parameters
    ----------

    """

    if ub:
        if parameter < lb or parameter > ub:
            return ("`{name}` kwarg value {value} out of range {lb}-{ub}.").format(
                name=name, value=parameter, lb=lb, ub=ub
            )
    else:
        if parameter < lb:
            return (
                "`{name}` kwarg value {value} out of range {ub}>{name}>{lb}."
            ).format(name=name, value=parameter, lb=lb, ub=ub)


def _check_axis(array, name, axis, values):
    """

    Private function for checking axis size.

    """

    if np.shape(array)[axis] not in values:
        if len(values) > 1:
            values_str = ", ".join(values)
            splitpoint = values_str.rfind(",")
            values_str = values_str[:splitpoint] + " or" + values_str[splitpoint + 1 :]
        else:
            values_str = values[0]
        return (
            "`{name}` kwarg axis {axis} size invalid. "
            "Expected {values}, but got {size}."
        ).format(name=name, axis=axis, values=values_str, size=np.shape(array)[axis])


def _check_dim(array, name, value):
    """

    Private function for checking array dimensions.

    """

    if np.asarray(array).ndim != value:
        if value != 1 or (value == 1 and np.shape(np.asarray(array))[1] != 1):
            return (
                "`{name}` kwarg dimensions are invalid. "
                "Expected {value}D-array, but got a {dim}D-array"
            ).format(
                name=name,
                value=value,
                dim=array.ndim,
            )


def _check_comp(parameter1, name1, parameter2, name2):
    if parameter1 > parameter2:
        return "`{name1}`>`{name2}`: {value1}>{value2}.".format(
            name1=name1,
            name2=name2,
            value1=parameter1,
            value2=parameter2,
        )


def _check_path(path, name):
    if os.path.isdir(path) is False:
        return "`{name}` does not exist at the path supplied:\n{path}".format(
            name=name,
            path=path,
        )


def _check_character(parameter, character, index):
    if parameter[index] != character:
        if index == -1:
            parameter += character
        else:
            parameter = parameter[:index] + character + parameter[index:]
    return parameter


def _check_solved(data):
    if data["solved"] is not True:
        return (
            "{data_type} not yet solved therefore nothing to inspect. "
            "First, run :meth:`~geopyv.{lcdata_type}.{data_type}.solve()` to solve."
        ).format(data_type=data["type"], lcdata_type=data["type"].lower())


def _conversion(parameter, name, new_type, show=True):
    if show:
        return ("`{name}` kwarg converted to a {type}: {value}.").format(
            name=name,
            type=new_type.__name__,
            value=parameter,
        )
    else:
        return ("`{name}` kwarg converted to a {type}.").format(
            name=name,
            type=new_type.__name__,
        )


def _report(msg, error_type):
    if msg and error_type != "Warning":
        log.error(msg)
    elif msg and error_type == "Warning":
        log.warning(msg)
    if error_type == "ValueError":
        raise ValueError(msg)
    elif error_type == "TypeError":
        raise TypeError(msg)
