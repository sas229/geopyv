"""

Validation module for geopyv.

"""

import logging
import geopyv as gp
import numpy as np
from geopyv.object import Object

log = logging.getLogger(__name__)


class ValidationBase(Object):
    """

    Validation base class to be used as a mixin. Contains plot functionality.

    """

    def __init__(self):
        super().__init__(object_type="Validation")
        """

        Validation base class initialiser.

        """

    def error(
        self,
        *,
        component=None,
        metric="se",
        zero=True,
        position=True,
        xlim=None,
        ylim=None,
        logscale=True,
        show=True,
        block=True,
        save=None
    ):
        # Check if solved.
        if self.data["solved"] is not True:
            log.error(
                "Validation not yet solved therefore no convergence data to plot. "
                "First, run :meth:`~geopyv.validation.Validation.solve()` to solve."
            )
            raise ValueError(
                "Validation not yet solved therefore no convergence data to plot. "
                "First, run :meth:`~geopyv.validation.Validation.solve()` to solve."
            )

        # Check input.
        self._report(gp.check._check_type(component, "component", [int]), "TypeError")
        self._report(
            gp.check._check_range(component, "component", 0, ub=11),
            "IndexError",
        )
        self._report(gp.check._check_type(metric, "metric", [str]), "TypeError")
        self._report(
            gp.check._check_value(metric, "metric", ["se", "kde", "pf"]), "ValueError"
        )
        self._report(gp.check._check_type(zero, "zero", [bool]), "TypeError")
        self._report(gp.check._check_type(position, "position", [bool]), "TypeError")
        types = [tuple, list, np.ndarray, type(None)]
        self._report(gp.check._check_type(xlim, "xlim", types), "TypeError")
        if xlim is not None:
            self._report(gp.check._check_dim(xlim, "xlim", 1), "ValueError")
            self._report(gp.check._check_axis(xlim, "xlim", 0, [2]), "ValueError")
        self._report(gp.check._check_type(ylim, "ylim", types), "TypeError")
        if ylim is not None:
            self._report(gp.check._check_dim(ylim, "ylim", 1), "ValueError")
            self._report(gp.check._check_axis(ylim, "ylim", 0, [2]), "ValueError")
        self._report(gp.check._check_type(logscale, "logscale", [bool]), "TypeError")
        self._report(gp.check._check_type(show, "show", [bool]), "TypeError")
        self._report(gp.check._check_type(block, "block", [bool]), "TypeError")
        self._report(gp.check._check_type(save, "save", [str, type(None)]), "TypeError")

        fig, ax = gp.plots.error_validation(
            data=self.data,
            component=component,
            metric=metric,
            zero=zero,
            position=position,
            xlim=xlim,
            ylim=ylim,
            logscale=logscale,
            show=show,
            block=block,
            save=save,
        )

    def _report(self, msg, error_type):
        if msg and error_type != "Warning":
            log.error(msg)
        elif msg and error_type == "Warning":
            log.warning(msg)
            return True
        if error_type == "ValueError" and msg:
            raise ValueError(msg)
        elif error_type == "TypeError" and msg:
            raise TypeError(msg)
        elif error_type == "IndexError" and msg:
            raise IndexError(msg)


class Validation(ValidationBase):
    def __init__(self, *, speckle=None, fields=None, labels=None):
        # Input checks.
        self._report(
            gp.check._check_type(
                speckle, "speckle", [gp.speckle.Speckle, gp.speckle.SpeckleResults]
            ),
            "Error",
        )
        self._report(gp.check._check_type(fields, "fields", [list]), "Error")
        for field in fields:
            self._report(
                gp.check._check_type(
                    field, "field", [gp.field.Field, gp.field.FieldResults]
                ),
                "Error",
            )
        self._report(gp.check._check_type(labels, "labels", [list]), "Error")
        for label in labels:
            self._report(gp.check._check_type(label, "label", [str]), "Error")

        # Store variables.
        self._speckle = speckle
        self._fields = fields
        self._labels = labels
        self.solved = False
        # data.
        self.data = {
            "type": "Validation",
            "solved": self.solved,
            "labels": self._labels,
            "speckle": self._speckle,
            "fields": self._fields,
        }

    def solve(self):
        self._applied = []  # field, image,point, warp
        self._observed = []

        for field in self._fields:
            applied = np.zeros(
                (
                    self._speckle.data["image_no"] - 1,
                    np.shape(field.data["field"]["coordinates"])[0],
                    12,
                )
            )
            observed = np.zeros(
                (
                    self._speckle.data["image_no"] - 1,
                    np.shape(field.data["field"]["coordinates"])[0],
                    12,
                )
            )
            for i in range(1, self._speckle.data["image_no"]):
                applied[i - 1] = self._speckle._warp(
                    i, field.data["field"]["coordinates"]
                )
            for i in range(np.shape(field.data["particles"])[0]):
                if np.shape(field.data["particles"][i]["warps"])[1] == 6:
                    observed[:, i, :6] = field.data["particles"][i]["warps"][1:]
                elif np.shape(field.data["particles"][i]["warps"])[1] == 12:
                    observed[:, i] = field.data["particles"][i]["warps"][1:]

            self._applied.append(applied)
            self._observed.append(observed)

        self.solved = True
        self.data["solved"] = self.solved
        self.data.update({"applied": self._applied, "observed": self._observed})


class ValidationResults(ValidationBase):
    """

    ValidationResults class for geopyv.

    Parameters
    ----------
    data : dict
        geopyv data dict from Validation object.


    Attributes
    ----------
    data : dict
        geopyv data dict from Validation object.

    """

    def __init__(self, data):
        """Initialisation of geopyv ValidationResults class."""
        self.data = data
