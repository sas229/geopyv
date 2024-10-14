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

    def standard_error(
        self,
        *,
        component=None,
        observing=None,
        xlim=None,
        ylim=None,
        scale=True,
        prev_series=None,
        prev_series_label=None,
        plot="scatter",
        show=True,
        block=True,
        save=None,
        xlabel=None,
        ylabel=None
    ):
        # Check if solved.
        if self.data["solved"] is not True:
            log.error(
                "Validation not yet solved therefore no standard error data to plot. "
                "First, run :meth:`~geopyv.validation.Validation.solve()` to solve."
            )
            raise ValueError(
                "Validation not yet solved therefore no standard error data to plot. "
                "First, run :meth:`~geopyv.validation.Validation.solve()` to solve."
            )

        # Check input.
        self._report(gp.check._check_type(component, "component", [int]), "TypeError")
        self._report(
            gp.check._check_range(component, "component", 0, ub=13),
            "IndexError",
        )
        types = [tuple, list, np.ndarray, type(None)]
        self._report(gp.check._check_type(xlim, "xlim", types), "TypeError")
        if xlim is not None:
            self._report(gp.check._check_dim(xlim, "xlim", 1), "ValueError")
            self._report(gp.check._check_axis(xlim, "xlim", 0, [2]), "ValueError")
        self._report(gp.check._check_type(ylim, "ylim", types), "TypeError")
        if ylim is not None:
            self._report(gp.check._check_dim(ylim, "ylim", 1), "ValueError")
            self._report(gp.check._check_axis(ylim, "ylim", 0, [2]), "ValueError")
        self._report(gp.check._check_type(scale, "scale", [str]), "TypeError")
        if scale == "lin" or scale == "Lin" or scale == "Linear":
            scale = "linear"
        elif scale == "Log" or scale == "ln":
            scale = "log"
        self._report(
            gp.check._check_value(scale, "scale", ["linear", "log"]), "ValueError"
        )
        self._report(gp.check._check_type(plot, "plot", [str]), "TypeError")
        self._report(
            gp.check._check_value(plot, "plot", ["scatter", "line"]), "ValueError"
        )
        self._report(gp.check._check_type(show, "show", [bool]), "TypeError")
        self._report(gp.check._check_type(block, "block", [bool]), "TypeError")
        self._report(gp.check._check_type(save, "save", [str, type(None)]), "TypeError")

        fig, ax = gp.plots.standard_error_validation(
            data=self.data,
            component=component,
            observing=observing,
            xlim=xlim,
            ylim=ylim,
            scale=scale,
            prev_series=prev_series,
            prev_series_label=prev_series_label,
            plot=plot,
            show=show,
            block=block,
            save=save,
            xlabel=xlabel,
            ylabel=ylabel,
        )

        return fig, ax

    def mean_error(
        self,
        *,
        component=None,
        xlim=None,
        ylim=None,
        scale=True,
        prev_series=None,
        prev_series_label=None,
        plot="scatter",
        show=True,
        block=True,
        save=None
    ):
        # Check if solved.
        if self.data["solved"] is not True:
            log.error(
                "Validation not yet solved therefore no standard error data to plot. "
                "First, run :meth:`~geopyv.validation.Validation.solve()` to solve."
            )
            raise ValueError(
                "Validation not yet solved therefore no standard error data to plot. "
                "First, run :meth:`~geopyv.validation.Validation.solve()` to solve."
            )

        # Check input.
        self._report(gp.check._check_type(component, "component", [int]), "TypeError")
        self._report(
            gp.check._check_range(component, "component", 0, ub=13),
            "IndexError",
        )
        types = [tuple, list, np.ndarray, type(None)]
        self._report(gp.check._check_type(xlim, "xlim", types), "TypeError")
        if xlim is not None:
            self._report(gp.check._check_dim(xlim, "xlim", 1), "ValueError")
            self._report(gp.check._check_axis(xlim, "xlim", 0, [2]), "ValueError")
        self._report(gp.check._check_type(ylim, "ylim", types), "TypeError")
        if ylim is not None:
            self._report(gp.check._check_dim(ylim, "ylim", 1), "ValueError")
            self._report(gp.check._check_axis(ylim, "ylim", 0, [2]), "ValueError")
        self._report(gp.check._check_type(scale, "scale", [str]), "TypeError")
        if scale == "lin" or scale == "Lin" or scale == "Linear":
            scale = "linear"
        elif scale == "Log" or scale == "ln":
            scale = "log"
        self._report(
            gp.check._check_value(scale, "scale", ["linear", "log"]), "ValueError"
        )
        self._report(gp.check._check_type(plot, "plot", [str]), "TypeError")
        self._report(
            gp.check._check_value(plot, "plot", ["scatter", "line"]), "ValueError"
        )
        self._report(gp.check._check_type(show, "show", [bool]), "TypeError")
        self._report(gp.check._check_type(block, "block", [bool]), "TypeError")
        self._report(gp.check._check_type(save, "save", [str, type(None)]), "TypeError")

        fig, ax = gp.plots.mean_error_validation(
            data=self.data,
            component=component,
            xlim=xlim,
            ylim=ylim,
            scale=scale,
            prev_series=prev_series,
            prev_series_label=prev_series_label,
            plot=plot,
            show=show,
            block=block,
            save=save,
        )

        return fig, ax

    def noise_standard_error(
        self,
        *,
        component=None,
        observing=None,
        xlim=None,
        ylim=None,
        scale=True,
        plot="scatter",
        show=True,
        block=True,
        save=None,
        xlabel=None,
        ylabel=None
    ):
        # Check if solved.
        if self.data["solved"] is not True:
            log.error(
                "Validation not yet solved therefore no standard error data to plot. "
                "First, run :meth:`~geopyv.validation.Validation.solve()` to solve."
            )
            raise ValueError(
                "Validation not yet solved therefore no standard error data to plot. "
                "First, run :meth:`~geopyv.validation.Validation.solve()` to solve."
            )

        # Check input.
        self._report(gp.check._check_type(component, "component", [int]), "TypeError")
        self._report(
            gp.check._check_range(component, "component", 0, ub=13),
            "IndexError",
        )
        types = [tuple, list, np.ndarray, type(None)]
        self._report(gp.check._check_type(xlim, "xlim", types), "TypeError")
        if xlim is not None:
            self._report(gp.check._check_dim(xlim, "xlim", 1), "ValueError")
            self._report(gp.check._check_axis(xlim, "xlim", 0, [2]), "ValueError")
        self._report(gp.check._check_type(ylim, "ylim", types), "TypeError")
        if ylim is not None:
            self._report(gp.check._check_dim(ylim, "ylim", 1), "ValueError")
            self._report(gp.check._check_axis(ylim, "ylim", 0, [2]), "ValueError")
        self._report(gp.check._check_type(scale, "scale", [str]), "TypeError")
        if scale == "lin" or scale == "Lin" or scale == "Linear":
            scale = "linear"
        elif scale == "Log" or scale == "ln":
            scale = "log"
        self._report(
            gp.check._check_value(scale, "scale", ["linear", "log"]), "ValueError"
        )
        self._report(gp.check._check_type(plot, "plot", [str]), "TypeError")
        self._report(
            gp.check._check_value(plot, "plot", ["scatter", "line"]), "ValueError"
        )
        self._report(gp.check._check_type(show, "show", [bool]), "TypeError")
        self._report(gp.check._check_type(block, "block", [bool]), "TypeError")
        self._report(gp.check._check_type(save, "save", [str, type(None)]), "TypeError")

        fig, ax = gp.plots.noise_standard_error_validation(
            data=self.data,
            component=component,
            observing=observing,
            xlim=xlim,
            ylim=ylim,
            scale=scale,
            plot=plot,
            show=show,
            block=block,
            save=save,
            xlabel=xlabel,
            ylabel=ylabel,
        )

        return fig, ax

    def noise_mean_error(
        self,
        *,
        component=None,
        xlim=None,
        ylim=None,
        scale=True,
        plot="scatter",
        show=True,
        block=True,
        save=None
    ):
        # Check if solved.
        if self.data["solved"] is not True:
            log.error(
                "Validation not yet solved therefore no standard error data to plot. "
                "First, run :meth:`~geopyv.validation.Validation.solve()` to solve."
            )
            raise ValueError(
                "Validation not yet solved therefore no standard error data to plot. "
                "First, run :meth:`~geopyv.validation.Validation.solve()` to solve."
            )

        # Check input.
        self._report(gp.check._check_type(component, "component", [int]), "TypeError")
        self._report(
            gp.check._check_range(component, "component", 0, ub=13),
            "IndexError",
        )
        types = [tuple, list, np.ndarray, type(None)]
        self._report(gp.check._check_type(xlim, "xlim", types), "TypeError")
        if xlim is not None:
            self._report(gp.check._check_dim(xlim, "xlim", 1), "ValueError")
            self._report(gp.check._check_axis(xlim, "xlim", 0, [2]), "ValueError")
        self._report(gp.check._check_type(ylim, "ylim", types), "TypeError")
        if ylim is not None:
            self._report(gp.check._check_dim(ylim, "ylim", 1), "ValueError")
            self._report(gp.check._check_axis(ylim, "ylim", 0, [2]), "ValueError")
        self._report(gp.check._check_type(scale, "scale", [str]), "TypeError")
        if scale == "lin" or scale == "Lin" or scale == "Linear":
            scale = "linear"
        elif scale == "Log" or scale == "ln":
            scale = "log"
        self._report(
            gp.check._check_value(scale, "scale", ["linear", "log"]), "ValueError"
        )
        self._report(gp.check._check_type(plot, "plot", [str]), "TypeError")
        self._report(
            gp.check._check_value(plot, "plot", ["scatter", "line"]), "ValueError"
        )
        self._report(gp.check._check_type(show, "show", [bool]), "TypeError")
        self._report(gp.check._check_type(block, "block", [bool]), "TypeError")
        self._report(gp.check._check_type(save, "save", [str, type(None)]), "TypeError")

        fig, ax = gp.plots.noise_mean_error_validation(
            data=self.data,
            component=component,
            xlim=xlim,
            ylim=ylim,
            scale=scale,
            plot=plot,
            show=show,
            block=block,
            save=save,
        )

    def strain_error(
        self,
        *,
        component=None,
        xlim=None,
        ylim=None,
        scale=True,
        plot="scatter",
        show=True,
        block=True,
        save=None
    ):
        # Check if solved.
        if self.data["solved"] is not True:
            log.error(
                "Validation not yet solved therefore no standard error data to plot. "
                "First, run :meth:`~geopyv.validation.Validation.solve()` to solve."
            )
            raise ValueError(
                "Validation not yet solved therefore no standard error data to plot. "
                "First, run :meth:`~geopyv.validation.Validation.solve()` to solve."
            )

        # Check input.
        self._report(gp.check._check_type(component, "component", [int]), "TypeError")
        self._report(
            gp.check._check_range(component, "component", 0, ub=13),
            "IndexError",
        )
        types = [tuple, list, np.ndarray, type(None)]
        self._report(gp.check._check_type(xlim, "xlim", types), "TypeError")
        if xlim is not None:
            self._report(gp.check._check_dim(xlim, "xlim", 1), "ValueError")
            self._report(gp.check._check_axis(xlim, "xlim", 0, [2]), "ValueError")
        self._report(gp.check._check_type(ylim, "ylim", types), "TypeError")
        if ylim is not None:
            self._report(gp.check._check_dim(ylim, "ylim", 1), "ValueError")
            self._report(gp.check._check_axis(ylim, "ylim", 0, [2]), "ValueError")
        self._report(gp.check._check_type(scale, "scale", [str]), "TypeError")
        if scale == "lin" or scale == "Lin" or scale == "Linear":
            scale = "linear"
        elif scale == "Log" or scale == "ln":
            scale = "log"
        self._report(
            gp.check._check_value(scale, "scale", ["linear", "log"]), "ValueError"
        )
        self._report(gp.check._check_type(plot, "plot", [str]), "TypeError")
        self._report(
            gp.check._check_value(plot, "plot", ["scatter", "line"]), "ValueError"
        )
        self._report(gp.check._check_type(show, "show", [bool]), "TypeError")
        self._report(gp.check._check_type(block, "block", [bool]), "TypeError")
        self._report(gp.check._check_type(save, "save", [str, type(None)]), "TypeError")

        fig, ax = gp.plots.strain_error_validation(
            data=self.data,
            component=component,
            xlim=xlim,
            ylim=ylim,
            scale=scale,
            plot=plot,
            show=show,
            block=block,
            save=save,
        )

        return fig, ax

    def spatial_error(
        self,
        *,
        field_index=0,
        time_index=0,
        quantity="R",
        imshow=True,
        colorbar=True,
        ticks=None,
        alpha=0.5,
        levels=None,
        xlim=None,
        ylim=None,
        show=True,
        block=True,
        save=None
    ):
        # Check if solved.
        if self.data["solved"] is not True:
            log.error(
                "Validation not yet solved therefore no standard error data to plot. "
                "First, run :meth:`~geopyv.validation.Validation.solve()` to solve."
            )
            raise ValueError(
                "Validation not yet solved therefore no standard error data to plot. "
                "First, run :meth:`~geopyv.validation.Validation.solve()` to solve."
            )

        # Check input.
        self._report(
            gp.check._check_type(quantity, "quantity", [str, type(None)]), "TypeError"
        )
        types = [
            "u",
            "v",
            "R",
        ]
        if quantity:
            self._report(
                gp.check._check_value(quantity, "quantity", types), "ValueError"
            )
        self._report(gp.check._check_type(imshow, "imshow", [bool]), "TypeError")
        self._report(gp.check._check_type(colorbar, "colorbar", [bool]), "TypeError")
        types = [tuple, list, np.ndarray, type(None)]
        self._report(gp.check._check_type(ticks, "ticks", types), "TypeError")
        check = gp.check._check_type(alpha, "alpha", [float])
        if check:
            try:
                alpha = float(alpha)
                self._report(gp.check._conversion(alpha, "alpha", float), "Warning")
            except Exception:
                self._report(check, "TypeError")
        self._report(gp.check._check_range(alpha, "alpha", 0.0, 1.0), "ValueError")
        types = [int, tuple, list, np.ndarray, type(None)]
        self._report(gp.check._check_type(levels, "levels", types), "TypeError")
        self._report(gp.check._check_type(xlim, "xlim", types), "TypeError")
        if xlim is not None:
            self._report(gp.check._check_dim(xlim, "xlim", 1), "ValueError")
            self._report(gp.check._check_axis(xlim, "xlim", 0, [2]), "ValueError")
        self._report(gp.check._check_type(ylim, "ylim", types), "TypeError")
        if ylim is not None:
            self._report(gp.check._check_dim(ylim, "ylim", 1), "ValueError")
            self._report(gp.check._check_axis(ylim, "ylim", 0, [2]), "ValueError")
        self._report(gp.check._check_type(show, "show", [bool]), "TypeError")
        self._report(gp.check._check_type(block, "block", [bool]), "TypeError")
        self._report(gp.check._check_type(save, "save", [str, type(None)]), "TypeError")

        fig, ax = gp.plots.spatial_error_validation(
            data=self.data,
            field_index=field_index,
            time_index=time_index,
            quantity=quantity,
            imshow=imshow,
            colorbar=colorbar,
            ticks=ticks,
            alpha=alpha,
            levels=levels,
            xlim=xlim,
            ylim=ylim,
            show=show,
            block=block,
            save=save,
        )

        return fig, ax

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

    def solve(self, *, cumulative=True, skim=None, rot = False):
        self._skim = skim
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
            if cumulative:
                for i in range(1, self._speckle.data["image_no"]):
                    applied[i - 1] = self._speckle._warp(
                        i, field.data["field"]["coordinates"],
                    )
            else:
                particle_coordinates = np.zeros(
                    (
                        np.shape(field.data["particles"])[0],
                        self._speckle.data["image_no"],
                        2,
                    )
                )
                for j in range(np.shape(field.data["particles"])[0]):
                    particle_coordinates[j] = field.data["particles"][j].data[
                        "results"
                    ]["coordinates"]
                for i in range(1, self._speckle.data["image_no"]):
                    applied[i - 1] = self._speckle._warp(
                        i, field.data["field"]["coordinates"]
                    ) - self._speckle._warp(i - 1, field.data)
            for i in range(np.shape(field.data["particles"])[0]):
                if (
                    np.shape(field.data["particles"][i].data["results"]["warps"])[1]
                    == 6
                ):
                    observed[
                        : len(field.data["particles"][i].data["results"]["warps"]) - 1,
                        i,
                        :6,
                    ] = field.data["particles"][i].data["results"]["warps"][1:]
                elif (
                    np.shape(field.data["particles"][i].data["results"]["warps"])[1]
                    == 12
                ):
                    observed[
                        : len(field.data["particles"][i].data["results"]["warps"]) - 1,
                        i,
                    ] = field.data["particles"][i].data["results"]["warps"][1:]
            self._applied.append(applied)
            self._observed.append(observed)
        self._anomalies()

        self.solved = True
        self.data["solved"] = self.solved
        self.data.update({"applied": self._applied, "observed": self._observed})

    def _anomalies(self):
        """

        Remove particles that distort picture of performance.

        """

        if self._skim is not None:
            for j in range(len(self._applied)):  # Field.
                applied = np.zeros(
                    (
                        self._speckle.data["image_no"] - 1,
                        np.shape(self._fields[j].data["field"]["coordinates"])[0]
                        - self._skim,
                        12,
                    )
                )
                observed = np.zeros(
                    (
                        self._speckle.data["image_no"] - 1,
                        np.shape(self._fields[j].data["field"]["coordinates"])[0]
                        - self._skim,
                        12,
                    )
                )
                for i in range(len(self._applied[j])):  # Time step.
                    errors = np.sqrt(
                        np.sum(
                            (self._applied[j][i, :, :2] - self._observed[j][i, :, :2])
                            ** 2,
                            axis=1,
                        )
                    )
                    errors_args = np.argsort(errors)[::-1]
                    applied[i] = np.delete(
                        self._applied[j][i], errors_args[: self._skim], axis=0
                    )
                    observed[i] = np.delete(
                        self._observed[j][i], errors_args[: self._skim], axis=0
                    )
                self._applied[j] = applied
                self._observed[j] = observed


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
