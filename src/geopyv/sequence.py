"""

Sequence module for geopyv.

"""
import logging
import numpy as np
import geopyv as gp
from geopyv.object import Object
import re
import os
import sys
import glob

log = logging.getLogger(__name__)


class SequenceBase(Object):
    """

    Sequence base class to be used as a mixin. Contains plot functionality.

    """

    def __init__(self):
        super().__init__(object_type="Sequence")
        """

        Sequence base class initialiser.

        """

    def inspect(
        self,
        *,
        mesh_index=None,
        subset_index=None,
        show=True,
        block=True,
        save=None,
    ):
        """Method to show the sequence and associated mesh and subset quality metrics using
        :mod: `~geopyv.plots.inspect_subset` for subsets and
        :mod: `~geopyv.plots.inspect_mesh` for meshes.

        Parameters
        ----------
        mesh_index : int
            Index of the mesh to inspect.
        subset_index : int, optional
            Index of the subset to inspect. If `None', the mesh is inspected instead.
        show : bool, optional
            Control whether the plot is displayed.
        block : bool, optional
            Control whether the plot blocks execution until closed.
        save : str, optional
            Name to use to save plot. Uses default extension of `.png`.

        Returns
        -------
        fig :  `matplotlib.pyplot.figure`
            Figure object.
        ax : `matplotlib.pyplot.axes`
            Axes object.

        .. note::
            * The figure and axes objects can be returned allowing standard
              matplotlib functionality to be used to augment the plot generated.
              See the :ref:`plots tutorial <Plots Tutorial>` for guidance.

        .. seealso::
            :meth:`~geopyv.plots.inspect_subset`
            :meth:`~geopyv.plots.inspect_mesh`
        """

        # Check if solved.
        if self.data["solved"] is not True:
            log.error(
                "Sequence not yet solved therefore nothing to inspect. "
                "First, run :meth:`~geopyv.sequence.Sequence.solve()` to solve."
            )
            raise ValueError(
                "Sequence not yet solved therefore nothing to inspect. "
                "First, run :meth:`~geopyv.sequence.Sequence.solve()` to solve."
            )

        # Check mesh_index input.
        if type(mesh_index) != int:
            log.error(
                (
                    "`mesh_index` keyword argument type invalid. "
                    "Expected an `int`, but got a `{type2}`."
                ).format(type2=type(mesh_index).__name__)
            )
            raise TypeError(
                (
                    "`mesh_index` keyword argument type invalid. "
                    "Expected an `int`, but got a `{type2}`."
                ).format(type2=type(mesh_index).__name__)
            )
        elif mesh_index < 0 or mesh_index >= np.shape(self.data["meshes"])[0]:
            log.error(
                (
                    "`mesh_index` {input_value} is out of bounds for "
                    "axis 0 with size {max_value}."
                ).format(
                    max_value=np.shape(self.data["meshes"])[0],
                    input_value=mesh_index,
                )
            )
            raise IndexError(
                (
                    "`mesh_index` {input_value} is out of bounds for "
                    "axis 0 with size {max_value}."
                ).format(
                    max_value=np.shape(self.data["meshes"])[0],
                    input_value=mesh_index,
                )
            )

        # Load/access selected mesh object.
        if self.data["file_settings"]["save_by_reference"]:
            mesh = gp.io.load(
                filename=self.data["file_settings"]["mesh_folder"]
                + self.data["meshes"][mesh_index]
            ).data
        else:
            mesh = self.data["meshes"][mesh_index]

        # Check remaining input.
        if type(subset_index) != int and subset_index is not None:
            log.error(
                "`subset_index` keyword argument type invalid. "
                "Expected an `int` or a `NoneType`, but got a `{type2}`.".format(
                    type2=type(subset_index).__name__
                )
            )
            raise TypeError(
                "`subset_index` keyword argument type invalid. "
                "Expected an `int` or a `NoneType`, but got a `{type2}`.".format(
                    type2=type(subset_index).__name__
                )
            )
        elif subset_index is not None and (
            subset_index < 0 or subset_index >= np.shape(mesh["results"]["subsets"])[0]
        ):
            log.error(
                (
                    "`subset_index` {input_value} is out of bounds for "
                    "axis 0 with size {max_value}."
                ).format(
                    max_value=np.shape(mesh["results"]["subsets"])[0] - 1,
                    input_value=subset_index,
                )
            )
            raise IndexError(
                (
                    "`subset_index` {input_value} is out of bounds for "
                    "axis 0 with size {max_value}."
                ).format(
                    max_value=np.shape(mesh["results"]["subsets"])[0] - 1,
                    input_value=subset_index,
                )
            )
        if type(show) != bool:
            log.error(
                (
                    "`show` keyword argument type invalid. "
                    "Expected a `bool`, but got a `{type2}`."
                ).format(type2=type(show).__name__)
            )
            raise TypeError(
                (
                    "`show` keyword argument type invalid. "
                    "Expected a `bool`, but got a `{type2}`."
                ).format(type2=type(show).__name__)
            )
        if type(block) != bool:
            log.error(
                (
                    "`block` keyword argument type invalid. "
                    "Expected a `bool`, but got a `{type2}`."
                ).format(type2=type(block).__name__)
            )
            raise TypeError(
                (
                    "`block` keyword argument type invalid. "
                    "Expected a `bool`, but got a `{type2}`."
                ).format(type2=type(block).__name__)
            )
        if type(save) != str and save is not None:
            log.error(
                (
                    "`save` keyword argument type invalid. "
                    "Expected a `str` or `NoneType`, but got a `{type3}`."
                ).format(type3=type(save).__name__)
            )
            raise TypeError(
                (
                    "`save` keyword argument type invalid. "
                    "Expected a `str` or `NoneType`, but got a `{type3}`."
                ).format(type3=type(save).__name__)
            )

        # Inspect.
        if subset_index is not None:
            subset_data = mesh["results"]["subsets"][subset_index]
            mask = np.asarray(mesh["mask"])
            log.info(
                "Inspecting subset {subset} of mesh {mesh}...".format(
                    subset=subset_index, mesh=mesh_index
                )
            )
            fig, ax = gp.plots.inspect_subset(
                data=subset_data,
                mask=mask,
                show=show,
                block=block,
                save=save,
            )
            return fig, ax
        else:
            log.info("Inspecting mesh {mesh}...".format(mesh=mesh_index))
            fig, ax = gp.plots.inspect_mesh(
                data=mesh,
                show=show,
                block=block,
                save=save,
            )
            return fig, ax

    def convergence(
        self,
        *,
        mesh_index=None,
        subset_index=None,
        quantity=None,
        show=True,
        block=True,
        save=None,
    ):
        """
        Method to plot the rate of convergence using
        :mod: `~geopyv.plots.convergence_subset` for subsets and
        :mod: `~geopyv.plots.convergence_mesh` for meshes.

        Parameters
        ----------
        mesh_index : int
            Index of the mesh to inspect.
        subset_index : int, optional
            Index of the subset to inspect. If `None',
            the convergence plot is for the mesh instead.
        quantity : str, optional
            Selector for histogram convergence property if
            the convergence plot is for mesh. Defaults
            to `C_ZNCC` if left as default None.
        show : bool, optional
            Control whether the plot is displayed.
        block : bool, optional
            Control whether the plot blocks execution until closed.
        save : str, optional
            Name to use to save plot. Uses default extension of `.png`.


        Returns
        -------
        fig :  matplotlib.pyplot.figure
            Figure object.
        ax : `matplotlib.pyplot.axes`
            Axes object.


        .. note::
            * The figure and axes objects can be returned allowing standard
              matplotlib functionality to be used to augment the plot generated.
              See the :ref:`plots tutorial <Plots Tutorial>` for guidance.

        .. warning::
            * Can only be used once the mesh has been solved using the
              :meth:`~geopyv.mesh.Mesh.solve` method.

        .. seealso::
            :mod: `~geopyv.plots.convergence_subset`
            :meth:`~geopyv.plots.convergence_mesh`

        """

        # Check if solved.
        if self.data["solved"] is not True:
            log.error(
                "Sequence not yet solved therefore nothing to inspect. "
                "First, run :meth:`~geopyv.sequence.Sequence.solve()` to solve."
            )
            raise ValueError(
                "Sequence not yet solved therefore nothing to inspect. "
                "First, run :meth:`~geopyv.sequence.Sequence.solve()` to solve."
            )

        # Check mesh_index input.
        if type(mesh_index) != int:
            log.error(
                (
                    "`mesh_index` keyword argument type invalid. "
                    "Expected an `int`, but got a `{type2}`."
                ).format(type2=type(mesh_index).__name__)
            )
            raise TypeError(
                (
                    "`mesh_index` keyword argument type invalid. "
                    "Expected an `int`, but got a `{type2}`."
                ).format(type2=type(mesh_index).__name__)
            )
        elif mesh_index < 0 or mesh_index >= np.shape(self.data["meshes"])[0]:
            log.error(
                (
                    "`mesh_index` {input_value} is out of bounds for "
                    "axis 0 with size {max_value}."
                ).format(
                    max_value=np.shape(self.data["meshes"])[0],
                    input_value=mesh_index,
                )
            )
            raise IndexError(
                (
                    "`mesh_index` {input_value} is out of bounds for "
                    "axis 0 with size {max_value}."
                ).format(
                    max_value=np.shape(self.data["meshes"])[0],
                    input_value=mesh_index,
                )
            )

        # Load/access selected mesh object.
        if self.data["file_settings"]["save_by_reference"]:
            mesh = gp.io.load(
                filename=self.data["file_settings"]["mesh_folder"]
                + self.data["meshes"][mesh_index]
            ).data
        else:
            mesh = self.data["meshes"][mesh_index]

        # Check remaining input.
        if type(subset_index) != int and subset_index is not None:
            log.error(
                "`subset_index` keyword argument type invalid. "
                "Expected an `int` or a `NoneType`, but got a `{type2}`.".format(
                    type2=type(subset_index).__name__
                )
            )
            raise TypeError(
                "`subset_index` keyword argument type invalid. "
                "Expected an `int` or a `NoneType`, but got a `{type2}`.".format(
                    type2=type(subset_index).__name__
                )
            )
        elif subset_index is not None and (
            subset_index < 0 or subset_index >= np.shape(mesh["results"]["subsets"])[0]
        ):
            log.error(
                (
                    "`subset_index` {input_value} is out of bounds for "
                    "axis 0 with size {max_value}."
                ).format(
                    max_value=np.shape(mesh["results"]["subsets"])[0] - 1,
                    input_value=subset_index,
                )
            )
            raise IndexError(
                (
                    "`subset_index` {input_value} is out of bounds for "
                    "axis 0 with size {max_value}."
                ).format(
                    max_value=np.shape(mesh["results"]["subsets"])[0] - 1,
                    input_value=subset_index,
                )
            )
        if type(quantity) != str and quantity is not None:
            log.error(
                (
                    "`quantity` keyword argument type invalid. "
                    "Expected a `str` or `NoneType`, but got a {type3}."
                ).format(type3=type(quantity).__name__)
            )
            raise TypeError(
                (
                    "`quantity` keyword argument type invalid. "
                    "Expected a `str` or `NoneType`, but got a {type3}."
                ).format(type3=type(quantity).__name__)
            )
        elif quantity not in ["C_ZNCC", "iterations", "norm"] and quantity is not None:
            log.error(
                (
                    "`quantity` keyword argument value invalid. "
                    "Expected `C_ZNCC`, `iterations` or `norm`, but got {value}."
                ).format(value=quantity)
            )
            raise ValueError(
                "`quantity` keyword argument value invalid. "
                "Expected `C_ZNCC`, `iterations` or `norm`, but got {value}."
            )
        if type(show) != bool:
            log.error(
                (
                    "`show` keyword argument type invalid. "
                    "Expected a `bool`, but got a `{type2}`."
                ).format(type2=type(show).__name__)
            )
            raise TypeError(
                (
                    "`show` keyword argument type invalid. "
                    "Expected a `bool`, but got a `{type2}`."
                ).format(type2=type(show).__name__)
            )
        if type(block) != bool:
            log.error(
                (
                    "`block` keyword argument type invalid. "
                    "Expected a `bool`, but got a `{type2}`."
                ).format(type2=type(block).__name__)
            )
            raise TypeError(
                (
                    "`block` keyword argument type invalid. "
                    "Expected a `bool`, but got a `{type2}`."
                ).format(type2=type(block).__name__)
            )
        if type(save) != str and save is not None:
            log.error(
                (
                    "`save` keyword argument type invalid. "
                    "Expected a `str` or `NoneType`, but got a `{type3}`."
                ).format(type3=type(save).__name__)
            )
            raise TypeError(
                (
                    "`save` keyword argument type invalid. "
                    "Expected a `str` or `NoneType`, but got a `{type3}`."
                ).format(type3=type(save).__name__)
            )

        # Plot convergence.
        if subset_index is not None:
            log.info(
                (
                    "Generating convergence plots for subset {subset} "
                    "of mesh {mesh}..."
                ).format(subset=subset_index, mesh=mesh_index)
            )
            fig, ax = gp.plots.convergence_subset(
                mesh["results"]["subsets"][subset_index],
                show=show,
                block=block,
                save=save,
            )
            return fig, ax
        else:
            if quantity is None:
                quantity = "C_ZNCC"
            elif quantity not in ["C_ZNCC", "iterations", "norm"]:
                log.warning(
                    (
                        "Invalid quantity specified: {quantity}. "
                        "Reverting to C_ZNCC."
                    ).format(quantity=quantity)
                )
                quantity = "C_ZNCC"
            log.info(
                (
                    "Generating {quantity} convergence histogram " "for mesh {mesh}..."
                ).format(quantity=quantity, mesh=mesh_index)
            )
            fig, ax = gp.plots.convergence_mesh(
                data=mesh,
                quantity=quantity,
                show=show,
                block=block,
                save=save,
            )
            return fig, ax

    def contour(
        self,
        *,
        mesh_index=None,
        quantity="C_ZNCC",
        imshow=True,
        colorbar=True,
        ticks=None,
        mesh=False,
        alpha=0.75,
        levels=None,
        axis=True,
        xlim=None,
        ylim=None,
        show=True,
        block=True,
        save=None,
    ):
        """

        Method to plot the contours of a given quantity.

        Parameters
        ----------
        mesh_index : int
            Index of the mesh to inspect.
        quantity : str, optional
            Selector for contour parameter. Must be in:
            [`C_ZNCC`, `u`, `v`, `u_x`, `v_x`,`u_y`,`v_y`, `R`]
            Defaults to `C_ZNCC`.
        imshow : bool, optional
            Control whether the reference image is plotted.
            Defaults to True.
        colorbar : bool, optional
            Control whether the colour bar is plotted.
            Defaults to True.
        ticks : list, optional
            Overwrite default colourbar ticks.
            Defaults to None.
        mesh : bool, optional
            Control whether the mesh is plotted.
            Defaults to False.
        alpha : float, optional
            Control contour opacity. Must be between 0.0-1.0.
            Defaults to 0.75.
        levels : int or array-like, optional
            Control number and position of contour lines.
            Defaults to None.
        axis : bool, optional
            Control whether the axes are plotted.
            Defaults to True.
        xlim : array-like, optional
            Set the plot x-limits (lower_limit,upper_limit).
            Defaults to None.
        ylim : array-like, optional
            Set the plot y-limits (lower_limit,upper_limit).
            Defaults to None.
        show : bool, optional
            Control whether the plot is displayed.
            Defaults to True.
        block : bool, optional
            Control whether the plot blocks execution until closed.
            Defaults to False.
        save : str, optional
            Name to use to save plot. Uses default extension of `.png`.


        Returns
        -------
        fig :  matplotlib.pyplot.figure
            Figure object.
        ax : `matplotlib.pyplot.axes`
            Axes object.


        .. note::
            * The figure and axes objects can be returned allowing standard
              matplotlib functionality to be used to augment the plot generated.
              See the :ref:`plots tutorial <Plots Tutorial>` for guidance.

        .. warning::
            * Can only be used once the sequence has been solved using the
              :meth:`~geopyv.sequence.Sequence.solve` method.

        .. seealso::
            :meth:`~geopyv.plots.contour_sequence`

        """
        # Check if solved.
        if self.data["solved"] is not True:
            log.error(
                "Sequence not yet solved therefore nothing to inspect. "
                "First, run :meth:`~geopyv.sequence.Sequence.solve()` to solve."
            )
            raise ValueError(
                "Sequence not yet solved therefore nothing to inspect. "
                "First, run :meth:`~geopyv.sequence.Sequence.solve()` to solve."
            )

        # Check input.
        if type(mesh_index) != int:
            log.error(
                (
                    "`mesh_index` keyword argument type invalid. "
                    "Expected an `int`, but got a `{type2}`."
                ).format(type2=type(mesh_index).__name__)
            )
            raise TypeError(
                (
                    "`mesh_index` keyword argument type invalid. "
                    "Expected an `int`, but got a `{type2}`."
                ).format(type2=type(mesh_index).__name__)
            )
        elif mesh_index < 0 or mesh_index >= np.shape(self.data["meshes"])[0]:
            log.error(
                (
                    "`mesh_index` {input_value} is out of bounds for "
                    "axis 0 with size {max_value}."
                ).format(
                    max_value=np.shape(np.shape(self.data["meshes"])[0])[0] - 1,
                    input_value=mesh_index,
                )
            )
            raise IndexError(
                (
                    "`mesh_index` {input_value} is out of bounds for "
                    "axis 0 with size {max_value}."
                ).format(
                    max_value=np.shape(np.shape(self.data["meshes"])[0])[0] - 1,
                    input_value=mesh_index,
                )
            )
        if type(quantity) != str and quantity is not None:
            log.error(
                (
                    "`quantity` keyword argument type invalid. "
                    "Expected a `str` or `NoneType`, but got a {type3}."
                ).format(type3=type(quantity).__name__)
            )
            raise TypeError(
                (
                    "`quantity` keyword argument type invalid. "
                    "Expected a `str` or `NoneType`, but got a {type3}."
                ).format(type3=type(quantity).__name__)
            )
        elif quantity not in [
            "C_ZNCC",
            "iterations",
            "norm",
            "u",
            "v",
            "u_x",
            "v_x",
            "u_y",
            "v_y",
            "R",
        ]:
            log.error(
                (
                    "`quantity` keyword argument value invalid. "
                    "Expected `C_ZNCC`,`iterations, `norm`, `u`, `v`, `u_x`, "
                    "`v_x`, u_y`, `v_y` or `R`, but got {value}."
                ).format(value=quantity)
            )
            raise ValueError(
                (
                    "`quantity` keyword argument value invalid. "
                    "Expected `C_ZNCC`, `iterations, `norm`, `u`, `v`, `u_x`, "
                    "`v_x`, u_y`, `v_y` or `R`, but got {value}."
                ).format(value=quantity)
            )
        if type(imshow) != bool:
            log.error(
                (
                    "`imshow` keyword argument type invalid. "
                    "Expected a `bool`, but got a `{type2}`."
                ).format(type2=type(imshow).__name__)
            )
            raise TypeError(
                (
                    "`imshow` keyword argument type invalid. "
                    "Expected a `bool`, but got a `{type2}`."
                ).format(type2=type(imshow).__name__)
            )
        if type(colorbar) != bool:
            log.error(
                (
                    "`colorbar` keyword argument type invalid. "
                    "Expected a `bool`, but got a `{type2}`."
                ).format(type2=type(colorbar).__name__)
            )
            raise TypeError(
                (
                    "`colorbar` keyword argument type invalid. "
                    "Expected a `bool`, but got a `{type2}`."
                ).format(type2=type(colorbar).__name__)
            )
        if isinstance(ticks, (tuple, list, np.ndarray)) is False and ticks is not None:
            log.error(
                (
                    "`ticks` keyword argument type invalid. "
                    "Expected a `tuple`, `list` or `numpy.ndarray`, "
                    "but got a `{type2}`."
                ).format(type2=type(ticks).__name__)
            )
            raise TypeError(
                (
                    "`ticks` keyword argument type invalid. "
                    "Expected a `tuple`, `list` or `numpy.ndarray`, "
                    "but got a `{type2}`."
                ).format(type2=type(ticks).__name__)
            )
        if type(mesh) != bool:
            log.error(
                (
                    "`mesh` keyword argument type invalid. "
                    "Expected a `bool`, but got a `{type2}`."
                ).format(type2=type(mesh).__name__)
            )
            raise TypeError(
                (
                    "`mesh` keyword argument type invalid. "
                    "Expected a `bool`, but got a `{type2}`."
                ).format(type2=type(mesh).__name__)
            )
        if type(alpha) != float:
            log.warning(
                (
                    "`alpha` keyword argument type invalid. "
                    "Expected a `float`, but got a `{type2}`.\n"
                    "Attempting conversion..."
                ).format(type2=type(alpha).__name__)
            )
            try:
                alpha = float(alpha)
                log.warning(
                    (
                        "`alpha` keyword argument type conversion successful. "
                        "New value: {value}"
                    ).format(value=alpha)
                )
            except ValueError:
                log.error(
                    "`alpha` keyword argument type conversion failed. "
                    "Input a `float`, 0.0-1.0."
                )
                raise TypeError(
                    "`alpha` keyword argument type conversion failed. "
                    "Input a `float`, 0.0-1.0."
                )
        elif alpha < 0.0 or alpha > 1.0:
            log.error(
                (
                    "`alpha` keyword argument value {value} out of range 0.0-1.0. "
                    "Input a `float`, 0.0-1.0."
                ).format(value=alpha)
            )
            raise ValueError(
                (
                    "`alpha` keyword argument value {value} out of range 0.0-1.0. "
                    "Input a `float`, 0.0-1.0."
                ).format(value=alpha)
            )
        if (
            isinstance(levels, (int, tuple, list, np.ndarray)) is False
            and levels is not None
        ):
            log.error(
                (
                    "`levels` keyword argument type invalid. "
                    "Expected an `int`, `tuple`, `list`, "
                    "`numpy.ndarray` or `NoneType`, but got a {type6}."
                ).format(type6=type(levels).__name__)
            )
            raise TypeError(
                (
                    "`levels` keyword argument type invalid. "
                    "Expected an `int`, `tuple`, `list`, "
                    "`numpy.ndarray` or `NoneType`, but got a {type6}."
                ).format(type6=type(levels).__name__)
            )
        if type(axis) != bool:
            log.error(
                (
                    "`axis` keyword argument type invalid. "
                    "Expected a `bool`, but got a `{type2}`."
                ).format(type2=type(axis).__name__)
            )
            raise TypeError(
                (
                    "`axis` keyword argument type invalid. "
                    "Expected a `bool`, but got a `{type2}`."
                ).format(type2=type(axis).__name__)
            )
        if xlim is not None:
            if isinstance(xlim, (tuple, list, np.ndarray)) is False:
                log.error(
                    (
                        "`xlim` keyword argument type invalid. "
                        "Expected a `tuple`, `list`, `numpy.ndarray` or `NoneType`, "
                        "but got a {type5}."
                    ).format(type5=type(xlim).__name__)
                )
                raise TypeError(
                    (
                        "`xlim` keyword argument type invalid. "
                        "Expected a `tuple`, `list`, `numpy.ndarray` or `NoneType`, "
                        "but got a {type5}."
                    ).format(type5=type(xlim).__name__)
                )
            elif np.shape(xlim)[0] != 2:
                log.error(
                    (
                        "`xlim` keyword argument primary axis size invalid. "
                        "Expected 2, but got {size}."
                    ).format(size=np.shape(xlim)[0])
                )
                raise ValueError(
                    (
                        "`xlim` keyword argument primary axis size invalid. "
                        "Expected 2, but got {size}."
                    ).format(size=np.shape(xlim)[0])
                )
        if ylim is not None:
            if isinstance(ylim, (tuple, list, np.ndarray)) is False:
                log.error(
                    (
                        "`ylim` keyword argument type invalid. "
                        "Expected a `tuple`, `list`, `numpy.ndarray` or `NoneType`, "
                        "but got a {type5}."
                    ).format(type5=type(ylim).__name__)
                )
                raise TypeError(
                    (
                        "`ylim` keyword argument type invalid. "
                        "Expected a `tuple`, `list`, `numpy.ndarray` or `NoneType`, "
                        "but got a {type5}."
                    ).format(type5=type(ylim).__name__)
                )
            elif np.shape(ylim)[0] != 2:
                log.error(
                    (
                        "`ylim` keyword argument primary axis size invalid. "
                        "Expected 2, but got {size}."
                    ).format(size=np.shape(ylim)[0])
                )
                raise ValueError(
                    (
                        "`ylim` keyword argument primary axis size invalid. "
                        "Expected 2, but got {size}."
                    ).format(size=np.shape(ylim)[0])
                )
        if type(show) != bool:
            log.error(
                (
                    "`show` keyword argument type invalid. "
                    "Expected a `bool`, but got a `{type2}`."
                ).format(type2=type(show).__name__)
            )
            raise TypeError(
                (
                    "`show` keyword argument type invalid. "
                    "Expected a `bool`, but got a `{type2}`."
                ).format(type2=type(show).__name__)
            )
        if type(block) != bool:
            log.error(
                (
                    "`block` keyword argument type invalid. "
                    "Expected a `bool`, but got a `{type2}`."
                ).format(type2=type(block).__name__)
            )
            raise TypeError(
                (
                    "`block` keyword argument type invalid. "
                    "Expected a `bool`, but got a `{type2}`."
                ).format(type2=type(block).__name__)
            )
        if type(save) != str and save is not None:
            log.error(
                (
                    "`save` keyword argument type invalid. "
                    "Expected a `str` or `NoneType`, but got a `{type3}`."
                ).format(type3=type(save).__name__)
            )
            raise TypeError(
                (
                    "`save` keyword argument type invalid. "
                    "Expected a `str` or `NoneType`, but got a `{type3}`."
                ).format(type3=type(save).__name__)
            )

        # Load/access selected mesh object.
        if self.data["file_settings"]["save_by_reference"]:
            mesh_obj = gp.io.load(
                filename=self.data["file_settings"]["mesh_folder"]
                + self.data["meshes"][mesh_index]
            ).data
        else:
            mesh_obj = self.data["meshes"][mesh_index]

        log.info(
            "Generating {quantity} contour plot for mesh {mesh}...".format(
                quantity=quantity, mesh=mesh_index
            )
        )
        fig, ax = gp.plots.contour_mesh(
            data=mesh_obj,
            imshow=imshow,
            quantity=quantity,
            colorbar=colorbar,
            ticks=ticks,
            mesh=mesh,
            alpha=alpha,
            levels=levels,
            axis=axis,
            xlim=xlim,
            ylim=ylim,
            show=show,
            block=block,
            save=save,
        )
        return fig, ax

    def quiver(
        self,
        *,
        mesh_index=None,
        scale=1.0,
        imshow=True,
        mesh=False,
        axis=True,
        xlim=None,
        ylim=None,
        show=True,
        block=True,
        save=None,
    ):
        """

        Method to plot a quiver plot of the displacements.

        Parameters
        ----------
        mesh_index : int
            Index of the mesh to inspect.
        scale : float, optional
            Control size of quiver arrows.
        imshow : bool, optional
            Control whether the reference image is plotted.
            Defaults to True.
        mesh : bool, optional
            Control whether the mesh is plotted.
            Defaults to False.
        axis : bool, optional
            Control whether the axes are plotted.
            Defaults to True.
        xlim : array-like, optional
            Set the plot x-limits (lower_limit,upper_limit).
            Defaults to None.
        ylim : array-like, optional
            Set the plot y-limits (lower_limit,upper_limit).
            Defaults to None.
        show : bool, optional
            Control whether the plot is displayed.
            Defaults to True.
        block : bool, optional
            Control whether the plot blocks execution until closed.
            Defaults to False.
        save : str, optional
            Name to use to save plot. Uses default extension of `.png`.

        Returns
        -------
        fig :  matplotlib.pyplot.figure
            Figure object.
        ax : `matplotlib.pyplot.axes`
            Axes object.


        .. note::
            * The figure and axes objects can be returned allowing standard
              matplotlib functionality to be used to augment the plot generated.
              See the :ref:`plots tutorial <Plots Tutorial>` for guidance.

        .. warning::
            * Can only be used once the mesh has been solved using the
              :meth:`~geopyv.mesh.Mesh.solve` method.

        .. seealso::
            :meth:`~geopyv.plots.quiver_mesh`
        """
        # Check if solved.
        if self.data["solved"] is not True:
            log.error(
                "Sequence not yet solved therefore nothing to inspect. "
                "First, run :meth:`~geopyv.sequence.Sequence.solve()` to solve."
            )
            raise ValueError(
                "Sequence not yet solved therefore nothing to inspect. "
                "First, run :meth:`~geopyv.sequence.Sequence.solve()` to solve."
            )

        # Check input.
        if type(mesh_index) != int:
            log.error(
                (
                    "`mesh_index` keyword argument type invalid. "
                    "Expected an `int`, but got a `{type2}`."
                ).format(type2=type(mesh_index).__name__)
            )
            raise TypeError(
                (
                    "`mesh_index` keyword argument type invalid. "
                    "Expected an `int`, but got a `{type2}`."
                ).format(type2=type(mesh_index).__name__)
            )
        elif mesh_index < 0 or mesh_index >= np.shape(self.data["meshes"])[0]:
            log.error(
                (
                    "`mesh_index` {input_value} is out of bounds for "
                    "axis 0 with size {max_value}."
                ).format(
                    max_value=np.shape(np.shape(self.data["meshes"])[0])[0] - 1,
                    input_value=mesh_index,
                )
            )
            raise IndexError(
                (
                    "`mesh_index` {input_value} is out of bounds for "
                    "axis 0 with size {max_value}."
                ).format(
                    max_value=np.shape(np.shape(self.data["meshes"])[0])[0] - 1,
                    input_value=mesh_index,
                )
            )
        if type(scale) != float:
            log.error(
                (
                    "`scale` keyword argument type invalid. "
                    "Expected a `float`, but got a `{type2}`."
                ).format(type2=type(scale).__name__)
            )
            raise TypeError(
                (
                    "`scale` keyword argument type invalid. "
                    "Expected a `float`, but got a `{type2}`."
                ).format(type2=type(scale).__name__)
            )
        elif scale <= 0.0:
            log.error(
                (
                    "`scale` keyword argument value {value} out of range. "
                    "Input an `float` > 0.0"
                ).format(value=scale)
            )
            raise ValueError(
                (
                    "`scale` keyword argument value {value} out of range. "
                    "Input an `float` > 0.0"
                ).format(value=scale)
            )
        if type(imshow) != bool:
            log.error(
                (
                    "`imshow` keyword argument type invalid. "
                    "Expected a `bool`, but got a `{type2}`."
                ).format(type2=type(imshow).__name__)
            )
            raise TypeError(
                (
                    "`imshow` keyword argument type invalid. "
                    "Expected a `bool`, but got a `{type2}`."
                ).format(type2=type(imshow).__name__)
            )
        if type(mesh) != bool:
            log.error(
                (
                    "`mesh` keyword argument type invalid. "
                    "Expected a `bool`, but got a `{type2}`."
                ).format(type2=type(mesh).__name__)
            )
            raise TypeError(
                (
                    "`mesh` keyword argument type invalid. "
                    "Expected a `bool`, but got a `{type2}`."
                ).format(type2=type(mesh).__name__)
            )
        if type(axis) != bool:
            log.error(
                (
                    "`axis` keyword argument type invalid. "
                    "Expected a `bool`, but got a `{type2}`."
                ).format(type2=type(axis).__name__)
            )
            raise TypeError(
                (
                    "`axis` keyword argument type invalid. "
                    "Expected a `bool`, but got a `{type2}`."
                ).format(type2=type(axis).__name__)
            )
        if xlim is not None:
            if isinstance(xlim, (tuple, list, np.ndarray)) is False:
                log.error(
                    (
                        "`xlim` keyword argument type invalid. "
                        "Expected a `tuple`, `list`, `numpy.ndarray` or `NoneType`, "
                        "but got a {type5}."
                    ).format(type5=type(xlim).__name__)
                )
                raise TypeError(
                    (
                        "`xlim` keyword argument type invalid. "
                        "Expected a `tuple`, `list`, `numpy.ndarray` or `NoneType`, "
                        "but got a {type5}."
                    ).format(type5=type(xlim).__name__)
                )
            elif np.shape(xlim)[0] != 2:
                log.error(
                    (
                        "`xlim` keyword argument primary axis size invalid. "
                        "Expected 2, but got {size}."
                    ).format(size=np.shape(xlim)[0])
                )
                raise ValueError(
                    (
                        "`xlim` keyword argument primary axis size invalid. "
                        "Expected 2, but got {size}."
                    ).format(size=np.shape(xlim)[0])
                )
        if ylim is not None:
            if isinstance(ylim, (tuple, list, np.ndarray)) is False:
                log.error(
                    (
                        "`ylim` keyword argument type invalid. "
                        "Expected a `tuple`, `list`, `numpy.ndarray` or `NoneType`, "
                        "but got a {type5}."
                    ).format(type5=type(ylim).__name__)
                )
                raise TypeError(
                    (
                        "`ylim` keyword argument type invalid. "
                        "Expected a `tuple`, `list`, `numpy.ndarray` or `NoneType`, "
                        "but got a {type5}."
                    ).format(type5=type(ylim).__name__)
                )
            elif np.shape(ylim)[0] != 2:
                log.error(
                    (
                        "`ylim` keyword argument primary axis size invalid. "
                        "Expected 2, but got {size}."
                    ).format(size=np.shape(ylim)[0])
                )
                raise ValueError(
                    (
                        "`ylim` keyword argument primary axis size invalid. "
                        "Expected 2, but got {size}."
                    ).format(size=np.shape(ylim)[0])
                )
        if type(show) != bool:
            log.error(
                (
                    "`show` keyword argument type invalid. "
                    "Expected a `bool`, but got a `{type2}`."
                ).format(type2=type(show).__name__)
            )
            raise TypeError(
                (
                    "`show` keyword argument type invalid. "
                    "Expected a `bool`, but got a `{type2}`."
                ).format(type2=type(show).__name__)
            )
        if type(block) != bool:
            log.error(
                (
                    "`block` keyword argument type invalid. "
                    "Expected a `bool`, but got a `{type2}`."
                ).format(type2=type(block).__name__)
            )
            raise TypeError(
                (
                    "`block` keyword argument type invalid. "
                    "Expected a `bool`, but got a `{type2}`."
                ).format(type2=type(block).__name__)
            )
        if type(save) != str and save is not None:
            log.error(
                (
                    "`save` keyword argument type invalid. "
                    "Expected a `str` or `NoneType`, but got a `{type3}`."
                ).format(type3=type(save).__name__)
            )
            raise TypeError(
                (
                    "`save` keyword argument type invalid. "
                    "Expected a `str` or `NoneType`, but got a `{type3}`."
                ).format(type3=type(save).__name__)
            )

        # Load/access selected mesh object.
        if self.data["file_settings"]["save_by_reference"]:
            mesh_obj = gp.io.load(
                filename=self.data["file_settings"]["mesh_folder"]
                + self.data["meshes"][mesh_index]
            ).data
        else:
            mesh_obj = self.data["meshes"][mesh_index]

        # Plot quiver.
        log.info("Generating quiver plot for mesh {mesh}...".format(mesh=mesh_index))
        fig, ax = gp.plots.quiver_mesh(
            data=mesh_obj,
            scale=scale,
            imshow=imshow,
            mesh=mesh,
            axis=axis,
            xlim=xlim,
            ylim=ylim,
            show=show,
            block=block,
            save=save,
        )
        return fig, ax


class Sequence(SequenceBase):
    def __init__(
        self,
        *,
        image_folder=".",
        common_name="",
        image_file_type=".jpg",
        target_nodes=1000,
        boundary=None,
        exclusions=[],
        size_lower_bound=1.0,
        size_upper_bound=1000.0,
        save_by_reference=False,
        mesh_folder=".",
    ):
        """Initialisation of geopyv sequence object.

        Parameters
        ----------
        image_folder : str, optional
            Directory of images. Defaults to current working directory.
        image_file_type : str, optional
            Image file type. Options are ".jpg", ".png" or ".bmp". Defaults to .jpg.
        target_nodes : int
            Target node number. Defaults to 1000.
        boundary : `numpy.ndarray` (N,2)
            Array of coordinates to define the mesh boundary.
            Must be specified in clockwise or anti-clockwise order.
        exclusions: list, optional
            List of `numpy.ndarray` to define the mesh exclusions.
        size_lower_bound : int, optional
            Lower bound on element size. Defaults to a value of 1.
        upper_lower_bound : int, optional
            Lower bound on element size. Defaults to a value of 1000.
        save_by_reference : bool, optional
            False - sequence object stores meshes.
            True - sequence object stores mesh references.

        Attributes
        ----------
        data : dict
            Data object containing all settings and results.
            See the data structure :ref:`here <sequence_data_structure>`.
        solved : bool
            Boolean to indicate if the sequence has been solved.
        """

        # Set initialised boolean.
        self.initialised = False

        # Check types.
        if type(image_folder) != str:
            log.error(
                (
                    "`image_folder` keyword argument type invalid. "
                    "Expected a `str`, but got a {type2}."
                ).format(type2=type(image_folder).__name__)
            )
            raise TypeError(
                (
                    "`image_folder` keyword argument type invalid. "
                    "Expected a `str`, but got a {type2}."
                ).format(type2=type(image_folder).__name__)
            )
        elif os.path.isdir(image_folder) is False:
            log.error(
                "`image_folder` does not exist at the path supplied:\n{}".format(
                    image_folder
                )
            )
            gp.io._get_image_folder()
        if image_folder[-1] != "/":
            image_folder += "/"
        if type(image_file_type) != str:
            log.error(
                (
                    "`image_file_type` keyword argument type invalid. "
                    "Expected a `str`, but got a {type2}."
                ).format(type2=type(image_file_type).__name__)
            )
            raise TypeError(
                (
                    "`image_file_type` keyword argument type invalid. "
                    "Expected a `str`, but got a {type2}."
                ).format(type2=type(image_file_type).__name__)
            )
        elif image_file_type not in [".jpg", ".png", ".bmp"]:
            log.error(
                (
                    "`image_file_type` keyword argument value invalid. "
                    "Expected `.jpg`, `.png` or `.bmp`, but got {value}."
                ).format(value=image_file_type)
            )
            raise ValueError(
                (
                    "`image_file_type` keyword argument value invalid. "
                    "Expected `.jpg`, `.png` or `.bmp`, but got {value}."
                ).format(value=image_file_type)
            )
        if type(target_nodes) != int:
            log.warning(
                (
                    "`target_nodes` keyword argument type invalid. "
                    "Expected an `int`, but got a `{type2}`.\n"
                    "Attempting conversion..."
                ).format(type2=type(target_nodes).__name__)
            )
            try:
                target_nodes = int(target_nodes)
                log.warning(
                    (
                        "`target_nodes` keyword argument type conversion successful. "
                        "New value: {value}"
                    ).format(value=target_nodes)
                )
            except ValueError:
                log.error(
                    "`target_nodes` keyword argument type conversion failed. "
                    "Input an `int` > 0."
                )
                raise TypeError(
                    "`target_nodes` keyword argument type conversion failed. "
                    "Input an `int` > 0."
                )
        elif target_nodes <= 0:
            log.error(
                (
                    "`target_nodes` keyword argument value {value} out of range. "
                    "Input an `int` > 0."
                ).format(value=target_nodes)
            )
            raise ValueError(
                (
                    "`target_nodes` keyword argument value {value} out of range. "
                    "Input an `int` > 0."
                ).format(value=target_nodes)
            )
        if type(boundary) != np.ndarray:
            log.warning(
                (
                    "`boundary` keyword argument type invalid. "
                    "Expected a `numpy.ndarray`, but got a `{type2}`.\n"
                    "Attempting conversion..."
                ).format(type2=type(boundary).__name__)
            )
            try:
                boundary = np.asarray(boundary)
                log.warning(
                    (
                        "`boundary` keyword argument type conversion successful. "
                        "New value: {value}"
                    ).format(value=boundary)
                )
            except ValueError:
                log.error(
                    "`boundary` keyword argument type conversion failed. "
                    "Input a `numpy.ndarray` of shape (Nx,2)."
                )
                raise TypeError(
                    "`boundary` keyword argument type conversion failed. "
                    "Input a `numpy.ndarray` of shape (Nx,2)."
                )
        elif np.shape(boundary)[1] != 2:
            log.error(
                (
                    "`boundary` keyword argument secondary axis size invalid. "
                    "Expected 2, but got {size}."
                ).format(size=np.shape(boundary)[1])
            )
            raise ValueError(
                (
                    "`boundary` keyword argument secondary axis size invalid. "
                    "Expected 2, but got {size}."
                ).format(size=np.shape(boundary)[1])
            )
        elif boundary.ndim != 2:
            log.error(
                (
                    "`boundary` keyword argument dimensions invalid. "
                    "Expected 2, but got {size}."
                ).format(size=boundary.ndim)
            )
            raise ValueError(
                (
                    "`boundary` keyword argument dimensions invalid. "
                    "Expected 2, but got {size}."
                ).format(size=boundary.ndim)
            )
        if type(exclusions) != list:
            log.warning(
                (
                    "`exclusions` keyword argument type invalid. "
                    "Expected a `list`, but got a `{type2}`.\n"
                    "Attempting conversion..."
                ).format(type2=type(exclusions).__name__)
            )
            try:
                exclusions = list(exclusions)
                log.warning(
                    (
                        "`exclusions` keyword argument type conversion successful. "
                        "New value: {value}"
                    ).format(value=exclusions)
                )
            except ValueError:
                log.error(
                    "`exclusions` keyword arguement type conversion failed. "
                    "Input a `list` of `numpy.ndarray` of shape (Nx,2)."
                )
                raise TypeError(
                    "`exclusions` keyword arguement type conversion failed. "
                    "Input a `list` of `numpy.ndarray` of shape (Nx,2)."
                )
        for exclusion in exclusions:
            if type(exclusion) != np.ndarray:
                log.error(
                    (
                        "`exclusions` keyword argument value type invalid. "
                        "Expected a `numpy.ndarray`, but got a `{type2}`."
                    ).format(type2=type(exclusion).__name__)
                )
                raise TypeError(
                    (
                        "`exclusions` keyword argument value type invalid. "
                        "Expected a `numpy.ndarray`, but got a `{type2}`."
                    ).format(type2=type(exclusion).__name__)
                )
            elif np.shape(exclusion)[1] != 2:
                log.error(
                    (
                        "`exclusions` keyword argument value secondary axis "
                        "size invalid. Expected 2, but got {size}."
                    ).format(size=np.shape(exclusion)[1])
                )
                raise ValueError(
                    (
                        "`exclusions` keyword argument value secondary axis "
                        "size invalid. Expected 2, but got {size}."
                    ).format(size=np.shape(exclusion)[1])
                )
            elif exclusion.ndim != 2:
                log.error(
                    (
                        "`exclusions` keyword argument dimensions invalid. "
                        "Expected 2, but got {size}."
                    ).format(size=exclusion.ndim)
                )
                raise ValueError(
                    (
                        "`exclusions` keyword argument dimensions invalid. "
                        "Expected 2, but got {size}."
                    ).format(size=exclusion.ndim)
                )
        if isinstance(size_lower_bound, (int, float)) is False:
            log.error(
                (
                    "`size_lower_bound` keyword argument type invalid. "
                    "Expected an `int` or a `float`, but got a {type3}."
                ).format(type3=type(size_lower_bound).__name__)
            )
            raise TypeError(
                (
                    "`size_lower_bound` keyword argument type invalid. "
                    "Expected an `int` or a `float`, but got a {type3}."
                ).format(type3=type(size_lower_bound).__name__)
            )
        if isinstance(size_upper_bound, (int, float)) is False:
            log.error(
                (
                    "`size_upper_bound` keyword argument type invalid. "
                    "Expected an `int` or a `float`, but got a {type3}."
                ).format(type3=type(size_upper_bound).__name__)
            )
            raise TypeError(
                (
                    "`size_upper_bound` keyword argument type invalid. "
                    "Expected an `int` or a `float`, but got a {type3}."
                ).format(type3=type(size_upper_bound).__name__)
            )
        if size_upper_bound < size_lower_bound:
            log.error(
                (
                    "`size_upper_bound`<`size_lower_bound`: {value1}<{value2}. "
                    "Expected `size_upper_bound`>=`size_lower_bound`."
                ).format(value1=size_upper_bound, value2=size_lower_bound)
            )
            raise ValueError(
                (
                    "`size_upper_bound`<`size_lower_bound`: {value1}<{value2}. "
                    "Expected `size_upper_bound`>=`size_lower_bound`."
                ).format(value1=size_upper_bound, value2=size_lower_bound)
            )
        if type(save_by_reference) != bool:
            log.error(
                (
                    "`save_by_reference` keyword argument type invalid. "
                    "Expected a `bool`, but got a `{type2}`."
                ).format(type2=type(save_by_reference).__name__)
            )
            raise TypeError(
                (
                    "`save_by_reference` keyword argument type invalid. "
                    "Expected a `bool`, but got a `{type2}`."
                ).format(type2=type(save_by_reference).__name__)
            )
        if type(mesh_folder) != str:
            log.error(
                (
                    "`mesh_folder` keyword argument type invalid. "
                    "Expected a `str`, but got a {type2}."
                ).format(type2=type(mesh_folder).__name__)
            )
            raise TypeError(
                (
                    "`mesh_folder` keyword argument type invalid. "
                    "Expected a `str`, but got a {type2}."
                ).format(type2=type(mesh_folder).__name__)
            )
        if os.path.isdir(mesh_folder) is False:
            try:
                os.mkdir(mesh_folder)
            except Exception:
                log.error(
                    (
                        "`mesh_folder` does not exist at the path supplied and "
                        "cannot be created at:\n{folder}."
                    ).format(folder=mesh_folder)
                )
                gp.io._get_mesh_folder()
        if mesh_folder[-1] != "/":
            mesh_folder += "/"

        # Store variables.
        self._image_folder = image_folder
        self._image_file_type = image_file_type
        self._common_name = common_name
        platform = sys.platform
        if platform == "linux" or platform == "linux2" or platform == "darwin":
            split = "/"
        elif platform == "win32":
            split = r"\\"
        try:
            _images = glob.glob(
                self._image_folder
                + split
                + self._common_name
                + "*"
                + self._image_file_type
            )
            _image_indices_unordered = [int(re.findall(r"\d+", x)[-1]) for x in _images]
            _image_indices_arguments = np.argsort(_image_indices_unordered)
            self._images = [_images[index] for index in _image_indices_arguments]
            self._image_indices = np.sort(_image_indices_unordered)

        except Exception:
            log.error(
                "Issues encountered recognising image file names. "
                "Please refer to the documentation for naming guidance."
            )
            return
        self._target_nodes = target_nodes
        self._boundary = boundary
        self._exclusions = exclusions
        self._size_lower_bound = size_lower_bound
        self._size_upper_bound = size_upper_bound
        self._save_by_reference = save_by_reference
        self._mesh_folder = mesh_folder
        if self._save_by_reference:
            meshes = []
        else:
            meshes = np.empty(np.shape(self._image_indices)[0] - 1, dtype=dict)
        self.solved = False
        self._unsolvable = False

        # Data.
        file_settings = {
            "image_folder": self._image_folder,
            "images": self._images,
            "image_file_type": self._image_file_type,
            "image_indices": self._image_indices,
            "save_by_reference": self._save_by_reference,
            "mesh_folder": self._mesh_folder,
        }
        mesh_settings = {
            "target_nodes": self._target_nodes,
            "boundary": self._boundary,
            "exclusions": self._exclusions,
            "size_lower_bound": self._size_lower_bound,
            "size_upper_bound": self._size_upper_bound,
        }
        self.data = {
            "type": "Sequence",
            "solved": self.solved,
            "unsolvable": self._unsolvable,
            "meshes": meshes,
            "mesh_settings": mesh_settings,
            "file_settings": file_settings,
        }

        self._initialised = True

    def solve(
        self,
        *,
        seed_coord=None,
        template=None,
        max_norm=1e-3,
        max_iterations=15,
        adaptive_iterations=0,
        method="ICGN",
        mesh_order=2,
        subset_order=1,
        tolerance=0.7,
        seed_tolerance=0.9,
        alpha=0.5,
        track=False,
        hard_boundary=True,
        subset_size_compensation=False,
    ):
        """
        Method to solve for the sequence.

        Parameters
        ----------
        seed_coord : numpy.ndarray (2,)
            Seed coordinate for reliability-guided mesh solving.
        template : gp.template.Template object, optional
            subset template. Defaults to Circle(50).
        max_norm : float, optional
            Exit criterion for norm of increment in warp function.
            Defaults to value of
            :math:`1 cdot 10^{-3}`.
        max_iterations : int, optional
            Exit criterion for number of Gauss-Newton iterations.
            Defaults to value of 15.
        mesh_order : int, optional
            Mesh element order. Options are 1 and 2. Defaults to 2.
        subset_order : int, optional
            Warp function order. Options are 1 and 2. Defaults to 1.
        tolerance: float, optional
            Correlation coefficient tolerance. Defaults to a value of 0.7.
        seed_tolerance: float, optional
            Correlation coefficient tolerance for the seed subset.
            Defaults to a value of 0.9.
        method : str
            Solution method. Options are FAGN and ICGN.
            Default is ICGN since it is faster.
        adaptive_iterations : int, optional
            Number of mesh adaptivity iterations to perform.
            Defaults to a value of 0.
        alpha : float, optional
            Mesh adaptivity control parameter.
            Defaults to a value of 0.5.
        track : bool, optional
            Mesh boundary tracking at reference image updates. Options are:
            False - no movement,
            True - movement of initially defined boundary points tracked.
        hard_boundary : bool, optional
            Boolean to control whether the boundary is included in the
            binary mask. True -included, False - not included.
            Defaults to True.
        subset_size_compensation: bool, optional
            Boolean to control whether masked subsets are enlarged to
            maintain area (and thus better correlation).
            Defaults to False.

        Returns
        -------

        solved : bool
            Boolean to indicate if the subset instance has been solved.
        """
        # Check if solved.
        if self.data["solved"] is True:
            log.error("Sequence has already been solved. Cannot be solved again.")
            return

        # Check inputs.
        if type(seed_coord) != np.ndarray:
            log.warning(
                (
                    "`seed_coord` keyword argument type invalid. "
                    "Expected a `numpy.ndarray`, but got a `{type2}`.\n"
                    "Selecting `seed_coord`..."
                ).format(type2=type(seed_coord).__name__)
            )
            seed_coord = gp.gui.selectors.coordinate.CoordinateSelector()
        elif np.shape(seed_coord)[0] != 2:
            log.warning(
                (
                    "`see_coord` keyword argument primary axis size invalid. "
                    "Expected 2, but got {size}.\nSelecting `seed_coord`..."
                ).format(size=np.shape(seed_coord)[0])
            )
            seed_coord = gp.gui.selectors.coordinate.CoordinateSelector()
        elif seed_coord.ndim != 1:
            log.warning(
                (
                    "`seed_coord` keyword argument dimensions invalid. "
                    "Expected 1, but got {size}."
                ).format(size=seed_coord.ndim)
            )
            seed_coord = gp.gui.selectors.coordinate.CoordinateSelector()
        if type(adaptive_iterations) != int:
            log.warning(
                (
                    "`adaptive_iterations` keyword argument type invalid. "
                    "Expected an `int`, but got a `{type2}`.\n"
                    "Attempting conversion..."
                ).format(type2=type(adaptive_iterations).__name__)
            )
            try:
                adaptive_iterations = int(adaptive_iterations)
                log.warning(
                    (
                        "`adaptive_iterations` keyword argument "
                        "type conversion successful. "
                        "New value: {value}"
                    ).format(value=adaptive_iterations)
                )
            except ValueError:
                log.error(
                    "`adaptive_iterations` keyword argument type conversion failed. "
                    "Input an `int` >= 0."
                )
                raise TypeError(
                    "`adaptive_iterations` keyword argument type conversion failed. "
                    "Input an `int` >= 0."
                )
        elif adaptive_iterations < 0:
            log.error(
                (
                    "`adaptive_iterations` keyword argument value {value} "
                    "out of range. Input an `int` >= 0."
                ).format(value=adaptive_iterations)
            )
            raise ValueError(
                (
                    "`adaptive_iterations` keyword argument value {value} "
                    "out of range. Input an `int` >= 0."
                ).format(value=adaptive_iterations)
            )
        if template is None:
            template = gp.templates.Circle(50)
        elif (
            type(template) != gp.templates.Circle
            and type(template) != gp.templates.Square
        ):
            log.error(
                (
                    "`template` keyword argument value invalid. "
                    "Expected `gp.templates.Circle` or `gp.templates.Square`, "
                    "but got {type3}."
                ).format(type3=type(template).__name__)
            )
            raise ValueError(
                (
                    "`template` keyword argument value invalid. "
                    "Expected `gp.templates.Circle` or `gp.templates.Square`, "
                    "but got {type3}."
                ).format(type3=type(template).__name__)
            )
        if type(max_norm) != float:
            log.warning(
                (
                    "`max_norm` keyword argument type invalid. "
                    "Expected a `float`, but got a `{type2}`.\n"
                    "Attempting conversion..."
                ).format(type2=type(max_norm).__name__)
            )
            try:
                max_norm = float(max_norm)
                log.warning(
                    (
                        "`max_norm` keyword argument type conversion successful. "
                        "New value: {value}"
                    ).format(value=max_norm)
                )
            except ValueError:
                log.error(
                    "`max_norm` keyword argument type conversion failed. "
                    "Input a `float` > 0.0."
                )
                raise TypeError(
                    "`max_norm` keyword argument type conversion failed. "
                    "Input a `float` > 0.0."
                )
        elif max_norm <= 0.0:
            log.error(
                (
                    "`max_norm` keyword argument value {value} out of range >0.0. "
                    "Input a `float` > 0.0."
                ).format(value=max_norm)
            )
            raise ValueError(
                (
                    "`max_norm` keyword argument value {value} out of range >0.0. "
                    "Input a `float` > 0.0."
                ).format(value=max_norm)
            )
        if type(max_iterations) != int:
            log.warning(
                (
                    "`max_iterations` keyword argument type invalid. "
                    "Expected an `int`, but got a `{type2}`.\n"
                    "Attempting conversion..."
                ).format(type2=type(max_iterations).__name__)
            )
            try:
                max_iterations = int(max_iterations)
                log.warning(
                    (
                        "`max_iterations` keyword argument type conversion successful. "
                        "New value: {value}"
                    ).format(value=max_iterations)
                )
            except ValueError:
                log.error(
                    "`max_iterations` keyword argument type conversion failed. "
                    "Input an `int` > 0."
                )
                raise TypeError(
                    "`max_iterations` keyword argument type conversion failed. "
                    "Input an `int` > 0."
                )
        elif max_iterations <= 0:
            log.error(
                (
                    "`max_iterations` keyword argument value {value} out of range. "
                    "Input an `int` >= 0."
                ).format(value=max_iterations)
            )
            raise ValueError(
                (
                    "`max_iterations` keyword argument value {value} out of range. "
                    "Input an `int` >= 0."
                ).format(value=max_iterations)
            )
        if type(mesh_order) != int:
            log.error(
                (
                    "`mesh_order` keyword argument type invalid. "
                    "Expected an `int`, but got a {type3}."
                ).format(type3=type(mesh_order).__name__)
            )
            raise TypeError(
                (
                    "`mesh_order` keyword argument type invalid. "
                    "Expected an `int`, but got a {type3}."
                ).format(type3=type(mesh_order).__name__)
            )
        if mesh_order != 1 and mesh_order != 2:
            log.error(
                (
                    "`mesh_order` keyword argument value invalid. "
                    "Expected 1 or 2, but got {value}."
                ).format(value=mesh_order)
            )
            raise ValueError(
                (
                    "`mesh_order` keyword argument value invalid. "
                    "Expected 1 or 2, but got {value}."
                ).format(value=mesh_order)
            )
        if type(subset_order) != int:
            log.error(
                (
                    "`subset_order` keyword argument type invalid. "
                    "Expected a `int` , but got a {type3}."
                ).format(type3=type(subset_order).__name__)
            )
            raise TypeError(
                (
                    "`subset_order` keyword argument type invalid. "
                    "Expected a `int`, but got a {type3}."
                ).format(type3=type(subset_order).__name__)
            )
        if subset_order != 1 and subset_order != 2:
            log.error(
                (
                    "`subset_order` keyword argument value invalid. "
                    "Expected 1 or 2, but got {value}."
                ).format(value=subset_order)
            )
            raise ValueError(
                (
                    "`subset_order` keyword argument value invalid. "
                    "Expected 1 or 2, but got {value}."
                ).format(value=subset_order)
            )
        if type(method) != str:
            log.error(
                (
                    "`method` keyword argument type invalid. "
                    "Expected a `str` , but got a {type3}."
                ).format(type3=type(method).__name__)
            )
            raise TypeError(
                (
                    "`method` keyword argument type invalid. "
                    "Expected a `str`, but got a {type3}."
                ).format(type3=type(method).__name__)
            )
        elif method not in ["FAGN", "ICGN"]:
            log.error(
                (
                    "`method` keyword argument value invalid. "
                    "Expected `FAGN` or `ICGN`, but got {value}."
                ).format(value=method)
            )
            raise ValueError(
                (
                    "`method` keyword argument value invalid. "
                    "Expected `FAGN` or `ICGN`, but got {value}."
                ).format(value=method)
            )
        if type(tolerance) != float:
            log.warning(
                (
                    "`tolerance` keyword argument type invalid. "
                    "Expected a `float`, but got a `{type2}`.\n"
                    "Attempting conversion..."
                ).format(type2=type(tolerance).__name__)
            )
            try:
                tolerance = float(tolerance)
                log.warning(
                    (
                        "`tolerance` keyword argument type conversion successful. "
                        "New value: {value}"
                    ).format(value=tolerance)
                )
            except ValueError:
                log.error(
                    "`tolerance` keyword argument type conversion failed. "
                    "Input a `float`, 0.0-1.0."
                )
                raise TypeError(
                    "`tolerance` keyword argument type conversion failed. "
                    "Input a `float`, 0.0-1.0."
                )
        elif tolerance <= 0.0 or tolerance > 1.0:
            log.error(
                (
                    "`tolerance` keyword argument value {value} out of range 0.0-1.0. "
                    "Input a `float`, 0.0-1.0."
                ).format(value=tolerance)
            )
            raise ValueError(
                (
                    "`tolerance` keyword argument value {value} out of range 0.0-1.0. "
                    "Input a `float`, 0.0-1.0."
                ).format(value=tolerance)
            )
        if type(seed_tolerance) != float:
            log.warning(
                (
                    "`seed_tolerance` keyword argument type invalid. "
                    "Expected a `float`, but got a `{type2}`.\n"
                    "Attempting conversion..."
                ).format(type2=type(seed_tolerance).__name__)
            )
            try:
                seed_tolerance = float(seed_tolerance)
                log.warning(
                    (
                        "`seed_tolerance` keyword argument type conversion successful. "
                        "New value: {value}"
                    ).format(value=seed_tolerance)
                )
            except ValueError:
                log.error(
                    "`seed_tolerance` keyword argument type conversion failed. "
                    "Input a `float`, 0.0-1.0."
                )
                raise TypeError(
                    "`seed_tolerance` keyword argument type conversion failed. "
                    "Input a `float`, 0.0-1.0."
                )
        elif seed_tolerance <= 0.0 or seed_tolerance > 1.0:
            log.error(
                (
                    "`seed_tolerance` keyword argument value {value} "
                    "out of range 0.0-1.0. "
                    "Input a `float`, 0.0-1.0."
                ).format(value=seed_tolerance)
            )
            raise ValueError(
                (
                    "`seed_tolerance` keyword argument value {value} "
                    "out of range 0.0-1.0. "
                    "Input a `float`, 0.0-1.0."
                ).format(value=seed_tolerance)
            )
        if seed_tolerance < tolerance:
            log.error(
                (
                    "`seed_tolerance`<`tolerance`: {value1}<{value2}. "
                    "Expected `seed_tolerance`>=`tolerance`."
                ).format(value1=seed_tolerance, value2=tolerance)
            )
            raise ValueError(
                (
                    "`seed_tolerance`<`tolerance`: {value1}<{value2}. "
                    "Expected `seed_tolerance`>=`tolerance`."
                ).format(value1=seed_tolerance, value2=tolerance)
            )
        if type(alpha) != float:
            log.warning(
                (
                    "`alpha` keyword argument type invalid. "
                    "Expected a `float`, but got a `{type2}`.\n"
                    "Attempting conversion..."
                ).format(type2=type(alpha).__name__)
            )
            try:
                alpha = float(alpha)
                log.warning(
                    (
                        "`alpha` keyword argument type conversion successful. "
                        "New value: {value}"
                    ).format(value=alpha)
                )
            except ValueError:
                log.error(
                    "`alpha` keyword argument type conversion failed. "
                    "Input a `float`, 0.0-1.0."
                )
                raise TypeError(
                    "`alpha` keyword argument type conversion failed. "
                    "Input a `float`, 0.0-1.0."
                )
        elif alpha < 0.0 or alpha > 1.0:
            log.error(
                (
                    "`alpha` keyword argument value {value} out of range 0.0-1.0. "
                    "Input a `float`, 0.0-1.0."
                ).format(value=alpha)
            )
            raise ValueError(
                (
                    "`alpha` keyword argument value {value} out of range 0.0-1.0. "
                    "Input a `float`, 0.0-1.0."
                ).format(value=alpha)
            )
        if type(track) != bool:
            log.error(
                (
                    "`track` keyword argument type invalid. "
                    "Expected a `bool`, but got a `{type2}`."
                ).format(type2=type(track).__name__)
            )
            raise TypeError(
                (
                    "`track` keyword argument type invalid. "
                    "Expected a `bool`, but got a `{type2}`."
                ).format(type2=type(track).__name__)
            )
        if type(hard_boundary) != bool:
            log.error(
                (
                    "`hard_boundary` keyword argument type invalid. "
                    "Expected a `bool`, but got a {type3}."
                ).format(type3=type(hard_boundary).__name__)
            )
            raise TypeError(
                (
                    "`hard_boundary` keyword argument type invalid. "
                    "Expected a `bool`, but got a {type3}."
                ).format(type3=type(hard_boundary).__name__)
            )
        if type(subset_size_compensation) != bool:
            log.error(
                (
                    "`subset_size_compensation` keyword argument type invalid. "
                    "Expected a `bool`, but got a {type3}."
                ).format(type3=type(subset_size_compensation).__name__)
            )
            raise TypeError(
                (
                    "`subset_size_compensation` keyword argument type invalid. "
                    "Expected a `bool`, but got a {type3}."
                ).format(type3=type(subset_size_compensation).__name__)
            )

        # Store variables.
        self._seed_coord = seed_coord
        self._template = template
        self._max_iterations = max_iterations
        self._max_norm = max_norm
        self._adaptive_iterations = adaptive_iterations
        self._method = method
        self._subset_order = subset_order
        self._mesh_order = mesh_order
        self._tolerance = tolerance
        self._seed_tolerance = seed_tolerance
        self._alpha = alpha
        self._track = track
        self._hard_boundary = hard_boundary
        self._subset_size_compensation = subset_size_compensation
        self._p_0 = np.zeros(6 * self._subset_order)

        # Solve.
        _f_index = 0
        _g_index = 1
        _f_img = gp.image.Image(self._images[_f_index])
        _g_img = gp.image.Image(self._images[_g_index])
        while _g_index < len(self._image_indices):
            log.info(
                "Solving for image pair {}-{}.".format(
                    self._image_indices[_f_index],
                    self._image_indices[_g_index],
                )
            )
            mesh = gp.mesh.Mesh(
                f_img=_f_img,
                g_img=_g_img,
                target_nodes=self._target_nodes,
                boundary=self._boundary,
                exclusions=self._exclusions,
                size_lower_bound=self._size_lower_bound,
                size_upper_bound=self._size_upper_bound,
                mesh_order=self._mesh_order,
                hard_boundary=self._hard_boundary,
                subset_size_compensation=self._subset_size_compensation,
            )  # Initialise mesh object.
            mesh.solve(
                seed_coord=self._seed_coord,
                template=self._template,
                max_iterations=self._max_iterations,
                max_norm=self._max_norm,
                adaptive_iterations=self._adaptive_iterations,
                method=self._method,
                subset_order=self._subset_order,
                tolerance=self._tolerance,
                seed_tolerance=self._seed_tolerance,
                alpha=self._alpha,
            )  # Solve mesh.
            if mesh.solved:
                if self._save_by_reference:
                    gp.io.save(
                        object=mesh,
                        filename=self._mesh_folder
                        + "mesh_"
                        + str(self._image_indices[_f_index])
                        + "_"
                        + str(self._image_indices[_g_index]),
                    )
                    self.data["meshes"].append(
                        "mesh_"
                        + str(self._image_indices[_f_index])
                        + "_"
                        + str(self._image_indices[_g_index])
                    )
                else:
                    self.data["meshes"][_g_index - 1] = mesh.data
                if track:
                    self._boundary_tags = mesh._boundary_tags
                    self._exclusions_tags = mesh._exclusions_tags
                _g_index += 1  # Iterate the target image index.
                del _g_img
                if _g_index != len(self._image_indices):
                    _g_img = gp.image.Image(self._images[_g_index])
                else:
                    self.solved = True
            elif _f_index + 1 < _g_index:
                _f_index = _g_index - 1
                if self._track:
                    self._tracking(_f_index)
                del _f_img
                _f_img = gp.image.Image(self._images[_f_index])
            else:
                log.error(
                    "Mesh for consecutive image pair {a}-{b} is unsolvable. "
                    "Sequence curtailed.".format(
                        a=self._image_indices[_f_index],
                        b=self._image_indices[_g_index],
                    )
                )
                self.data["meshes"] = np.asarray(self.data["meshes"])
                self._unsolvable = True
                del mesh
                return self.solved
            del mesh
        del _f_img

        # Pack data.
        self.data["solved"] = self.solved
        self.data["unsolvable"] = self._unsolvable
        self.data["meshes"] = np.asarray(self.data["meshes"])
        return self.solved

    def _tracking(self, _f_index):
        """
        Private method for tracking the movement of the mesh boundary
        and exclusions upon reference image updates.
        """

        log.info("Tracing boundary and exclusion displacements.")
        if self._save_by_reference:
            previous_mesh = gp.io.load(
                filename=self._mesh_folder + self.data["meshes"][_f_index - 1]
            ).data
        else:
            previous_mesh = self.data["meshes"][_f_index - 1]
        self._boundary = (
            previous_mesh["nodes"][self._boundary_tags]
            + previous_mesh["results"]["displacements"][self._boundary_tags]
        )
        _exclusions = []
        for i in range(np.shape(self._exclusions)[0]):
            _exclusions.append(
                previous_mesh["nodes"][self._exclusions_tags[i]]
                + previous_mesh["results"]["displacements"][self._exclusions_tags[i]]
            )
        self._exclusions = _exclusions


class SequenceResults(SequenceBase):
    """
    SequenceResults class for geopyv.

    Parameters
    ----------
    data : dict
        geopyv data dict from Sequence object.


    Attributes
    ----------
    data : dict
        geopyv data dict from Sequence object.

    """

    def __init__(self, data):
        """Initialisation of geopyv SequenceResults class."""
        self.data = data
