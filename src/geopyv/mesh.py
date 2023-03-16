"""

Mesh module for geopyv.

"""
import logging
import numpy as np
import scipy as sp
import geopyv as gp
from geopyv.object import Object
import gmsh
from copy import deepcopy
from scipy.optimize import minimize_scalar
import PIL.Image as ImagePIL
import PIL.ImageDraw as ImageDrawPIL
from alive_progress import alive_bar

log = logging.getLogger(__name__)


class MeshBase(Object):
    """

    Mesh base class to be used as a mixin. Contains plot functionality.

    """

    def __init__(self):
        super().__init__(object_type="Mesh")
        """

        Mesh base class initialiser.

        """

    def inspect(self, *, subset_index=None, show=True, block=True, save=None):
        """

        Method to show the mesh and associated subset quality metrics using
        :mod: `~geopyv.plots.inspect_subset` for subsets and
        :mod: `~geopyv.plots.inspect_mesh` for meshes.

        Parameters
        ----------
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

        # Check input.
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
            subset_index < 0 or subset_index >= np.shape(self.data["nodes"])[0]
        ):
            log.error(
                (
                    "`subset_index` {input_value} is out of bounds "
                    "for axis 0 with size {max_value}."
                ).format(
                    max_value=np.shape(self.data["nodes"])[0] - 1,
                    input_value=subset_index,
                )
            )
            raise IndexError(
                (
                    "`subset_index` {input_value} is out of bounds "
                    "for axis 0 with size {max_value}."
                ).format(
                    max_value=np.shape(self.data["nodes"])[0] - 1,
                    input_value=subset_index,
                )
            )
        if type(show) != bool:
            log.error(
                "`show` keyword argument type invalid. "
                "Expected a `bool`, but got a `{type2}`.".format(
                    type2=type(show).__name__
                )
            )
            raise TypeError(
                "`show` keyword argument type invalid. "
                "Expected a `bool`, but got a `{type2}`.".format(
                    type2=type(show).__name__
                )
            )
        if type(block) != bool:
            log.error(
                "`block` keyword argument type invalid. "
                "Expected a `bool`, but got a `{type2}`.".format(
                    type2=type(block).__name__
                )
            )
            raise TypeError(
                "`block` keyword argument type invalid. "
                "Expected a `bool`, but got a `{type2}`.".format(
                    type2=type(block).__name__
                )
            )
        if type(save) != str and save is not None:
            log.error(
                "`save` keyword argument type invalid. "
                "Expected a `str` or `NoneType`, but got a `{type3}`.".format(
                    type3=type(save).__name__
                )
            )
            raise TypeError(
                "`save` keyword argument type invalid. "
                "Expected a `str` or `NoneType`, but got a `{type3}`.".format(
                    type3=type(save).__name__
                )
            )

        # Inspect.
        if subset_index is not None:
            if self.data["solved"]:
                subset_data = self.data["results"]["subsets"][subset_index]
            else:
                subset_data = gp.subset.Subset(
                    f_coord=self._nodes[subset_index],
                    f_img=self._f_img,
                    g_img=self._g_img,
                    template=self._template,
                ).data
            mask = self.data["mask"]
            log.info("Inspecting subset {subset}...".format(subset=subset_index))
            fig, ax = gp.plots.inspect_subset(
                data=subset_data,
                mask=mask,
                show=show,
                block=block,
                save=save,
            )
            return fig, ax
        else:
            log.info("Inspecting mesh...")
            fig, ax = gp.plots.inspect_mesh(
                data=self.data, show=show, block=block, save=save
            )
            return fig, ax

    def convergence(
        self,
        *,
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
        subset_index : int, optional
            Index of the subset to inspect.
            If `None', the convergence plot is for the mesh instead.
        quantity : str, optional
            Selector for histogram convergence property if the convergence
            plot is for mesh. Defaults to `C_ZNCC` if left as default None.
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
                "Mesh not yet solved therefore no convergence data to plot. "
                "First, run :meth:`~geopyv.mesh.Mesh.solve()` to solve."
            )
            raise ValueError(
                "Mesh not yet solved therefore no convergence data to plot. "
                "First, run :meth:`~geopyv.mesh.Mesh.solve()` to solve."
            )

        # Check input.
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
            subset_index < 0 or subset_index >= np.shape(self.data["nodes"])[0]
        ):
            log.error(
                (
                    "`subset_index` {input_value} is out of bounds "
                    "for axis 0 with size {max_value}."
                ).format(
                    max_value=np.shape(self.data["nodes"])[0] - 1,
                    input_value=subset_index,
                )
            )
            raise IndexError(
                (
                    "`subset_index` {input_value} is out of bounds "
                    "for axis 0 with size {max_value}."
                ).format(
                    max_value=np.shape(self.data["nodes"])[0] - 1,
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
                "Generating convergence plots for subset {subset}...".format(
                    subset=subset_index
                )
            )
            fig, ax = gp.plots.convergence_subset(
                self.data["results"]["subsets"][subset_index],
                show=show,
                block=block,
                save=save,
            )
            return fig, ax
        else:
            if quantity is None:
                quantity = "C_ZNCC"
            log.info(
                "Generating {quantity} convergence histogram for mesh...".format(
                    quantity=quantity
                )
            )
            fig, ax = gp.plots.convergence_mesh(
                data=self.data,
                quantity=quantity,
                show=show,
                block=block,
                save=save,
            )
            return fig, ax

    def contour(
        self,
        *,
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
        quantity : str, optional
            Selector for contour parameter. Must be in:
            [`C_ZNCC`, `iterations`, `norm`, `u`, `v`, `u_x`, `v_x`,`u_y`,`v_y`, `R`]
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
            * Can only be used once the mesh has been solved using the
              :meth:`~geopyv.mesh.Mesh.solve` method.

        .. seealso::
            :meth:`~geopyv.plots.contour_mesh`

        """
        # Check if solved.
        if self.data["solved"] is not True:
            log.error(
                "Mesh not yet solved therefore no convergence data to plot. "
                "First, run :meth:`~geopyv.mesh.Mesh.solve()` to solve."
            )
            raise ValueError(
                "Mesh not yet solved therefore no convergence data to plot. "
                "First, run :meth:`~geopyv.mesh.Mesh.solve()` to solve."
            )

        # Check input.
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
                    "Expected `C_ZNCC`, `iterations, `norm`, `u`, `v`, `u_x`, "
                    "`v_x`, `u_y`, `v_y` or `R`, but got {value}."
                ).format(value=quantity)
            )
            raise ValueError(
                (
                    "`quantity` keyword argument value invalid. "
                    "Expected `C_ZNCC`, `iterations, `norm`, `u`, `v`, `u_x`, "
                    "`v_x`, `u_y`, `v_y` or `R`, but got {value}."
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

        # Plot contours.
        log.info(
            "Generating {quantity} contour plot for mesh...".format(quantity=quantity)
        )
        fig, ax = gp.plots.contour_mesh(
            data=self.data,
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
                "Mesh not yet solved therefore no convergence data to plot. "
                "First, run :meth:`~geopyv.mesh.Mesh.solve()` to solve."
            )
            raise ValueError(
                "Mesh not yet solved therefore no convergence data to plot. "
                "First, run :meth:`~geopyv.mesh.Mesh.solve()` to solve."
            )

        # Check inputs.
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

        # Plot quiver.
        log.info("Generating quiver plot for mesh...")
        fig, ax = gp.plots.quiver_mesh(
            data=self.data,
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


class Mesh(MeshBase):
    """
    Mesh class for geopyv.

    Private Attributes
    ------------------
    _f_img : `geopyv.image.Image`
        Reference image of geopyv.image.Image class, instantiated by
        :mod:`~image.Image`.
    _g_img : `geopyv.image.Image`
        Target image of geopyv.image.Image class, instantiated by
        :mod:`~image.Image`.
    _target_nodes : int
        Target number of nodes. Defaults to a value of 1000.
    _boundary : `numpy.ndarray` (Nx, 2)
        Array of coordinates that define the mesh boundary.
    _exclusions : list
        List of `numpy.ndarray` (Nx,2) that define the exclusion boundaries.
    _size_lower_bound : float
        Lower bound on element size.
    _size_upper_bound : float
        Lower bound on element size.
    _mesh_order : int
        Mesh element order. Either 1 or 2.
    _unsolvable : bool
        Boolean to indicate if the mesh cannot be solved.
    _nodes : `numpy.ndarray` (Nx,2)
        Array of subset coordinates.
    _elements : `numpy.ndarray(Nx,3*_mesh_order)
        Node connectivity matrix.

    """

    def __init__(
        self,
        *,
        f_img=None,
        g_img=None,
        target_nodes=1000,
        boundary=None,
        exclusions=[],
        size_lower_bound=1.0,
        size_upper_bound=1000.0,
        mesh_order=2,
        hard_boundary=True,
        subset_size_compensation=False,
    ):
        """

        Initialisation of geopyv mesh object.

        Parameters
        ----------
        f_img : geopyv.image.Image, optional
            Reference image of geopyv.image.Image class, instantiated
            by :mod:`~geopyv.image.Image`.
        g_img : geopyv.image.Image, optional
            Target image of geopyv.imageImage class, instantiated by
            :mod:`~geopyv.image.Image`.
        target_nodes : int, optional
            Target number of nodes. Defaults to a value of 1000.
        boundary : `numpy.ndarray` (N,2)
            Array of coordinates to define the mesh boundary.
        exclusions : list, optional
            List of `numpy.ndarray` to define the mesh exclusions.
        size_lower_bound : float, optional
            Lower bound on element size. Defaults to a value of 1.0.
        size_upper_bound : float, optional
            Lower bound on element size. Defaults to a value of 1000.0.
        mesh_order : int, optional
            Mesh element order. Options are 1 and 2. Defaults to 2.
        hard_boundary : bool, optional
            Boolean to control whether the boundary is included in the
            binary mask. True -included, False - not included.
            Defaults to True.
        subset_size_compensation: bool, optional
            Boolean to control whether masked subsets are enlarged to
            maintain area (and thus better correlation).
            Defaults to False.

        Attributes
        ----------
        data : dict
            Data object containing all settings and results.
            See the data structure :ref:`here <mesh_data_structure>`.
        solved : bool
            Boolean to indicate if the mesh has been solved.

        """
        # Set initialised boolean.
        self.initialised = False

        # Check types.
        if type(f_img) != gp.image.Image:
            f_img = gp.io._load_f_img()
        if type(g_img) != gp.image.Image:
            g_img = gp.io._load_g_img()
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

        # Store.
        self._initialised = False
        self._f_img = f_img
        self._g_img = g_img
        self._target_nodes = target_nodes
        self._boundary = boundary
        self._exclusions = exclusions
        self._size_lower_bound = size_lower_bound
        self._size_upper_bound = size_upper_bound
        self._mesh_order = mesh_order
        self._hard_boundary = hard_boundary
        self._subset_size_compensation = subset_size_compensation
        self.solved = False
        self._unsolvable = False

        # Define region of interest.
        self._define_RoI()

        # Initialize gmsh if not already initialized.
        if gmsh.isInitialized() == 0:
            gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 2)
        # 0: silent except for fatal errors, 1: +errors, 2: +warnings,
        # 3: +direct, 4: +information, 5: +status, 99: +debug.

        # Create initial mesh.
        log.info(
            "Generating mesh using gmsh with approximately {n} nodes.".format(
                n=self._target_nodes
            )
        )
        self._initial_mesh()
        log.info(
            "Mesh generated with {n} nodes and {e} elements.".format(
                n=len(self._nodes), e=len(self._elements)
            )
        )
        self._border_tags()
        gmsh.finalize()

        # Data.
        self.data = {
            "type": "Mesh",
            "solved": self.solved,
            "unsolvable": self._unsolvable,
            "images": {
                "f_img": self._f_img.filepath,
                "g_img": self._g_img.filepath,
            },
            "target_nodes": self._target_nodes,
            "boundary": self._boundary,
            "exclusions": self._exclusions,
            "size_lower_bound": self._size_lower_bound,
            "size_upper_bound": self._size_upper_bound,
            "nodes": self._nodes,
            "elements": self._elements,
            "mask": self._mask,
        }

        self.initialised = True

    def set_target_nodes(self, target_nodes):
        """

        Method to create a mesh with a target number of nodes.

        Parameters
        ----------
        target_nodes : int
            Target number of nodes.


        .. note::
            * This method can be used to update the number of target nodes.
            * It will generate a new initial mesh with the specified target
              number of nodes.

        """

        # Check inputs.
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
        if target_nodes <= 0:
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

        # Store.
        self._target_nodes = target_nodes

        # Initialize gmsh if not already initialized.
        if gmsh.isInitialized() == 0:
            gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 2)

        # Create mesh.
        log.info(
            "Generating mesh using gmsh with approximately {n} nodes.".format(
                n=self._target_nodes
            )
        )
        self._initial_mesh()
        log.info(
            "Mesh generated with {n} nodes and {e} elements.".format(
                n=len(self._nodes), e=len(self._elements)
            )
        )
        gmsh.finalize()

    def solve(
        self,
        *,
        seed_coord=None,
        template=None,
        max_norm=1e-3,
        max_iterations=15,
        subset_order=1,
        tolerance=0.7,
        seed_tolerance=0.9,
        method="ICGN",
        adaptive_iterations=0,
        alpha=0.5,
    ):
        r"""

        Method to solve for the mesh.

        Parameters
        ----------
        seed_coord : `numpy.ndarray` (2,)
            An image coordinate selected in a region of low deformation. The
            reliability-guided approach is initiated from the nearest subset.
        template : `geopyv.templates.Template`
            Subset template object.
        max_norm : float, optional
            Exit criterion for norm of increment in warp function.
            Defaults to value of
            :math:`1 \cdot 10^{-3}`.
        max_iterations : int, optional
            Exit criterion for number of Gauss-Newton iterations.
            Defaults to value
            of 15.
        subset_order : int
            Warp function order. Options are 1 and 2.
            Defaults to a value of 1.
        tolerance: float, optional
            Correlation coefficient tolerance.
            Defaults to a value of 0.7.
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


        Returns
        -------
        solved : bool
            Boolean to indicate if the mesh instance has been solved.

        .. note::
            * For guidance on how to use this class see the
              :ref:`mesh tutorial <Mesh Tutorial>`.

        """
        # Check if solved.
        if self.data["solved"] is True:
            log.error("Mesh has already been solved. Cannot be solved again.")
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
                    "`seed_coord` keyword argument primary axis size invalid. "
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

        # Store variables.
        self._seed_coord = seed_coord
        self._template = template
        self._max_iterations = max_iterations
        self._max_norm = max_norm
        self._adaptive_iterations = adaptive_iterations
        self._method = method
        self._subset_order = subset_order
        self._tolerance = tolerance
        self._seed_tolerance = seed_tolerance
        self._alpha = alpha
        self._subset_bgf_nodes = None
        self._subset_bgf_values = None
        self._update = False
        self._p_0 = np.zeros(6 * self._subset_order)

        # Initialize gmsh if not already initialized.
        if gmsh.isInitialized() == 0:
            gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 2)

        # Create initial mesh.
        self._initial_mesh()

        # Solve initial mesh.
        self._message = "Solving initial mesh"
        self._find_seed_node()
        try:
            self._reliability_guided()
            if self._unsolvable:
                log.error(
                    "Specified correlation coefficient tolerance not met. "
                    "Minimum correlation coefficient: "
                    "{min_C:.2f}; tolerance: {tolerance:.2f}.".format(
                        min_C=np.amin(self._C_ZNCC[np.where(self._C_ZNCC > 0.0)]),
                        tolerance=self._tolerance,
                    )
                )
                return self.solved
            # Solve adaptive iterations.
            for iteration in range(1, adaptive_iterations + 1):
                self._message = "Adaptive iteration {}".format(iteration)
                self._adaptive_mesh()
                self._update_mesh()
                self._adaptive_subset()
                self._find_seed_node()
                self._reliability_guided()
                if self._unsolvable:
                    log.error(
                        "Specified correlation coefficient tolerance not met. "
                        "Minimum correlation coefficient: "
                        "{min_C:.2f}; tolerance: {tolerance:.2f}.".format(
                            min_C=np.amin(self._C_ZNCC[np.where(self._C_ZNCC > 0.0)]),
                            tolerance=self._tolerance,
                        )
                    )
                    return self.solved
            log.info(
                "Solved mesh. Minimum correlation coefficient: {min_C:.2f}; "
                "maximum correlation coefficient: {max_C:.2f}.".format(
                    min_C=np.amin(self._C_ZNCC),
                    max_C=np.amax(self._C_ZNCC),
                )
            )

            # Pack data.
            self.solved = True
            self.data["nodes"] = self._nodes
            self.data["elements"] = self._elements
            self.data["areas"] = self._areas
            self.data["solved"] = self.solved
            self.data["unsolvable"] = self._unsolvable

            # Pack settings.
            self._settings = {
                "max_iterations": self._max_iterations,
                "max_norm": self._max_norm,
                "adaptive_iterations": self._adaptive_iterations,
                "method": self._method,
                "mesh_order": self._mesh_order,
                "tolerance": self._tolerance,
            }
            self.data.update({"settings": self._settings})

            # Extract data from subsets.
            subset_data = []
            for subset in self._subsets:
                subset_data.append(subset.data)

            # Pack results.
            self._results = {
                "subsets": subset_data,
                "displacements": self._displacements,
                "du": self._du,
                "d2u": self._d2u,
                "C_ZNCC": self._C_ZNCC,
            }
            self.data.update({"results": self._results})

        except Exception:
            log.error("Could not solve mesh. Not a correlation issue.")
            self._update = True
            self.solved = False
            self._unsolvable = True
        gmsh.finalize()
        return self.solved

    def _update_mesh(self):
        """

        Private method to update the mesh variables.

        """
        (
            _,
            nc,
            _,
        ) = gmsh.model.mesh.getNodes()  # Extracts: node coordinates.
        _, _, ent = gmsh.model.mesh.getElements(dim=2)  # Extracts: element node tags.
        self._nodes = np.column_stack(
            (nc[0::3], nc[1::3])
        )  # Nodal coordinate array (x,y).
        self._elements = np.reshape(
            (np.asarray(ent) - 1).flatten(), (-1, 6)
        )  # Element connectivity array.

    def _find_seed_node(self):
        """

        Private method to find seed node given seed coordinate.

        """
        dist = np.sqrt(
            (self._nodes[:, 0] - self._seed_coord[0]) ** 2
            + (self._nodes[:, 1] - self._seed_coord[1]) ** 2
        )
        self._seed_node = np.argmin(dist)

    def _define_RoI(self):
        """

        Private method to define the RoI.

        """
        # Create binary mask RoI.
        binary_img = ImagePIL.new(
            "L",
            (
                np.shape(self._f_img.image_gs)[1],
                np.shape(self._f_img.image_gs)[0],
            ),
            0,
        )
        # plt.imshow(np.asarray(binary_img))
        # plt.show()
        if self._hard_boundary:
            ImageDrawPIL.Draw(binary_img).polygon(
                self._boundary.flatten().tolist(), outline=1, fill=1
            )
        else:
            image_edge = np.asarray(
                [
                    [0.0, 0.0],
                    [0.0, np.shape(self._f_img.image_gs)[0]],
                    [
                        np.shape(self._f_img.image_gs)[1],
                        np.shape(self._f_img.image_gs)[0],
                    ],
                    [np.shape(self._f_img.image_gs)[1], 0.0],
                ]
            )
            ImageDrawPIL.Draw(binary_img).polygon(
                image_edge.flatten().tolist(), outline=1, fill=1
            )

        # Create objects for mesh generation.
        self._segments = np.empty(
            (np.shape(self._boundary)[0], 2), dtype=np.int32
        )  # Initiate segment array.
        self._segments[:, 0] = np.arange(
            np.shape(self._boundary)[0], dtype=np.int32
        )  # Fill segment array.
        self._segments[:, 1] = np.roll(self._segments[:, 0], -1)  # Fill segment array.
        self._curves = [list(self._segments[:, 0])]  # Create curve list.

        # Add exclusions.
        self._borders = self._boundary
        for exclusion in self._exclusions:
            ImageDrawPIL.Draw(binary_img).polygon(
                exclusion.flatten().tolist(), outline=1, fill=0
            )  # Add exclusion to binary mask.
            cur_max_idx = np.amax(
                self._segments
            )  # Highest index used by current segments.
            exclusion_segment = np.empty(
                np.shape(exclusion)
            )  # Initiate exclusion segment array.
            exclusion_segment[:, 0] = np.arange(
                cur_max_idx + 1,
                cur_max_idx + 1 + np.shape(exclusion)[0],
            )  # Fill exclusion segment array.
            exclusion_segment[:, 1] = np.roll(
                exclusion_segment[:, 0], -1
            )  # Fill exclusion segment array.
            self._borders = np.append(
                self._borders, exclusion, axis=0
            )  # Append exclusion to boundary array.
            self._segments = np.append(
                self._segments, exclusion_segment, axis=0
            ).astype(
                "int32"
            )  # Append exclusion segments to segment array.
            self._curves.append(
                list(exclusion_segment[:, 0].astype("int32"))
            )  # Append exclusion curve to curve list.

        # Finalise mask.
        self._mask = np.array(binary_img)

    def _border_tags(self):
        """
        Private method to extract the border and exclusions tags from gmsh.
        """
        self._boundary_tags = gmsh.model.occ.getCurveLoops(0)[1][0]
        self._exclusions_tags = []
        for i in range(len(self._exclusions)):
            self._exclusions_tags.append(gmsh.model.occ.getCurveLoops(0)[1][i + 1])

    def _initial_mesh(self):
        """

        Private method to optimize the element size to generate
        approximately the desired number of elements.

        """

        def f(size):
            return self._uniform_remesh(
                size,
                self._borders,
                self._segments,
                self._curves,
                self._target_nodes,
                self._size_lower_bound,
                self._size_upper_bound,
                self._mesh_order,
            )

        minimize_scalar(
            f,
            bounds=(self._size_lower_bound, self._size_upper_bound),
            method="bounded",
        )
        self._update_mesh()

    def _adaptive_mesh(self):
        """

        Private method to perform adaptive remeshing.

        """
        message = "Adaptively remeshing..."
        with alive_bar(dual_line=True, bar=None, title=message) as bar:
            D = (
                abs(self._du[:, 0, 1] + self._du[:, 1, 0]) * self._areas
            )  # Elemental shear strain-area products.
            D_b = np.mean(D)  # Mean elemental shear strain-area product.
            self._areas *= (
                np.clip(D / D_b, self._alpha, 1 / self._alpha)
            ) ** -2  # Target element areas calculated.

            def f(scale):
                return self._adaptive_remesh(
                    scale,
                    self._target_nodes,
                    self._nodes,
                    self._elements,
                    self._areas,
                    self._mesh_order,
                )

            minimize_scalar(f)
            bar()

    @staticmethod
    def _uniform_remesh(
        size,
        boundary,
        segments,
        curves,
        target_nodes,
        size_lower_bound,
        size_upper_bound,
        order,
    ):
        """

        Private method to create the initial mesh.

        Parameters
        ----------
        size : int
            Target size of elements.
        boundary : `numpy.ndarray` (Nx,Ny)
            Array of coordinates to define the mesh boundary.
        segments : `numpy.ndarray` (Nx,Ny)
            Array of segments for gmsh mesh generation.
        curves : `numpy.ndarray` (Nx,Ny)
            Array of curves for gmsh mesh generation.
        target_nodes : int
            Target number of nodes.
        size_lower_bound : int
            Lower bound on element size.


        Returns
        -------
        error : int
            Error between target and actual number of nodes.

        """
        # Make mesh.
        gmsh.model.add("base")  # Create model.

        # Add points.
        for i in range(np.shape(boundary)[0]):
            gmsh.model.occ.addPoint(boundary[i, 0], boundary[i, 1], 0, size, i)

        # Add line segments.
        for i in range(np.shape(segments)[0]):
            gmsh.model.occ.addLine(segments[i, 0], segments[i, 1], i)

        # Add curves.
        for i in range(len(curves)):
            gmsh.model.occ.addCurveLoop(curves[i], i)
        curve_indices = list(np.arange(len(curves), dtype=np.int32))

        # Create surface.
        gmsh.model.occ.addPlaneSurface(curve_indices, 0)

        # Generate mesh.
        gmsh.option.setNumber("Mesh.MeshSizeMin", size_lower_bound)
        gmsh.option.setNumber("Mesh.MeshSizeMax", size_upper_bound)
        gmsh.model.occ.synchronize()

        # Add physical groups.
        for i in range(len(curves)):
            gmsh.model.addPhysicalGroup(1, curves[i], i)

        # Mesh.
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.optimize()
        gmsh.model.mesh.setOrder(order)

        # Get mesh topology.
        (
            _,
            nc,
            _,
        ) = (
            gmsh.model.mesh.getNodes()
        )  # Extracts: node tags, node coordinates, parametric coordinates.
        nodes = np.column_stack((nc[0::3], nc[1::3]))  # Nodal coordinate array (x,y).
        number_nodes = len(nodes)
        # error = target_nodes - number_nodes
        error = number_nodes - target_nodes
        return error

    @staticmethod
    def _adaptive_remesh(scale, target, nodes, elements, areas, order):
        """

        Private method to perform adaptive mesh generation.

        Parameters
        ----------
        scale : float
            Scale factor on element size.
        target : int
            Target number of nodes.
        nodes : `numpy.ndarray`
            Nodes for background mesh.
        elements : `numpy.ndarray`
            Elements for background mesh.
        areas : float
            Target element areas.


        Returns
        -------
        error : float
            Error between target and actual number of nodes.

        """
        lengths = gp.geometry.utilities.area_to_length(
            areas * scale
        )  # Convert target areas to target characteristic lengths.
        bg = gmsh.view.add("bg", 1)  # Create background view.
        data = np.pad(
            nodes[elements[:, :3]],
            ((0, 0), (0, 0), (0, 2)),
            mode="constant",
        )  # Prepare data input (coordinates and buffer).
        data[:, :, 3] = np.reshape(
            np.repeat(lengths, 3), (-1, 3)
        )  # Fill data input buffer with target weights.
        data = np.transpose(data, (0, 2, 1)).flatten()  # Reshape for input.
        gmsh.view.addListData(bg, "ST", len(elements), data)  # Add data to view.
        bgf = gmsh.model.mesh.field.add("PostView")  # Add view to field.
        gmsh.model.mesh.field.setNumber(
            bgf, "ViewTag", bg
        )  # Establish field reference (important for function reuse).
        gmsh.model.mesh.field.setAsBackgroundMesh(bgf)  # Set field as background.
        gmsh.option.setNumber(
            "Mesh.MeshSizeExtendFromBoundary", 0
        )  # Prevent boundary influence on mesh.
        gmsh.option.setNumber(
            "Mesh.MeshSizeFromPoints", 0
        )  # Prevent point influence on mesh.
        gmsh.option.setNumber(
            "Mesh.MeshSizeFromCurvature", 0
        )  # Prevent curve influence on mesh.
        gmsh.option.setNumber("Mesh.Algorithm", 5)
        gmsh.model.mesh.clear()  # Tidy.
        gmsh.model.mesh.generate(2)  # Generate mesh.
        gmsh.model.mesh.optimize()
        gmsh.model.mesh.setOrder(order)

        (
            nt,
            nc,
            npar,
        ) = (
            gmsh.model.mesh.getNodes()
        )  # Extracts: node tags, node coordinates, parametric coordinates.
        nodes = np.column_stack((nc[0::3], nc[1::3]))  # Nodal coordinate array (x,y).
        error = (np.shape(nodes)[0] - target) ** 2
        return error

    def _adaptive_subset(self):
        """

        Private method to compute adaptive subset size.

        .. warning::
            * Implementation not yet complete.

        """
        subset_bgf = sp.interpolate.RBFInterpolator(
            self._subset_bgf_nodes,
            self._subset_bgf_values,
            neighbors=10,
            kernel="cubic",
        )
        subset_bgf(self._nodes)

    def _update_subset_bgf(self):
        """

        Private method to compute the background mesh.

        """
        if self._subset_bgf_nodes is not None:
            self._subset_bgf_nodes = np.append(
                self._subset_bgf_nodes,
                np.mean(self._nodes[self._elements], axis=1),
                axis=0,
            )
            self._subset_bgf_values = np.append(
                self._subset_bgf_values,
                np.mean(self._d2u, axis=(1, 2)),
                axis=0,
            )
        else:
            self._subset_bgf_nodes = np.mean(self._nodes[self._elements], axis=1)
            self._subset_bgf_values = np.mean(self._d2u, axis=(1, 2))

    def _element_area(self):
        """
        Private method to calculate the element areas.

        """
        M = np.ones((len(self._elements), 3, 3))
        M[:, 1] = self._nodes[self._elements[:, :3]][
            :, :, 0
        ]  # [:3] will provide corner nodes in both 1st and 2nd order element case.
        M[:, 2] = self._nodes[self._elements[:, :3]][:, :, 1]
        self._areas = 0.5 * np.linalg.det(M)

    def _element_strains(self):
        """

        Private method to calculate the elemental strain the "B" matrix
        relating element node displacements to elemental strain.

        """
        # Local coordinates
        A = np.ones((len(self._elements), 3, 3))
        A[:, :, 1:] = self._nodes[self._elements[:, :3]]

        # Weighting function (and derivatives to 2nd order).
        dN = np.asarray(
            [
                [1 / 3, 0, -1 / 3, 4 / 3, -4 / 3, 0],
                [0, 1 / 3, -1 / 3, 4 / 3, 0, -4 / 3],
            ]
        )
        d2N = np.asarray(
            [
                [4, 0, 4, 0, 0, -8],
                [0, 0, 4, 4, -4, -4],
                [0, 4, 4, 0, -8, 0],
            ]
        )

        # 1st Order Strains
        J_x_T = dN @ self._nodes[self._elements]
        J_u_T = dN @ self._displacements[self._elements]
        du = np.linalg.inv(J_x_T) @ J_u_T

        # 2nd Order Strains
        d2udzeta2 = d2N @ self._displacements[self._elements]
        J_zeta = np.zeros((len(self._elements), 2, 2))
        J_zeta[:, 0, 0] = (
            self._nodes[self._elements][:, 1, 1] - self._nodes[self._elements][:, 2, 1]
        )
        J_zeta[:, 0, 1] = (
            self._nodes[self._elements][:, 2, 0] - self._nodes[self._elements][:, 1, 0]
        )
        J_zeta[:, 1, 0] = (
            self._nodes[self._elements][:, 2, 1] - self._nodes[self._elements][:, 0, 1]
        )
        J_zeta[:, 1, 1] = (
            self._nodes[self._elements][:, 0, 0] - self._nodes[self._elements][:, 2, 0]
        )
        J_zeta /= np.linalg.det(A)[:, None, None]
        d2u = np.zeros((len(self._elements), 3, 2))
        d2u[:, 0, 0] = (
            d2udzeta2[:, 0, 0] * J_zeta[:, 0, 0] ** 2
            + 2 * d2udzeta2[:, 1, 0] * J_zeta[:, 0, 0] * J_zeta[:, 1, 0]
            + d2udzeta2[:, 2, 0] * J_zeta[:, 1, 0] ** 2
        )
        d2u[:, 0, 1] = (
            d2udzeta2[:, 0, 1] * J_zeta[:, 0, 0] ** 2
            + 2 * d2udzeta2[:, 1, 1] * J_zeta[:, 0, 0] * J_zeta[:, 1, 0]
            + d2udzeta2[:, 2, 1] * J_zeta[:, 1, 0] ** 2
        )
        d2u[:, 1, 0] = (
            d2udzeta2[:, 0, 0] * J_zeta[:, 0, 0] * J_zeta[:, 0, 1]
            + d2udzeta2[:, 1, 0]
            * (J_zeta[:, 0, 0] * J_zeta[:, 1, 1] + J_zeta[:, 1, 0] * J_zeta[:, 0, 1])
            + d2udzeta2[:, 2, 0] * J_zeta[:, 1, 0] * J_zeta[:, 1, 1]
        )
        d2u[:, 1, 1] = (
            d2udzeta2[:, 0, 1] * J_zeta[:, 0, 0] * J_zeta[:, 0, 1]
            + d2udzeta2[:, 1, 1]
            * (J_zeta[:, 0, 0] * J_zeta[:, 1, 1] + J_zeta[:, 1, 0] * J_zeta[:, 0, 1])
            + d2udzeta2[:, 2, 1] * J_zeta[:, 1, 0] * J_zeta[:, 1, 1]
        )
        d2u[:, 2, 0] = (
            d2udzeta2[:, 0, 0] * J_zeta[:, 0, 1] ** 2
            + 2 * d2udzeta2[:, 1, 0] * J_zeta[:, 0, 1] * J_zeta[:, 1, 1]
            + d2udzeta2[:, 2, 0] * J_zeta[:, 1, 1] ** 2
        )
        d2u[:, 2, 1] = (
            d2udzeta2[:, 0, 1] * J_zeta[:, 0, 1] ** 2
            + 2 * d2udzeta2[:, 1, 1] * J_zeta[:, 0, 1] * J_zeta[:, 1, 1]
            + d2udzeta2[:, 2, 1] * J_zeta[:, 1, 1] ** 2
        )

        self._du = du
        self._d2u = d2u

    def _reliability_guided(self):
        """

        Private method to perform reliability-guided (RG) PIV analysis.

        """
        # Set up.
        m = np.shape(self._nodes)[0]
        n = np.shape(self._p_0)[0]
        self._subset_solved = np.zeros(
            m, dtype=int
        )  # Solved/unsolved reference array (1 if unsolved, -1 if solved).
        self._C_ZNCC = np.zeros(m, dtype=np.float64)  # Correlation coefficient array.
        self._subsets = np.empty(m, dtype=object)  # Initiate subset array.
        self._p = np.zeros((m, n), dtype=np.float64)  # Warp function array.
        self._displacements = np.zeros(
            (m, 2), dtype=np.float64
        )  # Displacement output array.

        # All nodes.
        entities = gmsh.model.getEntities()
        self._node_tags = []
        for e in entities:
            tags, _, _ = gmsh.model.mesh.getNodes(e[0], e[1])
            self._node_tags = np.append(self._node_tags, tags.flatten()).astype(int)

        # Interior and boundary nodes.
        entities = gmsh.model.getEntities(2)
        self._interior_node_tags = []
        for e in entities:
            tags, _, _ = gmsh.model.mesh.getNodes(e[0], e[1])
            self._interior_node_tags = np.append(
                self._interior_node_tags, tags.flatten()
            ).astype(int)
        self._borders_node_tags = (
            np.setdiff1d(self._node_tags, self._interior_node_tags).astype(int) - 1
        )

        # Template masking using binary mask.
        for tag in range(len(self._node_tags)):
            if tag in self._borders_node_tags:
                centre = self._nodes[tag]
                template = deepcopy(self._template)
                template.mask(centre, self._mask)
                if self._subset_size_compensation:
                    if template.n_px < self._template.n_px:
                        size = int(
                            self._template.size
                            / np.sqrt(template.n_px / self._template.n_px)
                        )
                        if self._template.shape == "circle":
                            template = gp.templates.Circle(radius=size)
                        elif self._template.shape == "square":
                            template = gp.templates.Square(length=size)
                        template.mask(centre, self._mask)
                self._subsets[tag] = gp.subset.Subset(
                    f_coord=self._nodes[tag],
                    f_img=self._f_img,
                    g_img=self._g_img,
                    template=template,
                )  # Create masked boundary subset.
            else:
                self._subsets[tag] = gp.subset.Subset(
                    f_coord=self._nodes[tag],
                    f_img=self._f_img,
                    g_img=self._g_img,
                    template=self._template,
                )  # Create full subset.
        # Solve subsets in mesh.
        number_nodes = np.shape(self._nodes)[0]
        with alive_bar(
            number_nodes,
            dual_line=True,
            bar="blocks",
            title=self._message,
        ) as self._bar:
            # Solve for seed.
            self._bar.text = "-> Solving seed subset..."
            self._subsets[self._seed_node].solve(
                max_norm=self._max_norm,
                max_iterations=self._max_iterations,
                p_0=self._p_0,
                order=self._subset_order,
                method=self._method,
                tolerance=self._seed_tolerance,
            )  # Solve for seed subset.
            self._bar()

            # If seed not solved, log the error, otherwise store
            # the variables and solve neighbours.
            if not self._subsets[self._seed_node].data["solved"]:
                self._update = True
                log.error(
                    "Specified seed correlation coefficient tolerance not met."
                    "Minimum seed correlation coefficient:"
                    "{seed_C:.2f}; tolerance: {seed_tolerance:.2f}.".format(
                        seed_C=self._subsets[self._seed_node].data["results"]["C_ZNCC"],
                        seed_tolerance=self._seed_tolerance,
                    )
                )
            else:
                self._store_variables(self._seed_node, seed=True)

                # Solve for neighbours of the seed subset.
                p_0 = self._subsets[self._seed_node].data["results"][
                    "p"
                ]  # Set seed subset warp function as the preconditioning.
                self._neighbours(
                    self._seed_node, p_0
                )  # Solve for neighbouring subsets.

                # Solve through sorted queue.
                self._bar.text = (
                    "-> Solving remaining subsets using reliability guided approach..."
                )
                count = 1
                while np.max(self._subset_solved) > -1:
                    # Identify next subset.
                    cur_idx = np.argmax(
                        self._subset_solved * self._C_ZNCC
                    )  # Subset with highest correlation coefficient selected.
                    p_0 = self._subsets[cur_idx].data["results"][
                        "p"
                    ]  # Precondition based on selected subset.
                    self._subset_solved[cur_idx] = -1  # Set as solved.
                    solved = self._neighbours(
                        cur_idx, p_0
                    )  # Calculate for neighbouring subsets.
                    count += 1
                    self._bar()
                    if count == number_nodes:
                        break
                    elif solved is False:
                        break

        # Update check.
        if any(self._subset_solved != -1):
            self._update = True
            self.solved = False
            self._unsolvable = True
        else:
            # Compute element areas and strains.
            self._update = False
            self.solved = True
            self._unsolvable = False
            self._element_area()
            self._element_strains()
            self._update_subset_bgf()

    def _connectivity(self, idx):
        """

        A private method that returns the indices of nodes connected
        to the index node according to the input array.

        Parameters
        ----------
        idx : int
            Index of node.


        Returns
        -------
        pts_idx : `numpy.ndarray` (N)
            Mesh array.

        """
        element_idxs = np.argwhere(self._elements == idx)
        pts_idxs = []
        for i in range(len(element_idxs)):
            if element_idxs[i, 1] == 0:  # If 1
                pts_idxs.append(self._elements[element_idxs[i, 0], 3::2])  # Add 4,6
            elif element_idxs[i, 1] == 1:  # If 2
                pts_idxs.append(self._elements[element_idxs[i, 0], 3:5])  # Add 4,5
            elif element_idxs[i, 1] == 2:  # If 3
                pts_idxs.append(self._elements[element_idxs[i, 0], 4:])  # Add 5,6
            elif element_idxs[i, 1] == 3:  # If 4
                pts_idxs.append(self._elements[element_idxs[i, 0], :2])  # Add 1,2
            elif element_idxs[i, 1] == 4:  # If 5
                pts_idxs.append(self._elements[element_idxs[i, 0], 1:3])  # Add 2,3
            elif element_idxs[i, 1] == 5:  # If 6
                pts_idxs.append(self._elements[element_idxs[i, 0], :3:2])  # Add 1,3
        pts_idxs = np.unique(pts_idxs)

        return pts_idxs

    def _neighbours(self, cur_idx, p_0):
        """

        Private method to calculate the correlation coefficients and
        warp functions of the neighbouring nodes.

        Parameters
        __________
        p_0 : `numpy.ndarray` (N)
            Preconditioning warp function.


        Returns
        -------
        solved : bool
            Boolean to indicate whether the neighbouring subsets have been solved.

        """
        neighbours = self._connectivity(cur_idx)
        for idx in neighbours:
            if self._subset_solved[idx] == 0:  # If not previously solved.
                # Use nearest-neighbout pre-conditioning.
                self._subsets[idx].solve(
                    max_norm=self._max_norm,
                    max_iterations=self._max_iterations,
                    order=self._subset_order,
                    p_0=p_0,
                    method=self._method,
                    tolerance=self._tolerance,
                )
                if self._subsets[idx].data["solved"]:  # Check against tolerance.
                    self._store_variables(idx)
                else:
                    # Try more extrapolated pre-conditioning.
                    diff = self._nodes[idx] - self._nodes[cur_idx]
                    p = self._subsets[cur_idx].data["results"]["p"]
                    if np.shape(p)[0] == 6:
                        p_0[0] = p[0] + p[2] * diff[0] + p[3] * diff[1]
                        p_0[1] = p[1] + p[4] * diff[0] + p[5] * diff[1]
                    elif np.shape(p)[0] == 12:
                        p_0[0] = (
                            p[0]
                            + p[2] * diff[0]
                            + p[4] * diff[1]
                            + 0.5 * p[6] * diff[0] ** 2
                            + 0.5 * p[8] * diff[0] * diff[1]
                            + 0.5 * p[10] * diff[1] ** 2
                        )
                        p_0[1] = (
                            p[1]
                            + p[3] * diff[0]
                            + p[5] * diff[1]
                            + 0.5 * p[7] * diff[0] ** 2
                            + 0.5 * p[9] * diff[0] * diff[1]
                            + 0.5 * p[11] * diff[1] ** 2
                        )
                    self._subsets[idx].solve(
                        max_norm=self._max_norm,
                        max_iterations=self._max_iterations,
                        p_0=p_0,
                        method=self._method,
                        tolerance=self._tolerance,
                        order=self._subset_order,
                    )
                    if self._subsets[idx].data["solved"]:
                        self._store_variables(idx)
                    else:
                        # Finally, try the NCC initial guess.
                        self._subsets[idx].solve(
                            max_norm=self._max_norm,
                            max_iterations=self._max_iterations,
                            p_0=np.zeros(np.shape(p_0)),
                            method=self._method,
                            tolerance=self._tolerance,
                            order=self._subset_order,
                        )
                        if self._subsets[idx].data["solved"]:
                            self._store_variables(idx)
                            return True
                        else:
                            return False

    def _store_variables(self, idx, seed=False):
        """

        Private method to store variables.

        Parameters
        ----------
        idx : int
            Index of current subset.
        seed : bool
            Boolean to indicate whether this is the seed subset.

        """
        if seed is True:
            self._subset_solved[idx] = -1
        elif self._subsets[idx].data["solved"] is False:
            self._subset_solved[idx] = 0
        else:
            self._subset_solved[idx] = 1
        self._C_ZNCC[idx] = np.max(
            (self._subsets[idx].data["results"]["C_ZNCC"], 0)
        )  # Clip correlation coefficient to positive values.
        self._p[idx] = self._subsets[idx].data["results"]["p"].flatten()
        self._displacements[idx, 0] = self._subsets[idx].data["results"]["u"]
        self._displacements[idx, 1] = self._subsets[idx].data["results"]["v"]


class MeshResults(MeshBase):
    """

    MeshResults class for geopyv.

    Parameters
    ----------
    data : dict
        geopyv data dict from Mesh object.


    Attributes
    ----------
    data : dict
        geopyv data dict from Mesh object.

    """

    def __init__(self, data):
        """Initialisation of geopyv MeshResults class."""
        self.data = data
