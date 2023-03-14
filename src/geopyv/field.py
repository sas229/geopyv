"""

Field module for geopyv.

"""
import logging
import numpy as np
import geopyv as gp
from geopyv.object import Object
import gmsh
from scipy.optimize import minimize_scalar


log = logging.getLogger(__name__)


class FieldBase(Object):
    """
    Field base class to be used as a mixin.
    """

    def __init__(self):
        super().__init__(object_type="Field")
        """

        Field base class initialiser.

        """

    def inspect(self, mesh=True, show=True, block=True, save=None):
        """
        Method to show the particles and associated representative areas.

        Parameters
        ----------
        mesh : bool, optional
            Control whether the mesh is plotted.
            Defaults to True.
        show : bool, optional
            Control whether the plot is displayed.
        block : bool, optional
            Control whether the plot blocks execution until closed.
        save : str, optional
            Name to use to save plot. Uses default extension of `.png`.
        """

        # Check inputs.
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

        log.info("Inspecting field...")
        fig, ax = gp.plots.inspect_field(
            self.data, mesh=mesh, show=show, block=block, save=save
        )
        return fig, ax

    def trace(
        self,
        quantity="warps",
        component=0,
        imshow=True,
        colorbar=True,
        ticks=None,
        alpha=0.75,
        axis=None,
        xlim=None,
        ylim=None,
        show=True,
        block=True,
        save=None,
    ):
        """
        Method to plot an incremental quantity along the particle position path.

        Parameters
        ----------
        quantity : str, optional
            Specifier for which metric to plot along the particle path.
        component : int, optional
            Specifier for which component of the metric to plot along the particle path.
        imshow : bool, optional
            Control whether the reference image is plotted.
            Defaults to True.
        colorbar : bool, optional
            Control whether the colour bar is plotted.
            Defaults to True.
        ticks : list, optional
            Overwrite default colourbar ticks.
            Defaults to None.
        alpha : float, optional
            Control contour opacity. Must be between 0.0-1.0.
            Defaults to 0.75.
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
        """

        # Check if solved.
        if self.data["solved"] is not True or "results" not in self.data:
            log.error(
                "Particle not yet solved therefore no convergence data to plot. "
                "First, run :meth:`~geopyv.particle.Particle.solve()` to solve."
            )
            raise ValueError(
                "Particle not yet solved therefore no convergence data to plot. "
                "First, run :meth:`~geopyv.particle.Particle.solve()` to solve."
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
            "coordinates",
            "warps",
            "volumes",
            "stresses",
        ]:
            log.error(
                (
                    "`quantity` keyword argument value invalid. "
                    "Expected `coordinates`, `warps`, `volumes`, `stresses`, "
                    "but got {value}."
                ).format(value=quantity)
            )
            raise ValueError(
                (
                    "`quantity` keyword argument value invalid. "
                    "Expected `coordinates`, `warps`, `volumes`, `stresses`, "
                    "but got {value}."
                ).format(value=quantity)
            )
        if type(component) != int:
            log.error(
                (
                    "`component` keyword argument type invalid. "
                    "Expected a `int`, but got a {type3}."
                ).format(type3=type(component).__name__)
            )
            raise TypeError(
                (
                    "`component` keyword argument type invalid. "
                    "Expected a `int`, but got a {type3}."
                ).format(type3=type(component).__name__)
            )
        if component >= np.shape(self.data["results"][quantity])[1]:
            log.error(
                (
                    "`component` {input_value} is out of bounds for "
                    "axis 0 with size {max_value}."
                ).format(
                    max_value=np.shape(self.data["results"][quantity])[1] - 1,
                    input_value=component,
                )
            )
            raise IndexError(
                (
                    "`component` {input_value} is out of bounds for "
                    "axis 0 with size {max_value}."
                ).format(
                    max_value=np.shape(self.data["results"][quantity])[1] - 1,
                    input_value=component,
                )
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

        log.info("Tracing field...")
        fig, ax = gp.plots.trace_particle(
            data=self.data,
            quantity=quantity,
            component=component,
            imshow=imshow,
            colorbar=True,
            ticks=ticks,
            alpha=alpha,
            axis=axis,
            xlim=xlim,
            ylim=ylim,
            show=show,
            block=block,
            save=save,
        )
        return fig, ax


class Field(FieldBase):
    def __init__(
        self,
        *,
        series=None,
        target_particles=1000,
        track=True,
        boundary=None,
        exclusions=[],
        coordinates=None,
        volumes=None,
    ):
        """
        Initialisation of geopyv field object.

        Parameters
        ----------
        series : gp.sequence.Sequence object or gp.mesh.Mesh object
            The base series for field object interpolation.
        target_particles : int, optional
            Target number of particles. Defaults to a value of 1000.
        track : bool, optional
            Boolean to specify if particles should move or remain static.
            True: move (Lagrangian), False: static (Eularian). Defaults to True.
        boundary : numpy.ndarray (N,2), optional
            Array of coordinates to define the particle auto-distribution mesh
            boundary.
        exclusions : list, optional
            List to define the particle auto-distribution mesh exclusions.
            Shape of `numpy.ndarray` (N,2).
        coordinates : numpy.ndarray (N,2), optional
            Array of coordinates to define the initial particle positions.
        volumes : numpy.ndarray (N,), optional
            Array of volumes for particle representation.
            Defaults to np.ones(N) i.e. measure of volumetric strain.

        Note ::
        Two kwargs groups for particle distribution:
        if coordinates is not None:
            1. User-defined : coordinates, volumes.
        else:
            2. Auto-distributed: boundary, exclusions, target_particles.

        Attributes
        ----------
        data : dict
            Data object containing all settings and results.
            See the data structure :ref:`here <field_data_structure>`.
        solved : bool
            Boolean to indicate if the field has been solved.

        """
        self._initialised = False
        _auto_distribute = True

        # Check inputs.
        if series.data["type"] not in ["Sequence", "Mesh"]:
            log.error(
                (
                    "`series` keyword argument type invalid. "
                    "Expected `gp.sequence.Sequence`, `gp.mesh.Mesh, "
                    "but got {value}."
                ).format(value=type(series).__name__)
            )
            raise ValueError(
                (
                    "`series` keyword argument type invalid. "
                    "Expected `gp.sequence.Sequence` or `gp.mesh.Mesh, "
                    "but got {value}."
                ).format(value=type(series).__name__)
            )
        if type(target_particles) != int and coordinates is None:
            log.warning(
                (
                    "`target_particles` keyword argument type invalid. "
                    "Expected an `int`, but got a `{type2}`.\n"
                    "Attempting conversion..."
                ).format(type2=type(target_particles).__name__)
            )
            try:
                target_particles = int(target_particles)
                log.warning(
                    (
                        "`target_particles` keyword argument type "
                        "conversion successful. "
                        "New value: {value}"
                    ).format(value=target_particles)
                )
            except ValueError:
                log.error(
                    "`target_particles` keyword argument type conversion failed. "
                    "Input an `int` > 0."
                )
                raise TypeError(
                    "`target_particles` keyword argument type conversion failed. "
                    "Input an `int` > 0."
                )
        elif target_particles <= 0 and coordinates is None:
            log.error(
                (
                    "`target_particles` keyword argument value {value} out of range. "
                    "Input an `int` > 0."
                ).format(value=target_particles)
            )
            raise ValueError(
                (
                    "`target_particles` keyword argument value {value} out of range. "
                    "Input an `int` > 0."
                ).format(value=target_particles)
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
        if type(boundary) != np.ndarray and boundary is not None:
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
        elif np.shape(boundary)[1] != 2 and boundary is not None:
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
        elif boundary.ndim != 2 and boundary is not None:
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
        if type(coordinates) != np.ndarray and coordinates is not None:
            log.warning(
                (
                    "`coordinates` keyword argument type invalid. "
                    "Expected a `numpy.ndarray`, but got a `{type2}`.\n"
                    "Attempting conversion..."
                ).format(type2=type(coordinates).__name__)
            )
            try:
                coordinates = np.asarray(coordinates)
                log.warning(
                    (
                        "`coordinates` keyword argument type conversion successful. "
                        "New value: {value}"
                    ).format(value=coordinates)
                )
            except ValueError:
                log.error(
                    "`coordinates` keyword argument type conversion failed. "
                    "Input a `numpy.ndarray` of shape (Nx,2)."
                )
                raise TypeError(
                    "`coordinates` keyword argument type conversion failed. "
                    "Input a `numpy.ndarray` of shape (Nx,2)."
                )
        elif np.shape(coordinates)[1] != 2 and coordinates is not None:
            log.error(
                (
                    "`coordinates` keyword argument secondary axis size invalid. "
                    "Expected 2, but got {size}."
                ).format(size=np.shape(coordinates)[1])
            )
            raise ValueError(
                (
                    "`coordinates` keyword argument secondary axis size invalid. "
                    "Expected 2, but got {size}."
                ).format(size=np.shape(coordinates)[1])
            )
        elif coordinates.ndim != 2 and coordinates is not None:
            log.error(
                (
                    "`coordinates` keyword argument dimensions invalid. "
                    "Expected 2, but got {size}."
                ).format(size=coordinates.ndim)
            )
            raise ValueError(
                (
                    "`coordinates` keyword argument dimensions invalid. "
                    "Expected 2, but got {size}."
                ).format(size=coordinates.ndim)
            )
        if type(volumes) != np.ndarray and coordinates is not None:
            log.warning(
                (
                    "`volumes` keyword argument type invalid. "
                    "Expected a `numpy.ndarray`, but got a `{type2}`.\n"
                    "Attempting conversion..."
                ).format(type2=type(volumes).__name__)
            )
            try:
                volumes = np.asarray(volumes)
                log.warning(
                    (
                        "`volumes` keyword argument type conversion successful. "
                        "New value: {value}"
                    ).format(value=volumes)
                )
            except ValueError:
                log.error(
                    "`volumes` keyword argument type conversion failed. "
                    "Input a `numpy.ndarray` of shape (Nx,)."
                )
                raise TypeError(
                    "`volumes` keyword argument type conversion failed. "
                    "Input a `numpy.ndarray` of shape (Nx,)."
                )
        elif volumes.ndim != 1 and volumes is not None:
            log.error(
                (
                    "`volumes` keyword argument dimensions invalid. "
                    "Expected 1, but got {size}."
                ).format(size=volumes.ndim)
            )
            raise ValueError(
                (
                    "`volumes` keyword argument dimensions invalid. "
                    "Expected 1, but got {size}."
                ).format(size=volumes.ndim)
            )
        if np.shape(volumes)[0] != np.shape(coordinates)[0] and volumes is not None:
            log.error(
                (
                    "Array shape mismatch. {value1} `volumes` "
                    "given for {value2} `coordinates`."
                ).format(value1=np.shape(volumes)[0], value2=np.shape(coordinates)[0])
            )

        # Store.
        if series.data["type"] == "Sequence":
            self._series_type = "Sequence"
            if series.data["file_settings"]["save_by_reference"]:
                for i in range(np.shape(series.data["meshes"])[0]):
                    series.data["meshes"][i] = gp.io.load(
                        filename=series.data["file_settings"]["mesh_folder"]
                        + series.data["meshes"][i]
                    ).data
            series.data["file_settings"]["save_by_reference"] = False
            mesh_0 = series.data["meshes"][0]
            self._number_images = np.shape(series.data["meshes"])[0] + 1
        else:
            self._series_type = "Mesh"
            mesh_0 = series.data
            self._number_images = 1
        self._image_0 = mesh_0["images"]["f_img"]
        self._track = track
        self._series = series
        self.solved = False
        self._unsolvable = False

        if coordinates is not None:
            _auto_distribute = False
            image = gp.image.Image(self._image_0)
            for coord in coordinates:
                if (
                    coord[0] > np.shape(image.image_gs)[0]
                    or coord[0] < 0
                    or coord[1] > np.shape(image.image_gs)[1]
                    or coord[1] < 0
                ):
                    log.error(
                        (
                            "User-specified coordinate {value} "
                            "outside image boundary."
                        ).format(value=coord)
                    )
                    raise ValueError(
                        (
                            "User-specified coordinate {value} "
                            "outside image boundary."
                        ).format(value=coord)
                    )
            del image
            if volumes is None:
                volumes = np.ones(np.shape(coordinates)[0])
            self._target_particles = np.shape(coordinates)[0]
        else:
            self._target_particles = target_particles

        self.data = {
            "type": "Field",
            "solved": self.solved,
            "series_type": self._series_type,
            "number_images": self._number_images,
            "track": self._track,
            "target_particles": self._target_particles,
            "image_0": self._image_0,
        }

        # Particle distribution.
        if _auto_distribute is True:
            # Extract region of interest.
            if boundary is None:
                self._boundary = mesh_0["boundary"]
            if exclusions == []:
                self._exclusions = mesh_0["exclusions"]
            self._size_lower_bound = mesh_0["size_lower_bound"]
            self._size_upper_bound = mesh_0["size_upper_bound"]

            # Define region of interest.
            (
                self._boundary,
                self._segments,
                self._curves,
                _,
            ) = gp.geometry.meshing._define_RoI(
                gp.image.Image(self._image_0),
                self._boundary,
                self._exclusions,
            )

            # Initialize gmsh if not already initialized.
            if gmsh.isInitialized() == 0:
                gmsh.initialize()
            gmsh.option.setNumber("General.Verbosity", 2)

            log.info(
                "Generating mesh using gmsh with approximately {n} particles.".format(
                    n=self._target_particles
                )
            )

            self._initial_mesh()
            self._distribute_particles()
            log.info(
                "Field generated with {p} particles.".format(p=len(self._coordinates))
            )
            self._field = {
                "nodes": self._nodes,
                "elements": self._elements,
                "coordinates": self._coordinates,
                "volumes": self._volumes,
            }
        else:
            self._coordinates = coordinates
            self._volumes = volumes
            log.info("Using user-specified field.")
            self._field = {
                "coordinates": self._coordinates,
                "volumes": self._volumes,
            }

        self.data.update({"field": self._field})
        self._initialised = True

    def _initial_mesh(self):
        """

        Private method to optimize the element size to generate
        approximately the desired number of elements.

        """

        def f(size):
            return self._uniform_remesh(
                size,
                self._boundary,
                self._segments,
                self._curves,
                self._target_particles,
                self._size_lower_bound,
                self._size_upper_bound,
            )

        minimize_scalar(
            f,
            bounds=(self._size_lower_bound, self._size_upper_bound),
            method="bounded",
        )
        self._update_mesh()
        gmsh.finalize()

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

    def _distribute_particles(self):
        self._coordinates = np.mean(self._nodes[self._elements[:, :3]], axis=1)
        M = np.ones((len(self._elements[:, :3]), 3, 3))
        M[:, 1] = self._nodes[self._elements[:, :3]][:, :, 0]
        M[:, 2] = self._nodes[self._elements[:, :3]][:, :, 1]
        self._volumes = abs(0.5 * np.linalg.det(M))

    def solve(self):
        """
        Method to solve for the field.

        Returns
        -------
        solved : bool
            Boolean to indicate if the particle instances have been solved.
        """

        self._particles = np.empty(len(self._coordinates), dtype=dict)
        for i in range(len(self._coordinates)):
            particle = gp.particle.Particle(
                series=self._series,
                coordinate_0=self._coordinates[i],
                volume_0=self._volumes[i],
                track=self._track,
            )
            _particle_solved = particle.solve()
            if _particle_solved is False:
                self._unsolvable = True
                self.data["unsolvable"] = self._unsolvable
                return self.solved
            self._particles[i] = particle.data["results"]
            del particle
        self.data.update({"particles": self._particles})
        self.solved = True
        self.data["solved"] = self.solved

    @staticmethod
    def _uniform_remesh(
        size,
        boundary,
        segments,
        curves,
        target_particles,
        size_lower_bound,
        size_upper_bound,
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
        target_particles : int
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
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.optimize()
        gmsh.model.mesh.setOrder(2)

        # Get mesh topology.
        _, _, ent = gmsh.model.mesh.getElements(dim=2)
        elements = np.reshape((np.asarray(ent) - 1).flatten(), (-1, 6))
        number_particles = len(elements)
        error = abs(number_particles - target_particles)
        return error


class FieldResults(FieldBase):
    """

    FieldResults class for geopyv.

    Parameters
    ----------
    data : dict
        geopyv data dict from Field object.


    Attributes
    ----------
    data : dict
        geopyv data dict from Field object.

    """

    def __init__(self, data):
        """Initialisation of geopyv FieldResults class."""
        self.data = data
