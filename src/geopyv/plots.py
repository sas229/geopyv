"""

Plots module for geopyv.

"""
import logging
import sys
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.collections import LineCollection
import numpy as np
import re
import geopyv as gp
import pandas as pd
import seaborn as sns
import scipy.spatial as spsp

plt.rcParams["axes.prop_cycle"] = plt.cycler(
    "color", plt.cm.tab20(np.linspace(0, 1, 20))
)
plt.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"
# plt.rcParams['axes.grid'] = True

log = logging.getLogger(__name__)


def inspect_subset(data, mask, show, block, save):
    """

    Function to show the Subset and associated quality metrics.

    Parameters
    ----------
    data : dictgit
        Subset data dict.
    mask : `numpy.ndarray`
        Subset mask.
    show : bool
        Control whether the plot is displayed.
    block : bool
        Control whether the plot blocks execution until closed.
    save : str
        Name to use to save plot. Uses default extension of `.png`.


    Returns
    -------
    fig :  matplotlib.pyplot.figure
        Figure object.
    ax : matplotlib.pyplot.axes
        Axes object.


    .. note::
        * The figure and axes objects can be returned allowing standard matplotlib
          functionality to be used to augment the plot generated. See the
          :ref:`plots tutorial <Plots Tutorial>` for guidance.

    .. seealso::
        :meth:`~geopyv.subset.SubsetBase.inspect`

    """
    # Load data.
    image = cv2.imread(data["images"]["f_img"], cv2.IMREAD_COLOR)
    image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gs = cv2.GaussianBlur(image_gs, ksize=(5, 5), sigmaX=1.1, sigmaY=1.1)
    x = data["position"]["x"]
    y = data["position"]["y"]
    template_size = data["template"]["size"]
    sigma_intensity = data["quality"]["sigma_intensity"]
    SSSIG = data["quality"]["SSSIG"]
    title = "Inspect subset: f_coord = ({x},{y}) (px)".format(x=x, y=y)

    # Crop image and mask.
    x_min = (np.round(x, 0) - template_size).astype(int)
    x_max = (np.round(x, 0) + template_size).astype(int)
    y_min = (np.round(y, 0) - template_size).astype(int)
    y_max = (np.round(y, 0) + template_size).astype(int)
    image = image_gs.astype(np.float32)[y_min : y_max + 1, x_min : x_max + 1]
    if not isinstance(mask, type(None)):
        if type(mask) == np.ndarray:
            if np.shape(mask) == np.shape(image_gs):
                mask = mask.astype(np.float32)[y_min : y_max + 1, x_min : x_max + 1]
                invert_mask = np.abs(mask - 1) * 255
                image = np.maximum(image, invert_mask)

    # If a circular subset, mask pixels outside radius.
    if data["template"]["shape"] == "circle":
        x, y = np.meshgrid(
            np.arange(-template_size, template_size + 1, 1),
            np.arange(-template_size, template_size + 1, 1),
        )
        dist = np.sqrt(x**2 + y**2)
        mask = np.zeros(image.shape)
        mask[dist > template_size] = 255
        image = np.maximum(image, mask)

    # Plot figure.
    fig, ax = plt.subplots(num=title)
    ax.imshow(
        image,
        cmap="gist_gray",
        interpolation="nearest",
        aspect="equal",
        extent=(-0.5, 6.5, -0.5, 5.5),
    )
    size_str = r"Size: {} (px); ".format(template_size)
    sigma_str = r"Quality metrics: $\sigma_s$ = {:.2f} (-);".format(sigma_intensity)
    SSSIG_str = r"SSSIG = {:.2E} (-)".format(SSSIG)
    ax.text(
        3.0,
        -1.0,
        size_str + sigma_str + SSSIG_str,
        horizontalalignment="center",
    )
    ax.set_axis_off()
    plt.tight_layout()

    # Save
    if save is not None:
        plt.savefig(save, dpi=600)

    # Show or close.
    if show is True:
        plt.show(block=block)
    else:
        plt.close(fig)

    return fig, ax


def convergence_subset(data, show, block, save):
    """

    Function to plot Subset convergence.

    Parameters
    ----------
    data : dict
        Subset data dict.
    show : bool
        Control whether the plot is displayed.
    block : bool
        Control whether the plot blocks execution until closed.
    save : str
        Name to use to save plot. Uses default extension of `.png`.


    Returns
    -------
    fig :  matplotlib.pyplot.figure
        Figure object.
    ax : matplotlib.pyplot.axes
        Axes object.


    .. note::
        * The figure and axes objects can be returned allowing standard matplotlib
          functionality to be used to augment the plot generated. See the
          :ref:`plots tutorial <Plots Tutorial>` for guidance.

    .. seealso::
        :meth:`~geopyv.subset.SubsetBase.convergence`

    """
    # Load data.
    history = data["results"]["history"]
    max_iterations = data["settings"]["max_iterations"]
    max_norm = data["settings"]["max_norm"]
    tolerance = data["settings"]["tolerance"]
    x = data["position"]["x"]
    y = data["position"]["y"]

    # Create plot.
    title = "Subset convergence: f_coord = ({x},{y}) (px)".format(x=x, y=y)
    fig, ax = plt.subplots(2, 1, sharex=True, num=title)
    ax[0].semilogy(
        history[0, :],
        history[1, :],
        marker="o",
        clip_on=True,
        label="Convergence",
    )
    ax[0].plot(
        [1, max_iterations],
        [max_norm, max_norm],
        "--r",
        label="Threshold",
    )
    ax[0].set_ylabel(r"$\Delta$ Norm (-)")
    ax[0].set_ylim(max_norm / 1000, max_norm * 1000)
    ax[0].set_yticks(
        [
            max_norm * 1000,
            max_norm * 100,
            max_norm * 10,
            max_norm,
            max_norm / 10,
            max_norm / 100,
            max_norm / 1000,
        ]
    )
    ax[1].plot(
        history[0, :],
        history[2, :],
        marker="o",
        clip_on=True,
        label="Convergence",
    )
    ax[1].plot(
        [1, max_iterations],
        [tolerance, tolerance],
        "--r",
        label="Threshold",
    )
    ax[1].set_ylabel(r"$C_{CC}$ (-)")
    ax[1].set_xlabel("Iteration number (-)")
    ax[1].set_xlim(1, max_iterations)
    ax[1].set_ylim(0.0, 1)
    ax[1].set_yticks(np.linspace(0.0, 1.0, 6))
    ax[1].set_xticks(np.linspace(1, max_iterations, max_iterations))
    ax[0].legend(frameon=False)
    plt.tight_layout()

    # Save
    if save is not None:
        plt.savefig(save, dpi=600)

    # Show or close.
    if show is True:
        plt.show(block=block)
    else:
        plt.close(fig)

    return fig, ax


def convergence_mesh(data, quantity, show, block, save):
    """

    Function to plot Mesh convergence.

    Parameters
    ----------
    data : dict
        Mesh data dict.
    quantity : str
        Quantity to plot. Options are "C_ZNCC", "iterations", or "norm".
    show : bool
        Control whether the plot is displayed.
    block : bool
        Control whether the plot blocks execution until closed.
    save : str
        Name to use to save plot. Uses default extension of `.png`.


    Returns
    -------
    fig :  matplotlib.pyplot.figure
        Figure object.
    ax : matplotlib.pyplot.axes
        Axes object.


    .. note::
        * The figure and axes objects can be returned allowing standard matplotlib
          functionality to be used to augment the plot generated. See the
          :ref:`plots tutorial <Plots Tutorial>` for guidance.

    .. seealso::
        :meth:`~geopyv.mesh.MeshBase.convergence`

    """
    # Get image names.
    try:
        platform = sys.platform
        if platform == "linux" or platform == "linux2" or platform == "darwin":
            split = "/"
        elif platform == "win32":
            split = r"\\"
        f_img = data["images"]["f_img"][
            ([(m.end(0)) for m in re.finditer(split, data["images"]["f_img"])][-1]) :
        ]
        g_img = data["images"]["g_img"][
            ([(m.end(0)) for m in re.finditer(split, data["images"]["g_img"])][-1]) :
        ]
        title = "Convergence: f_img = {f_img}; g_img = {g_img}".format(
            f_img=f_img, g_img=g_img
        )
    except Exception:
        log.warning("Could not extract image names for plot.")
        title = "Convergence:"

    # Extract variables from data.
    subsets = data["results"]["subsets"]
    max_norm = data["settings"]["max_norm"]
    tolerance = data["settings"]["tolerance"]
    max_iterations = data["settings"]["max_iterations"]
    iterations = []
    norm = []
    C_ZNCC = []
    for s in subsets:
        iterations.append(s["results"]["history"][0, -1])
        norm.append(s["results"]["history"][1, -1])
        C_ZNCC.append(s["results"]["history"][2, -1])
    iterations = np.asarray(iterations)
    norm = np.asarray(norm)
    C_ZNCC = np.asarray(C_ZNCC)

    # Create plot.
    fig, ax = plt.subplots(num=title)
    if quantity == "C_ZNCC":
        ax.hist(C_ZNCC, bins=50)
        ax.set_xlabel(r"$C_{ZNCC}$ (-)")
        ax.set_xlim([tolerance, 1.0])
    elif quantity == "iterations":
        ax.hist(iterations)
        ax.set_xlabel(r"Iterations (-)")
        ax.set_xlim([0, max_iterations])
    elif quantity == "norm":
        ax.hist(norm, bins=50)
        ax.set_xlabel(r"$\Delta Norm$ (-)")
        ax.set_xlim([0, max_norm])
    ax.set_ylabel("Count (-)")
    plt.tight_layout()

    # Save
    if save is not None:
        plt.savefig(save, dpi=600)

    # Show or close.
    if show is True:
        plt.show(block=block)
    else:
        plt.close(fig)

    return fig, ax


def contour_mesh(
    data,
    quantity,
    view,
    coords,
    imshow,
    colorbar,
    ticks,
    mesh,
    alpha,
    levels,
    axis,
    xlim,
    ylim,
    show,
    block,
    save,
):
    """Function to plot contours of mesh data."""

    # Load data.
    obj = data
    data = data.data

    nodes = data["nodes"]
    elements = data["elements"]

    # Plot setup.
    try:
        platform = sys.platform
        if platform == "linux" or platform == "linux2" or platform == "darwin":
            split = "/"
        elif platform == "win32":
            split = r"\\"
        f_img = data["images"]["f_img"][
            ([(m.end(0)) for m in re.finditer(split, data["images"]["f_img"])][-1]) :
        ]
        g_img = data["images"]["g_img"][
            ([(m.end(0)) for m in re.finditer(split, data["images"]["g_img"])][-1]) :
        ]
        title = (
            "Contour: f_img = {f_img}; g_img = {g_img}; variable = {variable}".format(
                f_img=f_img, g_img=g_img, variable=quantity
            )
        )
    except Exception:
        log.warning("Could not extract image names for plot.")
        title = "Contour: variable = {variable}".format(variable=quantity)
    fig, ax = plt.subplots(num=title)

    # Show image in background.
    if imshow is True:
        image = cv2.imread(data["images"]["f_img"], cv2.IMREAD_COLOR)
        image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_gs = cv2.GaussianBlur(image_gs, ksize=(5, 5), sigmaX=1.1, sigmaY=1.1)
        plt.imshow(image_gs, cmap="gray")
    else:
        ax.set_aspect("equal", "box")

    # Plot mesh.
    mesh_triangulation, x_p, y_p = gp.geometry.utilities.plot_triangulation(
        elements, nodes[:, 0], nodes[:, 1], data["mesh_order"]
    )
    if mesh is True:
        for i in range(np.shape(x_p)[0]):
            ax.plot(x_p[i], y_p[i], color="k", alpha=0.25, linewidth=0.5)

    if view == "subset":
        if quantity == "C_ZNCC":
            value = data["results"]["C_ZNCC"]
        subsets = data["results"]["subsets"]
        # Data extraction.
        value = []
        for s in subsets:
            if quantity == "u":
                value.append(float((s["results"]["p"])[0]))
            elif quantity == "v":
                value.append(float((s["results"]["p"])[1]))
            elif quantity == "u_x":
                value.append(float((s["results"]["p"])[2]))
            elif quantity == "v_x":
                value.append(float((s["results"]["p"])[3]))
            elif quantity == "u_y":
                value.append(float((s["results"]["p"])[4]))
            elif quantity == "v_y":
                value.append(float((s["results"]["p"])[5]))
            elif quantity == "R":
                value.append(np.sqrt(s["results"]["u"] ** 2 + s["results"]["v"] ** 2))
            elif quantity == "size":
                value.append(s["template"]["size"])
            else:
                value.append(s["results"][quantity])
        value = np.asarray(value)
    elif view == "particle":
        field = gp.field.Field(series=obj, coordinates=nodes)
        field.solve()
        value = np.zeros(np.shape(nodes)[0])
        if quantity == "u":
            for i in range(len(field.data["particles"])):
                value[i] = field.data["particles"][i].data["results"]["warps"][1, 0]
        elif quantity == "v":
            for i in range(len(field.data["particles"])):
                value[i] = field.data["particles"][i].data["results"]["warps"][1, 1]
        elif quantity == "R":
            for i in range(len(field.data["particles"])):
                value[i] = np.sqrt(
                    np.sum(
                        field.data["particles"][i].data["results"]["warps"][1, :2] ** 2,
                    )
                )
        elif quantity == "ep_xx":
            for i in range(len(field.data["particles"])):
                value[i] = -field.data["particles"][i].data["results"]["warps"][1, 2]
        elif quantity == "ep_yy":
            for i in range(len(field.data["particles"])):
                value[i] = -field.data["particles"][i].data["results"]["warps"][1, 5]
        elif quantity == "ep_xy":
            for i in range(len(field.data["particles"])):
                value[i] = (
                    -(
                        field.data["particles"][i].data["results"]["warps"][1, 3]
                        + field.data["particles"][i].data["results"]["warps"][1, 4]
                    )
                    / 2
                )
        elif quantity == "ep_vol":
            for i in range(len(field.data["particles"])):
                value[i] = field.data["particles"][i].data["results"]["vol_strains"][-1]

    if len(value) == len(nodes):
        # Set levels and extend.
        extend = "neither"
        if not isinstance(levels, type(None)):
            if np.max(value) > np.max(levels) and np.min(value) < np.min(levels):
                extend = "both"
            elif np.max(value) > np.max(levels):
                extend = "max"
            elif np.min(value) < np.min(levels):
                extend = "min"
        # Plot contours.
        triangulation = tri.Triangulation(nodes[:, 0], nodes[:, 1], mesh_triangulation)
        contours = ax.tricontourf(
            triangulation,
            value,
            alpha=alpha,
            levels=levels,
            extend=extend,
        )
    else:
        if levels is not None:
            vmin = levels[0]
            vmax = levels[-1]
        else:
            vmin = None
            vmax = None
        contours = ax.tripcolor(
            nodes[:, 0],
            nodes[:, 1],
            elements[:, :3],
            vmin=vmin,
            vmax=vmax,
            facecolors=value,
            alpha=alpha,
            cmap="viridis",
        )

    if colorbar is True:
        if quantity == "iterations":
            label = "Iterations (-)"
        elif quantity == "C_ZNCC":
            label = r"$C_{ZNCC}$ (-)"
        elif quantity == "norm":
            label = r"$\Delta$ Norm (-)"
        elif quantity == "u":
            label = r"Horizontal Displacement, $u$ ($px$)"
        elif quantity == "v":
            label = r"Vertical Displacement, $v$ ($px$)"
        elif quantity == "u_x":
            label = r"Horizontal Deformation Gradient, $du/dx$ ($-$)"
        elif quantity == "v_x":
            label = r"Shear Deformation Gradient, $dv/dx$ ($-$)"
        elif quantity == "u_y":
            label = r"Shear Deformation Gradient, $du/dy$ ($-$)"
        elif quantity == "v_y":
            label = r"Vertical Deformation Gradient, $dv/dy$ ($-$)"
        elif quantity == "ep_xx":
            label = r"Horizontal Normal Strain, $\epsilon_{xx}$ ($-$)"
        elif quantity == "ep_yy":
            label = r"Vertical Normal Strain, $\epsilon_{yy}$ ($-$)"
        elif quantity == "ep_xy":
            label = r"Shear Strain, $\epsilon_{xy}$ ($-$)"
        elif quantity == "ep_vol":
            label = r"Volumetric Strain, $\epsilon_{vol}$ ($-$)"
        else:
            label = quantity
        fig.colorbar(contours, label=label, ticks=ticks)

    ax.grid(False)
    # Axis control.
    if axis is False:
        ax.set_axis_off()

    # Limit control.
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Save.
    if save is not None:
        plt.savefig(save, dpi=600)

    # Show or close.
    if show is True:
        plt.show(block=block)
    else:
        plt.close(fig)

    return fig, ax


def quiver_mesh(data, scale, imshow, mesh, axis, xlim, ylim, show, block, save):
    """Function to plot quiver plot of mesh data."""

    # Load data.
    elements = data["elements"]
    subsets = data["results"]["subsets"]

    # Extract variables from data.
    x = []
    y = []
    u = []
    v = []
    for s in subsets:
        x.append(s["position"]["x"])
        y.append(s["position"]["y"])
        u.append(s["results"]["u"])
        v.append(s["results"]["v"])
    x = np.asarray(x)
    y = np.asarray(y)
    u = np.asarray(u)
    v = np.asarray(v)
    L = np.sqrt(u**2 + v**2)
    U = u / L
    V = v / L
    m = np.max(L)
    S = scale / (1 + np.log(m / L))
    U1 = S * U * 1 / 2.54
    V1 = S * V * 1 / 2.54

    # Plot setup.
    try:
        platform = sys.platform
        if platform == "linux" or platform == "linux2" or platform == "darwin":
            split = "/"
        elif platform == "win32":
            split = r"\\"
        f_img = data["images"]["f_img"][
            ([(m.end(0)) for m in re.finditer(split, data["images"]["f_img"])][-1]) :
        ]
        g_img = data["images"]["g_img"][
            ([(m.end(0)) for m in re.finditer(split, data["images"]["g_img"])][-1]) :
        ]
        title = "Quiver: f_img = {f_img}; g_img = {g_img}".format(
            f_img=f_img, g_img=g_img
        )
    except Exception:
        log.warning("Could not extract image names for plot.")
        title = "Quiver:"
    fig, ax = plt.subplots(num=title)

    # Plot mesh.
    if mesh is True:
        # Triangulation.
        _, x_p, y_p = gp.geometry.utilities.plot_triangulation(
            elements, x, y, data["mesh_order"]
        )
        for i in range(np.shape(x_p)[0]):
            ax.plot(x_p[i], y_p[i], color="k", alpha=0.25, linewidth=0.5)

    # Show image in background.
    if imshow is True:
        image = cv2.imread(data["images"]["f_img"], cv2.IMREAD_COLOR)
        image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_gs = cv2.GaussianBlur(image_gs, ksize=(5, 5), sigmaX=1.1, sigmaY=1.1)
        plt.imshow(image_gs, cmap="gray")
    else:
        ax.set_aspect("equal", "box")

    # Plot contours.
    ax.quiver(x, y, U1, -V1, color="b", scale=1.0, units="inches")

    # Axis control.
    if axis == "off":
        ax.set_axis_off()

    # Limit control.
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Save.
    if save is not None:
        plt.savefig(save, dpi=600)

    # Show or close.
    if show is True:
        plt.show(block=block)
    else:
        plt.close(fig)

    return fig, ax


def inspect_mesh(data, show, block, save):
    """

    Function to inspect Mesh topology.

    Parameters
    ----------
    data : dict
        Mesh data dict.
    show : bool
        Control whether the plot is displayed.
    block : bool
        Control whether the plot blocks execution until closed.
    save : str
        Name to use to save plot. Uses default extension of `.png`.


    Returns
    -------
    fig :  matplotlib.pyplot.figure
        Figure object.
    ax : matplotlib.pyplot.axes
        Axes object.


    .. note::
        * The figure and axes objects can be returned allowing standard matplotlib
          functionality to be used to augment the plot generated. See the
          :ref:`plots tutorial <Plots Tutorial>` for guidance.

    .. seealso::
        :meth:`~geopyv.mesh.MeshBase.inspect`

    """
    # Load image.
    image_path = data["images"]["f_img"]
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gs = cv2.GaussianBlur(image_gs, ksize=(5, 5), sigmaX=1.1, sigmaY=1.1)

    # Load data.
    nodes = data["nodes"]
    elements = data["elements"]

    # Extract variables from data.
    x = []
    y = []
    value = []
    for n in nodes:
        x.append(n[0])
        y.append(n[1])
    x = np.asarray(x)
    y = np.asarray(y)
    value = np.asarray(value)

    # Plot setup.
    try:
        platform = sys.platform
        if platform == "linux" or platform == "linux2" or platform == "darwin":
            split = "/"
        elif platform == "win32":
            split = r"\\"
        f_img = data["images"]["f_img"][
            ([(m.end(0)) for m in re.finditer(split, data["images"]["f_img"])][-1]) :
        ]
        title = "Inspect mesh: f_img = {f_img}".format(f_img=f_img)
    except Exception:
        log.warning("Could not extract image names for plot.")
        title = "Inspect mesh:"

    # Triangulation.
    _, x_p, y_p = gp.geometry.utilities.plot_triangulation(
        elements, x, y, data["mesh_order"]
    )

    # Plot figure.
    fig, ax = plt.subplots(num=title)
    for i in range(np.shape(x_p)[0]):
        ax.plot(x_p[i], y_p[i], color="b", alpha=1.0, linewidth=1.0)
    ax.imshow(
        image,
        cmap="gist_gray",
        interpolation="nearest",
        aspect="equal",
    )
    details = r"{nodes} nodes; {elements} elements".format(
        nodes=np.shape(nodes)[0], elements=np.shape(elements)[0]
    )
    image_size = np.shape(image)
    ax.text(
        image_size[1] / 2,
        image_size[0] * 1.05,
        details,
        horizontalalignment="center",
    )
    ax.set_axis_off()
    plt.tight_layout()

    # Save
    if save is not None:
        plt.savefig(save, dpi=600)

    # Show or close.
    if show is True:
        plt.show(block=block)
    else:
        plt.close(fig)

    return fig, ax


def trace_particle(
    data,
    quantity,
    component,
    obj_type,
    imshow,
    colorbar,
    ticks,
    alpha,
    axis,
    xlim,
    ylim,
    show,
    block,
    save,
):
    title = "Path; variable = {variable}".format(variable=quantity)
    labels = [
        r"$u$ ($px$)",
        r"$v$ ($px$)",
        r"$du/dx$ ($-$)",
        r"$dv/dx$ ($-$)",
        r"$du/dy$ ($-$)",
        r"$dv/dy$ ($-$)",
        r"$d^2u/dx^2$ ($-$)",
        r"$d^2v/dx^2$ ($-$)",
        r"$d^2u/dxdy$ ($-$)",
        r"$d^2v/dxdy$ ($-$)",
        r"$d^2u/dy^2$ ($-$)",
        r"$d^2v/dy^2$ ($-$)",
    ]
    if data["calibrated"] is True:
        labels[0] = r"$u$ ($mm$)"
        labels[1] = r"$v$ ($mm$)"

    fig, ax = plt.subplots(num=title)

    if obj_type == "Particle":
        if data["calibrated"] is True:
            points = data["plotting_coordinates"].reshape(-1, 1, 2)
        else:
            points = data["results"]["coordinates"].reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        values = np.diff(data["results"][quantity][:, component], axis=0)
        norm = plt.Normalize(values.min(), values.max())
        lc = LineCollection(segments, cmap="viridis", norm=norm)
        lc.set_array(values)
        lines = ax.add_collection(lc)

    elif obj_type == "Field":
        values = np.empty(
            (
                len(data["particles"]),
                len(data["particles"][0].data["results"][quantity]) - 1,
            )
        )
        if data["number_images"] > 1:
            segments = np.empty(
                (
                    len(data["particles"]),
                    data["number_images"] - 1,
                    2,
                    2,
                )
            )
        else:
            segments = np.empty((len(data["particles"]), 2, 2))
        for i in range(len(data["particles"])):
            values[i] = np.diff(
                data["particles"][i].data["results"][quantity][:, component], axis=0
            )
            if data["calibrated"] is True:
                points = (
                    data["particles"][i].data["plotting_coordinates"].reshape(-1, 1, 2)
                )
            else:
                points = (
                    data["particles"][i]
                    .data["results"]["coordinates"]
                    .reshape(-1, 1, 2)
                )
            segments[i] = np.concatenate([points[:-1], points[1:]], axis=1)
        values = values.flatten()
        norm = plt.Normalize(values.min(), values.max())
        lc = LineCollection(segments.reshape(-1, 2, 2), cmap="viridis", norm=norm)
        lc.set_array(values.flatten())
        lines = ax.add_collection(lc)

    # Show image in background.
    if imshow is True:
        image = cv2.imread(data["image_0"], cv2.IMREAD_COLOR)
        image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_gs = cv2.GaussianBlur(image_gs, ksize=(5, 5), sigmaX=1.1, sigmaY=1.1)
        plt.imshow(image_gs, cmap="gray")
    else:
        ax.set_aspect("equal", "box")
    if data["calibrated"] is True:
        ax.axis("off")

    if colorbar is True:
        fig.colorbar(lines, label=labels[component])

        # Limit control.
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Save.
    if save is not None:
        plt.savefig(save, dpi=600)

    # Show or close.
    if show is True:
        plt.show(block=block)
    else:
        plt.close(fig)

    return fig, ax


def history_particle(data, quantity, components, xlim, ylim, show, block, save):
    title = "History; {quantity}".format(quantity=quantity)
    if quantity == "warps":
        labels = [
            r"$u$ ($px$)",
            r"$v$ ($px$)",
            r"$du/dx$ ($-$)",
            r"$dv/dx$ ($-$)",
            r"$du/dy$ ($-$)",
            r"$dv/dy$ ($-$)",
            r"$d^2u/dx^2$ ($-$)",
            r"$d^2v/dx^2$ ($-$)",
            r"$d^2u/dxdy$ ($-$)",
            r"$d^2v/dxdy$ ($-$)",
            r"$d^2u/dy^2$ ($-$)",
            r"$d^2v/dy^2$ ($-$)",
        ]
    elif quantity == "stresses":
        labels = [
            r"$\sigma_{xx}$ ($kPa$)",
            r"$\sigma_{yy}$ ($kPa$)",
            r"$\sigma_{zz}$ ($kPa$)",
            r"$\sigma_{yz}$ ($kPa$)",
            r"$\sigma_{zx}$ ($kPa$)",
            r"$\sigma_{xy}$ ($kPa$)",
        ]
    elif quantity == "strains":
        labels = [
            r"$\epsilon_{xx}$ ($-$)",
            r"$\epsilon_{yy}$ ($-$)",
            r"$\epsilon_{xy}$ ($-$)",
            r"$\epsilon_{vol}$ ($-$)",
        ]

    fig, ax = plt.subplots(num=title)
    if quantity == "strains":
        ax.plot(
            range(np.shape(data["results"]["strains"])[0]),
            data["results"]["strains"][:, 0],
            label=labels[0],
        )
        ax.plot(
            range(np.shape(data["results"]["strains"])[0]),
            data["results"]["strains"][:, 1],
            label=labels[1],
        )
        ax.plot(
            range(np.shape(data["results"]["strains"])[0]),
            data["results"]["strains"][:, 5],
            label=labels[2],
        )
        ax.plot(
            range(np.shape(data["results"]["vol_strains"])[0]),
            data["results"]["vol_strains"],
            label=labels[3],
        )
        ax.set_ylabel(r"Strain, $\epsilon$ ")
        plt.legend()
    elif quantity != "works":
        if components is None:
            components = range(np.shape(data["results"][quantity])[1])
        for component in components:
            ax.plot(
                range(np.shape(data["results"][quantity])[0]),
                data["results"][quantity][:, component],
                label=labels[component],
            )
        # Legend.
        plt.legend()  # bbox_to_anchor=(1.05, 0.5), loc="center left", borderaxespad=0)
        ax.set_ylabel(r"Value")
    else:
        ax.plot(
            range(np.shape(data["results"][quantity])[0]), data["results"][quantity]
        )
        ax.set_ylabel(r"Work, $W$, ($kJ$)")

    # General formatting.
    # Logscale.
    ax.set_xscale("linear")
    ax.set_yscale("linear")

    # Limit control.
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Axis labels.
    ax.set_xlabel(r"Image Number, $i$ ($-$)")

    # Save.
    if save is not None:
        plt.savefig(save, bbox_inches="tight", dpi=600)

    # Show or close.
    if show is True:
        plt.show(block=block)
    else:
        plt.close(fig)

    return fig, ax


def inspect_field(data, mesh, show, block, save):
    # Plot setup.
    title = "Inspect particle field"
    fig, ax = plt.subplots(num=title)

    # Load image.
    image_path = data["image_0"]
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gs = cv2.GaussianBlur(image_gs, ksize=(5, 5), sigmaX=1.1, sigmaY=1.1)

    # Plot image.
    ax.imshow(
        image,
        cmap="gist_gray",
        interpolation="nearest",
        aspect="equal",
    )

    if mesh:
        # Load data.
        try:
            nodes = data["field"]["nodes"]
            elements = data["field"]["elements"]

            # Extract variables from data.
            x = []
            y = []
            value = []
            for n in nodes:
                x.append(n[0])
                y.append(n[1])
            x = np.asarray(x)
            y = np.asarray(y)
            value = np.asarray(value)

            # Triangulation
            _, x_p, y_p = gp.geometry.utilities.plot_triangulation(elements, x, y, 1)

            # Plot mesh.
            for i in range(np.shape(x_p)[0]):
                ax.plot(
                    x_p[i],
                    y_p[i],
                    color="b",
                    alpha=1.0,
                    linewidth=1.0,
                )
        except Exception:
            log.warning(
                "Mesh requested but no field mesh information "
                "exists as field was user-specified."
            )

    for i in range(len(data["field"]["coordinates"])):
        ax.scatter(
            data["field"]["coordinates"][i, 0],
            data["field"]["coordinates"][i, 1],
            c="r",
            marker="o",
        )
    details = r"{elements} particles".format(elements=np.shape(elements)[0])
    image_size = np.shape(image)
    ax.text(
        image_size[1] / 2,
        image_size[0] * 1.05,
        details,
        horizontalalignment="center",
    )
    ax.set_axis_off()
    plt.tight_layout()

    # Save
    if save is not None:
        plt.savefig(save, dpi=600)

    # Show or close.
    if show is True:
        plt.show(block=block)
    else:
        plt.close(fig)

    return fig, ax


def contour_field(
    data,
    mesh_index,
    quantity,
    component,
    imshow,
    colorbar,
    ticks,
    alpha,
    levels,
    axis,
    xlim,
    ylim,
    show,
    block,
    save,
):
    # Plot setup.
    title = "Contour"
    fig, ax = plt.subplots(num=title)

    points = np.zeros((len(data["particles"]), 3))
    for i in range(len(data["particles"])):
        points[i, :2] = data["particles"][i]["plotting_coordinates"][mesh_index]
        points[i, 2] = data["particles"][i]["results"][quantity][mesh_index, component]

    # Show image in background.
    if imshow is True:
        image = cv2.imread(data["image_0"], cv2.IMREAD_COLOR)
        image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_gs = cv2.GaussianBlur(image_gs, ksize=(5, 5), sigmaX=1.1, sigmaY=1.1)
        plt.imshow(image_gs, cmap="gray")
    else:
        ax.set_aspect("equal", "box")

    # Set levels and extend.
    extend = "neither"
    value = points[:, 2]
    if not isinstance(levels, type(None)):
        if np.max(value) > np.max(levels) and np.min(value) < np.min(levels):
            extend = "both"
        elif np.max(value) > np.max(levels):
            extend = "max"
        elif np.min(value) < np.min(levels):
            extend = "min"
    extend = "both"

    contours = ax.tricontourf(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        alpha=alpha,
        levels=levels,
        extend=extend,
    )

    if colorbar is True:
        if quantity == "iterations":
            label = "Iterations (-)"
        elif quantity == "C_ZNCC":
            label = r"$C_{ZNCC}$ (-)"
        elif quantity == "norm":
            label = r"$\Delta$ Norm (-)"
        else:
            label = quantity
        fig.colorbar(contours, label=label, ticks=ticks)

    # Axis control.
    if axis is False:
        ax.set_axis_off()

    # Limit control.
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    # Save
    if save is not None:
        plt.savefig(save, dpi=600)

    # Show or close.
    if show is True:
        plt.show(block=block)
    else:
        plt.close(fig)

    return fig, ax


def accumulation_field(
    data,
    quantity,
    window,
    imshow,
    colorbar,
    ticks,
    alpha,
    levels,
    axis,
    xlim,
    ylim,
    show,
    block,
    save,
):
    if window is None:
        window = (0, data["number_images"])
    value = np.zeros((np.shape(data["particles"])[0]))
    plotting_coordinates = np.zeros((np.shape(data["particles"])[0], 2))
    for i in range(np.shape(data["particles"])[0]):
        if quantity == "u":
            value[i] = np.sum(
                # abs(
                data["particles"][i].data["results"]["warps"][window[0] : window[1], 0]
                # )
            )
        elif quantity == "v":
            value[i] = np.sum(
                # abs(
                data["particles"][i].data["results"]["warps"][window[0] : window[1], 1]
                # )
            )
        elif quantity == "u_x":
            value[i] = np.sum(
                # abs(
                data["particles"][i].data["results"]["warps"][window[0] : window[1], 2]
                # )
            )
        elif quantity == "e_xy":
            value[i] = np.sum(
                # abs
                (
                    data["particles"][i].data["results"]["warps"][
                        window[0] : window[1], 3
                    ]
                    + data["particles"][i].data["results"]["warps"][
                        window[0] : window[1], 4
                    ]
                )
                / 2
            )
        elif quantity == "v_y":
            value[i] = np.sum(
                # abs(
                data["particles"][i].data["results"]["warps"][window[0] : window[1], 5]
                # )
            )
        elif quantity == "R":
            value[i] == np.sum(
                np.sqrt(
                    data["particles"][i].data["results"]["warps"][
                        window[0] : window[1], 0
                    ]
                    ** 2
                    + data["particles"][i].data["results"]["warps"][
                        window[0] : window[1], 1
                    ]
                    ** 2
                )
            )
        plotting_coordinates[i] = data["particles"][i].data["plotting_coordinates"][0]

    # Plot setup.
    title = "Accumulation"
    fig, ax = plt.subplots(num=title)

    # Show image in background.
    if imshow is True:
        image = cv2.imread(data["image_0"], cv2.IMREAD_COLOR)
        image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_gs = cv2.GaussianBlur(image_gs, ksize=(5, 5), sigmaX=1.1, sigmaY=1.1)
        plt.imshow(image_gs, cmap="gray")
    else:
        ax.set_aspect("equal", "box")

    extend = "neither"
    if not isinstance(levels, type(None)):
        if np.max(value) > np.max(levels) and np.min(value) < np.min(levels):
            extend = "both"
        elif np.max(value) > np.max(levels):
            extend = "max"
        elif np.min(value) < np.min(levels):
            extend = "min"
    extend = "both"

    contours = ax.tricontourf(
        plotting_coordinates[:, 0],
        plotting_coordinates[:, 1],
        value,
        alpha=alpha,
        levels=levels,
        extend=extend,
    )

    if colorbar is True:
        qs = np.asarray(["u", "v", "u_x", "e_xy", "v_y", "R"])
        labels = [
            r"Accumulated horizontal displacement, $U$ ($mm$)",
            r"Accumulated vertical displacement, $V$ ($mm$)",
            r"Accumulated horizontal normal strain, $dU/dX$ ($-$)",
            r"Accumulated shear strain, $\epsilon_{xy}$ ($-$)",
            r"Accumulated vertical normal strain, $dV/dY$ ($-$)",
            r"Accumulated absolute displacement, $R$ ($mm$)",
        ]
        label = labels[np.argwhere(quantity == qs)[0][0]]
        fig.colorbar(contours, label=label, ticks=ticks)

    # Axis control.
    if axis is False:
        ax.set_axis_off()

    # Limit control.
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    # Save
    if save is not None:
        plt.savefig(save, dpi=600)

    # Show or close.
    if show is True:
        plt.show(block=block)
    else:
        plt.close(fig)

    return fig, ax


def standard_error_validation(
    data,
    component,
    observing,
    xlim,
    ylim,
    scale,
    prev_series,
    prev_series_label,
    plot,
    show,
    block,
    save,
    xlabel,
    ylabel,
):
    labels = [
        r"Horizontal displacement, $u$ ($px$)",
        r"Vertical displacement, $v$ ($px$)",
        r"Horizontal normal strain, $\epsilon_{xx}$ ($-$)",
        r"Shear strain component, $dv/dx$ ($-$)",
        r"Shear strain component, $du/dy$ ($-$)",
        r"Vertical normal strain, $\epsilon_{yy}$ ($-$)",
        r"Strain gradient component, $d^2u/dx^2$ ($-$)",
        r"Strain gradient component, $d^2v/dx^2$ ($-$)",
        r"Strain gradient component, $d^2u/dxdy$ ($-$)",
        r"Strain gradient component, $d^2v/dxdy$ ($-$)",
        r"Strain gradient component, $d^2u/dy^2$ ($-$)",
        r"Strain gradient component, $d^2v/dy^2$ ($-$)",
        r"Rotation, $\theta$ ($^o$)",
        r"Pure shear strain, $\epsilon_{xy}$ ($-$)",
    ]
    colours = ["r", "b", "g", "orange", "purple", "k"]
    markers = [
        "o",
        "^",
        "s",
        "v",
    ]
    title = r"Standard error: component = {component}".format(
        component=labels[component]
    )
    fig, ax = plt.subplots(num=title)
    for i in range(len(data["applied"])):
        series = np.zeros((np.shape(data["applied"][i])[0], 2))
        for j in range(np.shape(data["applied"][i])[0]):
            if component == 12:
                series[j, 0] = (
                    180 / np.pi * abs(np.arccos(data["speckle"].data["pm"][j + 1, 2]))
                )
            elif component == 13:
                series[j, 0] = abs(data["speckle"].data["pm"][j + 1, 3])
            else:
                series[j, 0] = abs(data["speckle"].data["pm"][j + 1, component])
            if observing is not None:
                series[j, 1] = np.std(
                    np.sqrt(
                        (
                            data["applied"][i][j, :, observing]
                            - data["observed"][i][j, :, observing]
                        )
                        ** 2
                    )
                )
            else:
                series[j, 1] = np.std(
                    np.sqrt(
                        np.sum(
                            (
                                data["applied"][i][j, :, :2]
                                - data["observed"][i][j, :, :2]
                            )
                            ** 2,
                            axis=-1,
                        )
                    )
                )
        if plot == "scatter":
            ax.scatter(
                series[:, 0],
                series[:, 1],
                facecolors="none",
                edgecolors=colours[i],
                marker=markers[i],
                label=data["labels"][i],
            )
        elif plot == "line":
            ax.plot(
                series[:, 0],
                series[:, 1],
                color=colours[i],
                label=data["labels"][i],
            )
    if prev_series is not None:
        ax.plot(
            prev_series[:, 0], prev_series[:, 1], color="k", label=prev_series_label
        )
    # General formatting.
    # Legend.
    plt.legend(bbox_to_anchor=(0.05, 0.95), loc="upper left", borderaxespad=0)

    # Logscale.
    ax.set_xscale(scale)
    ax.set_yscale("log")
    ax.grid(which="both")
    ax.set_axisbelow(True)

    # Limit control.
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Axis labels.
    if xlabel is None:
        ax.set_xlabel(r"{}".format(labels[component]))
    else:
        ax.set_xlabel(xlabel)
    if ylabel is None:
        if observing is not None:
            ax.set_ylabel(r"Error, $\Delta$" + labels[observing])
        else:
            ax.set_ylabel(r"Standard error, $\rho_{px}$ ($px$)")
    else:
        ax.set_ylabel(ylabel)

    # Save.
    if save is not None:
        plt.savefig(save, bbox_inches="tight", dpi=600)

    # Show or close.
    if show is True:
        plt.show(block=block)
    else:
        plt.close(fig)

    return fig, ax


def mean_error_validation(
    data,
    component,
    xlim,
    ylim,
    scale,
    prev_series,
    prev_series_label,
    plot,
    show,
    block,
    save,
):
    labels = [
        r"Horizontal displacement, $u$ ($px$)",
        r"Vertical displacement, $v$ ($px$)",
        r"Horizontal normal strain, $\epsilon_{xx}$ ($-$)",
        r"Shear strain component, $dv/dx$ ($-$)",
        r"Shear strain component, $du/dy$ ($-$)",
        r"Vertical normal strain, $\epsilon_{yy}$ ($-$)",
        r"Strain gradient component, $d^2u/dx^2$ ($-$)",
        r"Strain gradient component, $d^2v/dx^2$ ($-$)",
        r"Strain gradient component, $d^2u/dxdy$ ($-$)",
        r"Strain gradient component, $d^2v/dxdy$ ($-$)",
        r"Strain gradient component, $d^2u/dy^2$ ($-$)",
        r"Strain gradient component, $d^2v/dy^2$ ($-$)",
        r"Rotation, $\theta$ ($^o$)",
        r"Pure shear strain, $\epsilon_{xy}$ ($-$)",
    ]
    colours = ["r", "b", "g", "orange", "purple", "k"]
    markers = [
        "o",
        "^",
        "s",
        "v",
    ]
    title = r"Mean error: component = {component}".format(component=labels[component])
    fig, ax = plt.subplots(num=title)
    for i in range(len(data["applied"])):
        series = np.zeros((np.shape(data["applied"][i])[0], 2))
        for j in range(np.shape(data["applied"][i])[0]):
            if component == 12:
                series[j, 0] = (
                    180 / np.pi * abs(np.arccos(data["speckle"].data["pm"][j + 1, 2]))
                )
            elif component == 13:
                series[j, 0] = abs(data["speckle"].data["pm"][j + 1, 3])
            else:
                series[j, 0] = abs(data["speckle"].data["pm"][j + 1, component])
            series[j, 1] = np.mean(
                np.sqrt(
                    np.sum(
                        (data["applied"][i][j, :, :2] - data["observed"][i][j, :, :2])
                        ** 2,
                        axis=-1,
                    )
                )
            )
        if plot == "scatter":
            ax.scatter(
                series[:, 0],
                series[:, 1],
                facecolors="none",
                edgecolors=colours[i],
                marker=markers[i],
                label=data["labels"][i],
            )
        elif plot == "line":
            ax.plot(
                series[:, 0],
                series[:, 1],
                color=colours[i],
                label=data["labels"][i],
            )
    if prev_series is not None:
        ax.plot(
            prev_series[:, 0], prev_series[:, 1], color="k", label=prev_series_label
        )
    # General formatting.
    # Legend.
    plt.legend(bbox_to_anchor=(0.05, 0.95), loc="upper left", borderaxespad=0)

    # Logscale.
    ax.set_xscale(scale)
    ax.set_yscale("log")
    ax.grid(which="both")
    ax.set_axisbelow(True)

    # Limit control.
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Axis labels.
    ax.set_xlabel(r"{}".format(labels[component]))
    ax.set_ylabel(r"Mean absolute error, $\mu_{px}$ ($px$)")

    # Save.
    if save is not None:
        plt.savefig(save, bbox_inches="tight", dpi=600)

    # Show or close.
    if show is True:
        plt.show(block=block)
    else:
        plt.close(fig)

    return fig, ax


def spatial_error_validation(
    data,
    field_index,
    time_index,
    quantity,
    imshow,
    colorbar,
    ticks,
    alpha,
    levels,
    xlim,
    ylim,
    show,
    block,
    save,
):
    if quantity == "u":
        label = r"Horizontal displacement error, $\epsilon_u$ ($px$)"
        value = np.sqrt(
            np.sum(
                (
                    data["applied"][field_index][time_index, :, 0]
                    - data["observed"][field_index][time_index, :, 0]
                )
                ** 2,
                axis=1,
            )
        )
    elif quantity == "v":
        label = r"Vertical displacement error, $\epsilon_v$ ($px$)"
        value = np.sqrt(
            np.sum(
                (
                    data["applied"][field_index][time_index, :, 1]
                    - data["observed"][field_index][time_index, :, 1]
                )
                ** 2,
                axis=1,
            )
        )
    elif quantity == "R":
        label = r"Absolute displacement error, $\epsilon_R$ ($px$)"
        value = np.sqrt(
            np.sum(
                (
                    data["applied"][field_index][time_index, :, :2]
                    - data["observed"][field_index][time_index, :, :2]
                )
                ** 2,
                axis=1,
            )
        )

    par_coords = np.zeros(
        (np.shape(data["fields"][field_index].data["particles"])[0], 2)
    )
    for i in range(np.shape(data["fields"][field_index].data["particles"])[0]):
        par_coords[i] = (
            data["fields"][field_index]
            .data["particles"][i]
            .data["results"]["coordinates"][time_index]
        )
    delaunay = spsp.Delaunay(par_coords)
    mesh_triangulation, x_p, y_p = gp.geometry.utilities.plot_triangulation(
        delaunay.simplices, par_coords[:, 0], par_coords[:, 1], 1
    )

    title = r"Error: component = {component}".format(component=label)
    fig, ax = plt.subplots(num=title)
    extend = "neither"
    if not isinstance(levels, type(None)):
        if np.max(value) > np.max(levels) and np.min(value) < np.min(levels):
            extend = "both"
        elif np.max(value) > np.max(levels):
            extend = "max"
        elif np.min(value) < np.min(levels):
            extend = "min"
    triangulation = tri.Triangulation(
        par_coords[:, 0], par_coords[:, 1], mesh_triangulation
    )
    contours = ax.tricontourf(
        triangulation,
        value,
        alpha=alpha,
        levels=levels,
        extend=extend,
    )
    if colorbar is True:
        fig.colorbar(contours, label=label, ticks=ticks)

    # Show image in background.
    if imshow is True:
        image = cv2.imread(
            data["speckle"].data["image_dir"]
            + data["speckle"].data["name"]
            + "_"
            + str(time_index)
            + data["speckle"].data["file_format"],
            cv2.IMREAD_COLOR,
        )
        image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_gs = cv2.GaussianBlur(image_gs, ksize=(5, 5), sigmaX=1.1, sigmaY=1.1)
        plt.imshow(image_gs, cmap="gray")
    else:
        ax.set_aspect("equal", "box")
    ax.grid(False)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    # Save.
    if save is not None:
        plt.savefig(save, dpi=600)
    # Show or close.
    if show is True:
        plt.show(block=block)
    else:
        plt.close(fig)
    return fig, ax


def inspect_calibration(data, image_index):
    fig, ax = plt.subplots()
    frame = cv2.imread(data["file_settings"]["calibration_images"][image_index])
    ax.imshow(frame)
    ax.scatter(
        data["calibration"]["corners"][image_index][:, :, 0],
        data["calibration"]["corners"][image_index][:, :, 1],
        color="r",
    )
    plt.show()

    return fig, ax


def visualise_calibration(data, block, show, save):
    fig, ax = plt.subplots()
    for corner in data["calibration"]["corners"]:
        ax.scatter(
            corner[:, :, 0],
            corner[:, :, 1],
            color="r",
        )
    ax.set_xlim(0, data["file_settings"]["image_size"][0])
    ax.set_ylim(0, data["file_settings"]["image_size"][1])
    ax.axis("equal")

    # Save
    if save is not None:
        plt.savefig(save, dpi=600)

    # Show or close.
    if show is True:
        plt.show(block=block)
    else:
        plt.close(fig)

    return fig, ax


def contour_calibration(
    data,
    quantity,
    points,
    colorbar,
    ticks,
    alpha,
    levels,
    axis,
    xlim,
    ylim,
    block,
    show,
    save,
):
    # Undistort the image points

    image_points = np.concatenate(data["calibration"]["corners"], axis=0)
    object_points = cv2.undistortImagePoints(
        image_points,
        data["calibration"]["camera_matrix"],
        data["calibration"]["distortion"],
    ).reshape(-1, 2)
    image_points = image_points.reshape(-1, 2)
    # Calculate the reprojection error for each point
    if quantity == "R":
        values = np.sqrt(np.sum((image_points - object_points) ** 2, axis=1))
    elif quantity == "u":
        values = (image_points - object_points)[:, 0]
    elif quantity == "v":
        values = (image_points - object_points)[:, 1]

    fig, ax = plt.subplots(num="Reprojection Errors")

    # Set levels and extend.
    extend = "neither"
    if not isinstance(levels, type(None)):
        if np.max(values) > np.max(levels) and np.min(values) < np.min(levels):
            extend = "both"
        elif np.max(values) > np.max(levels):
            extend = "max"
        elif np.min(values) < np.min(levels):
            extend = "min"
    extend = "both"

    contours = ax.tricontourf(
        image_points[:, 0],
        image_points[:, 1],
        values,
        alpha=alpha,
        levels=levels,
        extend=extend,
    )
    if points:
        ax.scatter(image_points[:, 0], image_points[:, 1], color="k")

    # Axis control.
    if axis is False:
        ax.set_axis_off()
    ax.axis("equal")
    plt.tight_layout()
    # Limit control.
    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(0, data["file_settings"]["image_size"][1])
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(0, data["file_settings"]["image_size"][0])

    if colorbar is True:
        if quantity == "R":
            label = r"Resultant, $R$ ($px$)"
        if quantity == "u":
            label = r"Horizontal displacement, $u$ ($px$)"
        if quantity == "v":
            label = r"Vertical displacement, $v$ ($px$)"
        fig.colorbar(contours, label=label, ticks=ticks)

    # Save
    if save is not None:
        plt.savefig(save, dpi=600)

    # Show or close.
    if show is True:
        plt.show(block=block)
    else:
        plt.close(fig)

    return fig, ax


def error_calibration(
    data,
    quantity,
    points,
    colorbar,
    ticks,
    alpha,
    levels,
    axis,
    xlim,
    ylim,
    block,
    show,
    save,
):
    reimgpnts = np.concatenate(data["projection"]["reimgpnts"])
    imgpnts = np.concatenate(data["projection"]["imgpnts"])
    if quantity == "R":
        error = np.sqrt(np.sum((reimgpnts - imgpnts) ** 2, axis=1))
    elif quantity == "u":
        error = (reimgpnts - imgpnts)[:, 0]
    elif quantity == "v":
        error = (reimgpnts - imgpnts)[:, 1]

    fig, ax = plt.subplots(num="Reprojection Errors: image space")

    # Set levels and extend.
    extend = "neither"
    if not isinstance(levels, type(None)):
        if np.max(error) > np.max(levels) and np.min(error) < np.min(levels):
            extend = "both"
        elif np.max(error) > np.max(levels):
            extend = "max"
        elif np.min(error) < np.min(levels):
            extend = "min"
    extend = "both"

    contours = ax.tricontourf(
        imgpnts[:, 0], imgpnts[:, 1], error, alpha=alpha, levels=levels, extend=extend
    )
    if points:
        for p in range(len(data["projection"]["imgpnts"])):
            ax.scatter(
                data["projection"]["imgpnts"][p][:, 0],
                data["projection"]["imgpnts"][p][:, 1],
                label=data["file_settings"]["calibration_images"][p],
            )
            ax.legend()

    # Axis control.
    if axis is False:
        ax.set_axis_off()
    ax.axis("equal")
    plt.tight_layout()
    # Limit control.
    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(0, data["file_settings"]["image_size"][1])
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(0, data["file_settings"]["image_size"][0])

    if colorbar is True:
        if quantity == "R":
            label = r"Resultant, $R$ ($px$)"
        if quantity == "u":
            label = r"Horizontal displacement, $u$ ($px$)"
        if quantity == "v":
            label = r"Vertical displacement, $v$ ($px$)"
        fig.colorbar(contours, label=label, ticks=ticks)

    # Save
    if save is not None:
        plt.savefig(save, dpi=600)

    # Show or close.
    if show is True:
        plt.show(block=block)
    else:
        plt.close(fig)

    return fig, ax


def kde_chain(data, R_t, true, axis, xlim, ylim, save, block, show):
    if xlim is None:
        xlim = (data["prior"][1, 0], data["prior"][1, 1])
    if ylim is None:
        ylim = (data["prior"][0, 0], data["prior"][0, 1])
    chain = np.transpose(
        np.asarray(
            [
                data["k_c"][R_t:],
                data["s_c"][R_t:],
            ]
        )
    )
    df = pd.DataFrame(chain, columns=["k_c", "s_c"])
    g = sns.JointGrid(x="k_c", y="s_c", xlim=xlim, ylim=ylim, data=df, space=0)
    g.plot_joint(sns.kdeplot, cmap="Blues", n_levels=15, fill=False, thresh=False)
    g.plot_marginals(sns.kdeplot, fill=True, legend=False)
    g.set_axis_labels(r"Rate parameter, $k$ (-)", r"Sensitivity, $s_{ep-ini}$ (-)")
    if true is not None:
        sns.scatterplot(
            x=true[:, 0],
            y=true[:, 1],
            marker="+",
            color="r",
            s=100,
            linewidth=1.5,
            ax=g.ax_joint,
        )
        # g.scatter(true[0], true[1], marker = "+", facecolors="none", edgecolors="r",)
    if save is not None:
        plt.savefig(save, dpi=600)

    # Show or close.
    if show is True:
        plt.show(block=block)

    return g


def convergence_bayes(data, true, axis, klim, slim, save, block, show):
    if klim is None:
        klim = (
            data["chains"][0].data["prior"][1, 0],
            data["chains"][0].data["prior"][1, 1],
        )
    if slim is None:
        slim = (
            data["chains"][0].data["prior"][0, 0],
            data["chains"][0].data["prior"][0, 1],
        )
    fig, ((ax, bx), (cx, dx)) = plt.subplots(2, 2)
    for i in range(data["chain_no"]):
        ax.plot(range(1, data["sample_no"]), data["results"]["chain_means"][i, 1:, 0])
        cx.plot(range(1, data["sample_no"]), data["results"]["chain_means"][i, 1:, 1])
    bx.plot(range(1, data["sample_no"]), data["results"]["R"][:, 0], color="k")
    dx.plot(range(1, data["sample_no"]), data["results"]["R"][:, 1], color="k")
    ax.plot(
        [data["results"]["R_t"], data["results"]["R_t"]],
        [klim[0], klim[1]],
        color="b",
        ls="--",
        label="Convergence Threshold",
    )
    cx.plot(
        [data["results"]["R_t"], data["results"]["R_t"]],
        [slim[0], slim[1]],
        color="b",
        ls="--",
    )
    bx.plot(
        [data["results"]["R_t"], data["results"]["R_t"]],
        [np.min(data["results"]["R"][:, 0]), np.max(data["results"]["R"][:, 0])],
        color="b",
        ls="--",
    )
    dx.plot(
        [data["results"]["R_t"], data["results"]["R_t"]],
        [np.min(data["results"]["R"][:, 1]), np.max(data["results"]["R"][:, 1])],
        color="b",
        ls="--",
    )
    if true is not None:
        ax.plot(
            (1, data["sample_no"]),
            [true[0], true[0]],
            color="r",
            ls="--",
            label="True Value",
        )
        cx.plot((1, data["sample_no"]), [true[1], true[1]], color="r", ls="--")
    ax.set_ylabel(r"Mean Sensitivity Degradation Rate, $\mu_k$ ($-$)")
    cx.set_ylabel(r"Mean Sensitivity Degradation Magnitude, $\mu_{s,ep}$ ($-$)")
    bx.set_ylabel(r"Sensitivity Degradation Rate Convergence, $\hat{R}_k$ ($-$)")
    dx.set_ylabel(
        r"Sensitivity Degradation Magnitude Convergence, $\hat{R}_{s,ep}$ ($-$)"
    )
    cx.set_xlabel(r"Sample number, $S$ ($-$)")
    dx.set_xlabel(r"Sample number, $S$ ($-$)")
    ax.set_xscale("log")
    bx.set_xscale("log")
    cx.set_xscale("log")
    dx.set_xscale("log")
    bx.set_yscale("log")
    dx.set_yscale("log")
    ax.set_ylim(klim)
    cx.set_ylim(slim)
    ax.set_xlim(1, data["sample_no"])
    bx.set_xlim(1, data["sample_no"])
    cx.set_xlim(1, data["sample_no"])
    dx.set_xlim(1, data["sample_no"])
    fig.legend()
    if axis is False:
        ax.set_axis_off()
    if save is not None:
        plt.savefig(save, dpi=600)

    # Show or close.
    if show is True:
        plt.show(block=block)

    return fig, ax


def autocorrelation_bayes(data, axis, xlim, ylim, save, block, show):
    autocorlim = len(data["results"]["autocorrelation"])
    fig, (ax, bx) = plt.subplots(2, 1)
    for i in range(3):
        ax.plot(range(autocorlim), data["results"]["autocorrelation"][:, i, 0])
        bx.plot(range(autocorlim), data["results"]["autocorrelation"][:, i, 1])

    ax.set_ylabel(r"Sensitivity Degradation Rate Autocorrelation, $\A_k$ ($-$)")
    bx.set_ylabel(
        r"Sensitivity Degradation Magnitude Autocorrelation, $\A_{s,ep}$ ($-$)"
    )
    ax.set_xlabel(r"Lag, $k$ ($-$)")
    bx.set_xlabel(r"Lag, $k$ ($-$)")
    ax.set_xlim(0, autocorlim)
    bx.set_xlim(0, autocorlim)
    ax.set_ylim(min(0, 1.1 * np.min(data["results"]["autocorrelation"][:, :, 0])))
    bx.set_ylim(min(0, 1.1 * np.min(data["results"]["autocorrelation"][:, :, 1])))

    if axis is False:
        ax.set_axis_off()
    if save is not None:
        plt.savefig(save, dpi=600)

    # Show or close.
    if show is True:
        plt.show(block=block)

    return fig, ax
