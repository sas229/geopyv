"""

Plots module for geopyv.

"""
import logging
import sys
import cv2
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from matplotlib.collections import LineCollection
import scipy as sp
import numpy as np
import re
import geopyv as gp
import alphashape

plt.rcParams["axes.prop_cycle"] = plt.cycler(
    "color", plt.cm.tab20(np.linspace(0, 1, 20))
)
log = logging.getLogger(__name__)


def warp_visualiser(template, warp, show, block, save):
    title = "Visualise warp"

    f_coord = np.zeros(2)
    f_coords = template.coords
    g_coords = gp._subset_extensions._g_coords(f_coord, warp, f_coords)

    f_alpha = alphashape.alphashape(f_coords, 1)
    f_x, f_y = f_alpha.exterior.xy
    g_alpha = alphashape.alphashape(g_coords, 1)
    g_x, g_y = g_alpha.exterior.xy

    fig, ax = plt.subplots(num=title)

    ax.fill(f_x, f_y, alpha=0.5, color="b")
    ax.fill(g_x, g_y, alpha=0.5, color="r")

    ax.set_axis_off()
    ax.axis("equal")
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


def inspect_subset_warp(data, mask, show, block, save):
    image = cv2.imread(data["images"]["f_img"], cv2.IMREAD_COLOR)
    image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gs = cv2.GaussianBlur(image_gs, ksize=(5, 5), sigmaX=1.1, sigmaY=1.1)
    x = data["position"]["x"]
    y = data["position"]["y"]
    template_size = data["template"]["size"]
    title = "Inspect subset: f_coord = ({x},{y}) (px)".format(x=x, y=y)

    f_coord = np.asarray([x, y])
    if data["template"]["shape"] == "circle":
        template = gp.templates.Circle(template_size)
    elif data["template"]["shape"] == "square":
        template = gp.templates.Square(template_size)
    f_img = gp.image.Image(data["images"]["f_img"])
    f_coords = gp._subset_extensions._init_reference(
        f_coord, template.coords, f_img.QCQT
    )[0]
    g_coords = gp._subset_extensions._g_coords(f_coord, data["results"]["p"], f_coords)

    # Crop image and mask.
    x_min = (np.round(min(np.min(f_coords[:, 0]), np.min(g_coords[:, 0])), 0)).astype(
        int
    )
    x_max = (np.round(max(np.max(f_coords[:, 0]), np.max(g_coords[:, 0])), 0)).astype(
        int
    )
    y_min = (np.round(min(np.min(f_coords[:, 1]), np.min(g_coords[:, 1])), 0)).astype(
        int
    )
    y_max = (np.round(max(np.max(f_coords[:, 1]), np.max(g_coords[:, 1])), 0)).astype(
        int
    )

    # image = image_gs.astype(np.float32)[y_min : y_max + 1, x_min : x_max + 1]
    if not isinstance(mask, type(None)):
        if type(mask) == np.ndarray:
            if np.shape(mask) == np.shape(image_gs):
                mask = mask.astype(np.float32)[y_min : y_max + 1, x_min : x_max + 1]
                invert_mask = np.abs(mask - 1) * 255
                image = np.maximum(image, invert_mask)

    f_hull = sp.spatial.ConvexHull(f_coords)
    g_hull = sp.spatial.ConvexHull(g_coords)

    fig, ax = plt.subplots(num=title)
    ax.imshow(
        image,
        cmap="gist_gray",
        interpolation="nearest",
        aspect="equal",
    )
    ax.fill(
        f_coords[f_hull.vertices, 0], f_coords[f_hull.vertices, 1], alpha=0.5, color="b"
    )
    ax.fill(
        g_coords[g_hull.vertices, 0], g_coords[g_hull.vertices, 1], alpha=0.5, color="r"
    )

    ax.text(
        3.0,
        -1.0,
        "warp vector: " + str(data["results"]["p"]),
        horizontalalignment="center",
    )

    ax.set_xlim(x_min - 5, x_max + 5)
    ax.set_ylim(y_max + 5, y_min - 5)

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


def inspect_subset(data, mask, show, block, save):
    """

    Function to show the Subset and associated quality metrics.

    Parameters
    ----------
    data : dict
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
    nodes = data["nodes"]
    elements = data["elements"]
    subsets = data["results"]["subsets"]

    # Extract variables from data.
    x = []
    y = []
    value = []
    for s in subsets:
        x.append(s["position"]["x"])
        y.append(s["position"]["y"])

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
        else:
            value.append(s["results"][quantity])

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

    # Triangulation.
    mesh_triangulation, x_p, y_p = gp.geometry.utilities.plot_triangulation(
        elements, x, y, data["mesh_order"]
    )

    # Plot mesh.
    if mesh is True:
        for i in range(np.shape(x_p)[0]):
            ax.plot(x_p[i], y_p[i], color="k", alpha=0.25, linewidth=0.5)
    triangulation = tri.Triangulation(nodes[:, 0], nodes[:, 1], mesh_triangulation)

    # Set levels and extend.
    extend = "neither"
    if not isinstance(levels, type(None)):
        if np.max(value) > np.max(levels) and np.min(value) < np.min(levels):
            extend = "both"
        elif np.max(value) > np.max(levels):
            extend = "max"
        elif np.min(value) < np.min(levels):
            extend = "min"
    extend = "both"

    # Show image in background.
    if imshow is True:
        image = cv2.imread(data["images"]["f_img"], cv2.IMREAD_COLOR)
        image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_gs = cv2.GaussianBlur(image_gs, ksize=(5, 5), sigmaX=1.1, sigmaY=1.1)
        plt.imshow(image_gs, cmap="gray")
    else:
        ax.set_aspect("equal", "box")

    # Plot contours.
    contours = ax.tricontourf(
        triangulation,
        value,
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

    fig, ax = plt.subplots(num=title)

    if obj_type == "Particle":
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
                len(data["particles"][0]["results"][quantity]) - 1,
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
                data["particles"][i]["results"][quantity][:, component], axis=0
            )
            points = data["particles"][i]["results"]["coordinates"].reshape(-1, 1, 2)
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

    fig, ax = plt.subplots(num=title)
    if components is None:
        components = range(np.shape(data["results"][quantity])[1])
    for component in components:
        ax.plot(
            range(np.shape(data["results"][quantity])[0]),
            data["results"][quantity][:, component],
            label=labels[component],
        )

    # General formatting.
    # Legend.
    plt.legend(bbox_to_anchor=(1.05, 0.5), loc="center left", borderaxespad=0)

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
    ax.set_ylabel(r"Value")

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
):
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
        r"$\theta$ ($^o$)",
        r"$\epsilon_{\gamma}$ ($-$)",
    ]
    colours = [
        "r",
        "b",
        "g",
        "orange",
    ]
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
    for i in range(np.shape(data["applied"])[0]):
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

    # Limit control.
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Axis labels.
    ax.set_xlabel(r"Applied warp, {}".format(labels[component]))
    if observing is not None:
        ax.set_ylabel(r"Error, $\Delta$" + labels[observing])
    else:
        ax.set_ylabel(r"Standard error, $\rho_{px}$ ($px$)")

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
        r"$\theta$ ($^o$)",
        r"$\epsilon_{\gamma}$ ($-$)",
    ]
    colours = [
        "r",
        "b",
        "g",
        "orange",
    ]
    markers = [
        "o",
        "^",
        "s",
        "v",
    ]
    title = r"Mean error: component = {component}".format(component=labels[component])
    fig, ax = plt.subplots(num=title)
    for i in range(np.shape(data["applied"])[0]):
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

    # Limit control.
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Axis labels.
    ax.set_xlabel(r"Applied warp, {}".format(labels[component]))
    ax.set_ylabel(r"Mean error, $\mu_{px}$ ($px$)")

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
    component,
    imshow,
    colorbar,
    xlim,
    ylim,
    show,
    block,
    save,
):
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
        r"$\theta$ ($^o$)",
        r"$\epsilon_{\gamma}$ ($-$)",
    ]

    title = r"Error: component = {component}".format(component=labels[component])
    fig, ax = plt.subplots(num=title)
    values = np.empty(
        (
            len(data["fields"][field_index].data["particles"]),
            len(
                data["fields"][field_index].data["particles"][0]["results"][
                    "coordinates"
                ]
            )
            - 1,
        )
    )
    if data["fields"][field_index].data["number_images"] > 1:
        segments = np.empty(
            (
                len(data["fields"][field_index].data["particles"]),
                data["fields"][field_index].data["number_images"] - 1,
                2,
                2,
            )
        )
    else:
        segments = np.empty((len(data["fields"][field_index].data["particles"]), 2, 2))
    for i in range(len(data["fields"][field_index].data["particles"])):
        values[i] = np.sqrt(
            np.sum(
                (
                    data["applied"][field_index][:, i, :2]
                    - data["observed"][field_index][:, i, :2]
                )
                ** 2,
                axis=-1,
            )
        )
        points = (
            data["fields"][field_index]
            .data["particles"][i]["results"]["coordinates"]
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
        image = cv2.imread(
            data["fields"][field_index].data["image_0"], cv2.IMREAD_COLOR
        )
        image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_gs = cv2.GaussianBlur(image_gs, ksize=(5, 5), sigmaX=1.1, sigmaY=1.1)
        plt.imshow(image_gs, cmap="gray")
    else:
        ax.set_aspect("equal", "box")

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


def inspect_calibration(data, image_index):
    fig, ax = plt.subplots()
    frame = cv2.imread(data["file_settings"]["calibration_images"][image_index])
    print(data["calibration"]["corners"][image_index])
    print(data["calibration"]["ids"][image_index])

    # cv2.aruco.drawDetectedMarkers(frame, data["calibration"]["corners"][image_index],
    # data["calibration"]["ids"][image_index])
    ax.imshow(frame)
    ax.scatter(
        data["calibration"]["corners"][image_index][:, :, 0],
        data["calibration"]["corners"][image_index][:, :, 1],
        color="r",
    )
    plt.show()

    return fig, ax
