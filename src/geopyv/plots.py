import sys
import cv2 
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import re

def inspect_subset(data, show, block, save):
    """Function to show the subset and associated quality metrics."""

    # Load data.
    image = cv2.imread(data["images"]["f_img"], cv2.IMREAD_COLOR)
    image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gs = cv2.GaussianBlur(image_gs, ksize=(5,5), sigmaX=1.1, sigmaY=1.1)
    x = data["position"]["x"]
    y = data["position"]["y"]
    template_size = data["template"]["size"]
    sigma_intensity = data["quality"]["sigma_intensity"]
    SSSIG = data["quality"]["SSSIG"]
    title = "Subset: {x},{y} (px)".format(x=x, y=y)
    mask = data["template"]["mask"].astype(np.float32)

    # Figure setup.
    fig, ax = plt.subplots(num=title)
    x_min = (np.round(x, 0) - template_size).astype(int)
    x_max = (np.round(x, 0) + template_size).astype(int)
    y_min = (np.round(y, 0) - template_size).astype(int)
    y_max = (np.round(y, 0) + template_size).astype(int)
    image = image_gs.astype(np.float32)[y_min:y_max+1, x_min:x_max+1]

    # Apply subset mask.
    image = np.maximum(image, mask)

    # Create plot.
    ax.imshow(image, cmap="gist_gray", interpolation='nearest', aspect='equal', extent=(-0.5,6.5,-0.5,5.5))
    quality = r"Subset size: {} (px); Quality metrics: $\sigma_s$ = {:.2f} (px); SSSIG = {:.2E} (-)".format(template_size, sigma_intensity, SSSIG)
    ax.text(3.0, -1.0, quality, horizontalalignment="center")
    ax.set_axis_off()
    plt.tight_layout()

    # Save
    if save != None:
        plt.savefig(save)

    # Show or close.
    if show == True:
        plt.show(block=block)
    else:
        plt.close(fig)

    return fig, ax

def convergence_subset(data, show, block, save):
    """Function to plot subset convergence."""

    # Load data.
    history = data["results"]["history"]
    max_iterations = data["settings"]["max_iterations"]
    max_norm = data["settings"]["max_norm"]
    tolerance = data["settings"]["tolerance"]
    x = data["position"]["x"]
    y = data["position"]["y"]

    # Create plot.
    title = "Convergence: {x},{y} (px)".format(x=x, y=y)
    fig, ax = plt.subplots(2, 1, sharex=True, num=title)
    ax[0].semilogy(history[0,:], history[1,:], marker="o", clip_on=False, label="Convergence")
    ax[0].plot([1, max_iterations], [max_norm, max_norm], "--r", label="Threshold")
    ax[0].set_ylabel(r"$\Delta$ Norm (-)")
    ax[0].set_ylim(max_norm/1000, max_norm*1000)
    ax[0].set_yticks([max_norm*1000, max_norm*100, max_norm*10, max_norm, max_norm/10, max_norm/100, max_norm/1000])
    ax[1].plot(history[0,:], history[2,:], marker="o", clip_on=False, label="Convergence")
    ax[1].plot([1, max_iterations], [tolerance, tolerance], "--r", label="Threshold")
    ax[1].set_ylabel(r"$C_{CC}$ (-)")
    ax[1].set_xlabel("Iteration number (-)")
    ax[1].set_xlim(1, max_iterations)
    ax[1].set_ylim(0.0, 1)
    ax[1].set_yticks(np.linspace(0.0, 1.0, 6))
    ax[1].set_xticks(np.linspace(1, max_iterations, max_iterations))
    ax[0].legend(frameon=False)
    plt.tight_layout()

    # Save
    if save != None:
        plt.savefig(save)

    # Show or close.
    if show == True:
        plt.show(block=block)
    else:
        plt.close(fig)

    return fig, ax

def contour_mesh(data, quantity, imshow, colorbar, ticks, mesh, alpha, levels, axis, xlim, ylim, show, block, save):
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
        if quantity == "u_x":
            value.append(float((s["results"]["p"])[2]))
        elif quantity == "v_x":
            value.append(float((s["results"]["p"])[3]))
        elif quantity == "u_y":
            value.append(float((s["results"]["p"])[4]))
        elif quantity == "v_y":
            value.append(float((s["results"]["p"])[5]))
        elif quantity == "R":
            value.append(np.sqrt(s["results"]["u"]**2 + s["results"]["v"]**2))
        else:
            value.append(s["results"][quantity])
    x = np.asarray(x)
    y = np.asarray(y)
    value = np.asarray(value)

    # Plot setup.
    platform = sys.platform
    if platform == "linux" or platform == "linux2" or platform == "darwin":
        split = "/"
    elif platform == "win32":
        split = "\\"
    f_img = data["images"]["f_img"][([(m.end(0)) for m in re.finditer(split, data["images"]["f_img"])][-1]):]
    g_img = data["images"]["g_img"][([(m.end(0)) for m in re.finditer(split, data["images"]["f_img"])][-1]):]
    title = "Contour: f_img = {f_img}; g_img = {g_img}; variable = {variable}".format(f_img=f_img, g_img=g_img, variable=quantity)
    fig, ax = plt.subplots(num=title)
    
    # Triangulation.
    mesh_triangulation, x_p, y_p = _plot_triangulation(elements, x, y)
    
    # Plot mesh.
    if mesh == True:
        for i in range(np.shape(x_p)[0]):
            ax.plot(x_p[i], y_p[i], color="k", alpha=0.25, linewidth = "0.5")
    triangulation = tri.Triangulation(nodes[:,0], nodes[:,1], mesh_triangulation)

    # Set levels and extend.
    extend = "neither"
    if type(levels) != type(None):
        if np.max(value) > np.max(levels):
            extend = "max"
        elif np.min(value) < np.min(levels):
            extend = "min"
        elif np.max(value) > np.max(levels) and np.min(value) < np.min(levels):
            extend = "both"

    # Show image in background.
    if imshow == True:
        image = cv2.imread(data["images"]["f_img"], cv2.IMREAD_COLOR)
        image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_gs = cv2.GaussianBlur(image_gs, ksize=(5,5), sigmaX=1.1, sigmaY=1.1)
        plt.imshow(image_gs, cmap='gray')
    else:
        ax.set_aspect('equal', 'box')
            
    # Plot contours.
    contours = ax.tricontourf(triangulation, value, alpha=alpha, levels=levels, extend=extend)
    if colorbar == True:
        label = quantity
        fig.colorbar(contours, label=label, ticks=ticks)
    
    # Axis control.
    if axis == "off":
        ax.set_axis_off()
    
    # Limit control.
    if xlim != None:
        ax.set_xlim(xlim)
    if ylim != None:
        ax.set_ylim(ylim)

    # Save.
    if save != None:
        plt.savefig(save)

    # Show or close.
    if show == True:
        plt.show(block=block)
    else:
        plt.close(fig)

    return fig, ax

def _plot_triangulation(elements, x, y):
    """Method to compute a first order triangulation from a second order element."""
    plot_elements = []
    x_p = []
    y_p = []
    for element in elements:
        plot_elements.append([element[0], element[3], element[5]])
        plot_elements.append([element[1], element[3], element[4]])
        plot_elements.append([element[2], element[4], element[5]])
        plot_elements.append([element[3], element[4], element[5]])
        x_p.append([x[element[0]], x[element[3]], x[element[1]], x[element[4]], x[element[2]], x[element[5]], x[element[0]]])
        y_p.append([y[element[0]], y[element[3]], y[element[1]], y[element[4]], y[element[2]], y[element[5]], y[element[0]]])
    mesh_triangulation = np.asarray(plot_elements)
    x_p = np.asarray(x_p)
    y_p = np.asarray(y_p)
    return mesh_triangulation, x_p, y_p