import matplotlib.pyplot as plt
import numpy as np
from geopyv.templates import Circle

def inspect_subset(subset):
    """Function to show the subset and associated quality metrics."""
    f, ax = plt.subplots(num="Subset")
    x_min = (np.round(subset.x, 0) - subset.template.size).astype(int)
    x_max = (np.round(subset.x, 0) + subset.template.size).astype(int)
    y_min = (np.round(subset.y, 0) - subset.template.size).astype(int)
    y_max = (np.round(subset.y, 0) + subset.template.size).astype(int)
    img = subset.f_img.image_gs.astype(np.float32)[y_min:y_max+1, x_min:x_max+1]

    # If a circular subset, mask pixels outside radius.
    if type(subset.template) == Circle:
        x, y = np.meshgrid(
            np.arange(-subset.template.size, subset.template.size + 1, 1),
            np.arange(-subset.template.size, subset.template.size + 1, 1),
        )
        dist = np.sqrt(x ** 2 + y ** 2)
        mask = np.zeros(img.shape)
        mask[dist > subset.template.size] = 255
        img = np.maximum(img, mask)

    ax.imshow(img, cmap="gist_gray")
    quality = r"Quality metrics: $\sigma_s$ = {:.2f}; SSSIG = {:.2E}".format(subset.sigma_intensity, subset.SSSIG)
    ax.text(subset.template.size, 2*subset.template.size + 5, quality, horizontalalignment="center")
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

def convergence_subset(subset):
    """Function to plot subset convergence."""
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, num="Convergence")
    ax1.semilogy(subset.history[0,:], subset.history[1,:], marker="o", clip_on=False, label="Convergence")
    ax1.plot([1, subset.max_iterations], [subset.max_norm, subset.max_norm], "--r", label="Threshold")
    ax2.plot(subset.history[0,:], subset.history[2,:], marker="o", clip_on=False, label="Convergence")
    ax2.plot([1, subset.max_iterations], [subset.tolerance, subset.tolerance], "--r", label="Threshold")
    ax1.set_ylabel(r"$\Delta$ Norm (-)")
    ax1.set_ylim(subset.max_norm/1000, subset.max_norm*1000)
    ax1.set_yticks([subset.max_norm*1000, subset.max_norm*100, subset.max_norm*10, subset.max_norm, subset.max_norm/10, subset.max_norm/100, subset.max_norm/1000])
    ax2.set_ylabel(r"$C_{CC}$ (-)")
    ax2.set_xlabel("Iteration number (-)")
    ax2.set_xlim(1, subset.max_iterations)
    ax2.set_ylim(0.0, 1)
    ax2.set_yticks(np.linspace(0.0, 1.0, 6))
    ax2.set_xticks(np.linspace(1, subset.max_iterations, subset.max_iterations))
    ax1.legend(frameon=False)
    plt.tight_layout()
    plt.show()