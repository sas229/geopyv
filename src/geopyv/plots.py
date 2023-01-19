import matplotlib.pyplot as plt
import numpy as np
from geopyv.templates import Circle

def inspect_subset(img, x, y, template, sigma_intensity, SSSIG):
    """Function to show the subset and associated quality metrics."""
    f, ax = plt.subplots(num="Subset")
    x_min = (np.round(x, 0) - template.size).astype(int)
    x_max = (np.round(x, 0) + template.size).astype(int)
    y_min = (np.round(y, 0) - template.size).astype(int)
    y_max = (np.round(y, 0) + template.size).astype(int)
    subset = img.astype(np.float32)[y_min:y_max+1, x_min:x_max+1]

    # If a circular subset, mask pixels outside radius.
    if type(template) == Circle:
        x, y = np.meshgrid(
            np.arange(-template.size, template.size + 1, 1),
            np.arange(-template.size, template.size + 1, 1),
        )
        dist = np.sqrt(x ** 2 + y ** 2)
        mask = np.zeros(subset.shape)
        mask[dist > template.size] = 255
        subset = np.maximum(subset, mask)

    ax.imshow(subset, cmap="gist_gray")
    quality = r"Quality metrics: $\sigma_s$ = {:.2f}; SSSIG = {:.2E}".format(sigma_intensity, SSSIG)
    ax.text(template.size, 2*template.size + 5, quality, horizontalalignment="center")
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

def convergence_subset(history, max_iterations, max_norm, tolerance):
    """Function to plot subset convergence."""
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, num="Convergence")
    ax1.semilogy(history[0,:], history[1,:], marker="o", clip_on=False, label="Convergence")
    ax1.plot([1, max_iterations], [max_norm, max_norm], "--r", label="Threshold")
    ax2.plot(history[0,:], history[2,:], marker="o", clip_on=False, label="Convergence")
    ax2.plot([1, max_iterations], [tolerance, tolerance], "--r", label="Threshold")
    ax1.set_ylabel(r"$\Delta$ Norm (-)")
    ax1.set_ylim(max_norm/1000, max_norm*1000)
    ax1.set_yticks([max_norm*1000, max_norm*100, max_norm*10, max_norm, max_norm/10, max_norm/100, max_norm/1000])
    ax2.set_ylabel(r"$C_{CC}$ (-)")
    ax2.set_xlabel("Iteration number (-)")
    ax2.set_xlim(1, max_iterations)
    ax2.set_ylim(0.0, 1)
    ax2.set_yticks(np.linspace(0.0, 1.0, 6))
    ax2.set_xticks(np.linspace(1, max_iterations, max_iterations))
    ax1.legend(frameon=False)
    plt.tight_layout()
    plt.show()