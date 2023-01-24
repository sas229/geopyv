import numpy as np
import geopyv
import matplotlib.pyplot as plt
import os


# Images.
ref = geopyv.image.Image("./images/T-Bar/IMG_1062.jpg")
tar = geopyv.image.Image("./images/T-Bar/IMG_1064.jpg")

# Coordinate.
f_coord = np.asarray([1000,1000])

# Loop.
img_list = []
square_metrics = []
for i in np.arange(10, 110, 10):
    template = geopyv.templates.Square(int(i))
    subset = geopyv.subset.Subset(f_img=ref, g_img=tar, f_coord=f_coord, template=template)
    name = "Square_{}_px".format(i)
    save = "./docs/scripts/"+name+".png"
    subset.inspect(save=save, show=False, block=False)
    img_list.append(save)
    square_metrics.append([subset.data["template"]["size"], subset.data["quality"]["sigma_intensity"], subset.data["quality"]["SSSIG"]])

# Make gif of subset.
os.system('convert -loop 0 -delay 100 %s ./docs/source/images/square_subset_size.gif' % ' '.join(img_list))

# Loop.
img_list = []
circle_metrics = []
for i in np.arange(10, 110, 10):
    template = geopyv.templates.Circle(int(i))
    subset = geopyv.subset.Subset(f_img=ref, g_img=tar, f_coord=f_coord, template=template)
    name = "Circle_{}_px".format(i)
    save = "./docs/scripts/"+name+".png"
    subset.inspect(save=save, show=False, block=False)
    img_list.append(save)
    circle_metrics.append([subset.data["template"]["size"], subset.data["quality"]["sigma_intensity"], subset.data["quality"]["SSSIG"]])

# Make gif.
os.system('convert -loop 0 -delay 100 %s ./docs/source/images/circle_subset_size.gif' % ' '.join(img_list))

# Clean up images.
dir = "./docs/scripts/"
for fname in os.listdir(dir):
    if fname.endswith(".png"):
        os.remove(dir+fname)

# Make plot of metrics.
img_list = []
square_metrics = np.asarray(square_metrics)
circle_metrics = np.asarray(circle_metrics)
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.set_xlim([0, 110])
ax2.set_xlim([0, 110])
ax1.set_ylim([0, 20])
ax2.set_ylim([1E3, 1E6])
ax1.plot([0, 110], [15, 15], color='k', linestyle="-.", label="Target")
ax2.plot([0, 110], [1E5, 1E5], color='k', linestyle="-.")
ax1.set_ylabel(r"$\sigma_{s}$ (px)")
ax2.set_ylabel("SSSIG (-)")
ax2.set_xlabel("Subset size (px)")
ax1.plot(circle_metrics[:,0], circle_metrics[:,1], color ="b", label="Circle")
ax2.plot(circle_metrics[:,0], circle_metrics[:,2], color ="b")
ax1.scatter(circle_metrics[:,0], circle_metrics[:,1], color ="b", marker="o")
ax2.semilogy(circle_metrics[:,0], circle_metrics[:,2], color ="b", marker="o")
ax1.plot(square_metrics[:,0], square_metrics[:,1], color ="r", label="Square")
ax2.plot(square_metrics[:,0], square_metrics[:,2], color ="r")
ax1.scatter(square_metrics[:,0], square_metrics[:,1], color ="r", marker="s")
ax2.semilogy(square_metrics[:,0], square_metrics[:,2], color ="r", marker="s")
ax1.legend(frameon=False, loc="lower right")
name = "./docs/source/images/subset_quality.png"
plt.savefig(name)
