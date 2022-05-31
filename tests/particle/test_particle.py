import numpy as np
from geopyv.templates import Circle
from geopyv.image import Image
from geopyv.subset import Subset
from geopyv.mesh import Mesh
from geopyv.particle import Particle

class mini_mesh:
    def __init__(self, pts, tri, areas):
        self.pts = pts
        self.tri = tri
        self.areas = areas

def test_triloc():
    """Check ability to identify particle element."""

    pts = np.asarray([[0.0,0.0],[1.0,0.0],[0.0,1.0],[1.0,1.0],[0.5,0.5]])
    tri = np.asarray([[0,1,4],[1,3,4],[3,2,4],[2,0,4]])
    areas = np.asarray([0.25,0.25,0.25,0.25])
    mesh = mini_mesh(pts, tri, areas)

    par0 = Particle(np.asarray([0.5, 0.25]))
    par1 = Particle(np.asarray([0.75, 0.5]))
    par2 = Particle(np.asarray([0.5, 0.75]))
    par3 = Particle(np.asarray([0.25, 0.5]))
    par4 = Particle(np.asarray([0.1,0.1]))
    par0ele = par0.triloc(mesh)
    par1ele = par1.triloc(mesh)
    par2ele = par2.triloc(mesh)
    par3ele = par3.triloc(mesh)
    par4ele = par4.triloc(mesh)

    assert par0ele == 0 and par1ele == 1 and par2ele == 2 and par3ele == 3 and (par4ele == 0 or par4ele == 1)

#test_corloc not necessary as it is retrieving data from other tested functions. 

def test_W():
    """Check ability to produce Barycentric weighting values."""

    tri = np.asarray([[0.0,0.0],[1.0,0.0],[0.5,np.sin(60)]])
    par0 = Particle(np.asarray([0.0, 0.0]))
    par1 = Particle(np.asarray([0.5, 0.0]))
    par2 = Particle(np.asarray([0.5, 1/3*np.sin(60)]))
    par3 = Particle(np.asarray([0.5, 0.5*np.sin(60)]))
    par0W = par0._W(tri, 0.5*np.sin(60))
    par1W = par1._W(tri, 0.5*np.sin(60))
    par2W = par2._W(tri, 0.5*np.sin(60))
    par3W = par3._W(tri, 0.5*np.sin(60))
    assert (par0W == np.asarray([1.0,0.0,0.0])).all() and (par1W == np.asarray([0.5,0.5,0.0])).all() and (par2W.round(2) == np.asarray([0.33,0.33,0.33])).all() and (par3W.round(2) == np.asarray([0.25,0.25,0.5])).all()


# test_update to be written. 