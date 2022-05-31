from PIL import Image as Im
from geopyv.image import Image
from geopyv.sequence import Sequence
import numpy as np 
from scipy import special as sp
import matplotlib.pyplot as plt 

class Validator:
    """Validator image class:
    1. Creates incrementally warped speckle pattern images.
    2. Performs PIV on the generated images.
    3. Compares applied and observed warps. 

    """
    def __init__(self, name = "u", key = 0):
        """
        
        Parameters
        ----------
        name : str
            Primary warp component name i.e. "u", "dudx", "d2udx2", "ROT", "SINUSOIDAL" etc.
        key : int
            strain type: 0 - homogeneous, 1 - inhomogeneous"""

        self.name = name
        self.key = key
        self.labels = []
        self.out = []

    def isg(self, image_size = 401, img_typ = "jpg", speckle_no = 7640, rho = 4/np.sqrt(8), seq_len = 50, mmin = 0, mmax = 1, typ = 0, comp = 0, comp_val = 1, a = 0, b = 0, origin = np.asarray([0.0,0.0]), noise = np.asarray([[0.0, 0.0],[0,0]])):
        """ Image Sequence Generator (isg)

        Parameters
        ----------
        image_size : int
            Size of the image.
        speckle_no : int
            Number of points to be included in the speckle pattern images.
        rho : float
            Speckle size control parameter.
        seq_len : int
            Number of images in the sequence. 
        mmin : float
            Minimum increment on warp multiplier.
        mmax : float
            Maximum increment on warp multiplier. 
        typ : int
            Multiplier type. 0 - linear, 1 - log.
        comp : int 
            0 - u or rotation
            1 - v or sinusoidal
            2 - dudx or bar
            3 - dvdx 
            4 - dudy
            5 - dvdy 
            6 - d2udx2
            7 - d2vdx2
            8 - d2udxdy
            9 - d2vdxdy
            10 - d2udy2
            11 - d2vdy2
        comp_val : float
            Value of p component. 
        a : float
            Factor.
        b : float
            Factor.
        origin:
            Centre 
        noise : numpy.ndarray
            [[pos_noise_std, int_noise_std],
            [pos_incl_bool, int_inc_bool]]

        """
        
        # Parameter initiation
        self.image_size = image_size
        self.img_typ = img_typ
        self.speckle_no = speckle_no
        self.rho = rho
        self.seq_len = seq_len
        self.mmin = mmin 
        self.mmax = mmax 
        self.typ = typ
        self.comp = comp
        self.comp_val = comp_val
        self.noise = noise
        self.a = a 
        self.b = b  
        self.origin = origin
        

        # Initial grid and speckle positions.
        [self.X, self.Y] = np.meshgrid(range(self.image_size), range(self.image_size))
        self.I_p = np.random.rand(self.speckle_no, 2)*self.image_size
        self.p = np.zeros(12)
        self.p[comp] = self.comp_val

        # Image generation
        self._mult()
        for i in range(self.seq_len):
            T_p = self._warp(i, self.I_p)
            #T_p = self._warp(0, i, self.I_p)
            grid = self._t2g(i, T_p)
            self._create(i, grid)

    def _mult(self):
        self.pm = np.zeros((self.seq_len, 12))
        self.noisem = np.zeros((self.seq_len, 2))
        self.am = np.zeros((self.seq_len))
        self.bm = np.zeros((self.seq_len))
        if self.typ == 0:
            self.mult = np.linspace(self.mmin, self.mmax, self.seq_len, endpoint = True)
        elif self.typ == 1: 
            self.mult = np.zeros((self.seq_len+1))
            self.mult[1:] = np.logspace(self.mmin, self.mmax, self.seq_len, endpoint = True)
        for i in range(self.seq_len):
            self.pm[i] = self.p*self.mult[i]
            self.noisem[i] = self.noise[0]*self.noise[1]*self.mult[i]
            self.am[i] = self.a*self.mult[i]
            self.bm[i] =  self.b*self.mult[i]

    def _warp(self, i, pt, map = False):
        """Private method that applies warp. 
        
        Parameters
        ----------
        """

        warp = np.zeros(pt.shape)
        if self.key == 0: # Homogeneous strain.
            delta = pt - self.origin
            warp[:,0] = self.pm[i][0] + self.pm[i][2]*delta[:,0] + self.pm[i][4]*delta[:,1] + 0.5*self.pm[i][6]*delta[:,0]**2 + self.pm[i][8]*delta[:,0]*delta[:,1] + 0.5*self.pm[i][10]*delta[:,1]**2
            warp[:,1] = self.pm[i][1] + self.pm[i][3]*delta[:,0] + self.pm[i][5]*delta[:,1] + 0.5*self.pm[i][7]*delta[:,0]**2 + self.pm[i][9]*delta[:,0]*delta[:,1] + 0.5*self.pm[i][11]*delta[:,1]**2      
        elif self.key*self.comp == 1: # Rotation.
            delta = pt - self.origin
            warp[:,0] = self.origin[0] + delta[:,0] * np.cos(self.am[i]) + delta[:,1] * np.sin(self.am[i])
            warp[:,1] = self.origin[1] - delta[:,0] * np.sin(self.am[i]) + delta[:,1] * np.cos(self.am[i])
        elif self.key*self.comp == 2: # Sinusoidal. 
            warp[:,0] = (self.am[i])*np.sin(2*np.pi*pt[:,0]/(self.bm[i]))*np.sin(2*np.pi*pt[:,1]/(self.bm[i]))
        
        if map == True or self.key*self.comp == 1:
            return warp
        else:
            return warp+self.I_p

#    def _warp(self, o, i, pt, map = False):
#        """Private method that applies warp. 
#        
#        Parameters
#        ----------
#        """
#
#        warp = np.zeros(pt.shape)
#        if self.key == 0: # Homogeneous strain.
#            delta = pt - self.origin
#            warp[:,0] = (self.pm[i][0]-self.pm[o][0]) + (self.pm[i][2]-self.pm[o][2])*delta[:,0] + (self.pm[i][4]-self.pm[o][4])*delta[:,1] + 0.5*(self.pm[i][6]-self.pm[o][6])*delta[:,0]**2 + (self.pm[i][8]-self.pm[o][8])*delta[:,0]*delta[:,1] + 0.5*(self.pm[i][10]-self.pm[o][10])*delta[:,1]**2
#            warp[:,1] = (self.pm[i][1]-self.pm[o][1]) + (self.pm[i][3]-self.pm[o][3])*delta[:,0] + (self.pm[i][5]-self.pm[o][5])*delta[:,1] + 0.5*(self.pm[i][7]-self.pm[o][7])*delta[:,0]**2 + (self.pm[i][9]-self.pm[o][9])*delta[:,0]*delta[:,1] + 0.5*(self.pm[i][11]-self.pm[o][11])*delta[:,1]**2      
#        elif self.key*self.comp == 1: # Rotation.
#            delta = pt - self.origin
#            warp[:,0] = self.origin[0] + delta[:,0] * np.cos(self.am[i]-self.am[o]) + delta[:,1] * np.sin(self.am[i]-self.am[o])
#            warp[:,1] = self.origin[1] - delta[:,0] * np.sin(self.am[i]-self.am[o]) + delta[:,1] * np.cos(self.am[i]-self.am[o])
#        elif self.key*self.comp == 2: # Sinusoidal. 
#            warp[:,0] = (self.am[i]-self.am[o])*np.sin(2*np.pi*pt[:,0]/(self.bm[i]-self.bm[o]))*np.sin(2*np.pi*pt[:,1]/(self.bm[i]-self.bm[o]))
#        
#        if map == True or self.key*self.comp == 1:
#            return warp
#        else:
#            return warp+self.I_p
    
    def _t2g(self, i, T_p):
        """ Target to grid (or noise applicator). Applies spatial noise to the speckle positions, generates an intensity grid and then applies
        intensity noise. 
        """
        
        grid = np.zeros((self.image_size,self.image_size))
        if self.noisem[i,0] != 0.0:
            T_p = np.random.normal(loc = T_p, scale = self.noisem[i,0])
        for j in range(len(T_p)):
            di = np.exp(-((self.X-T_p[:,0][j])**2+(self.Y-T_p[:,1][j])**2)/(2*self.rho**2))
            grid += di
        if self.noisem[i,1] != 0.0:
            grid = np.random.normal(loc = grid, scale = self.noisem[i,1])
        grid = np.clip(grid*255, 0, 255)
        return grid

    def _create(self, i, grid):
        """Method to store generated speckle pattern intensity matrix as a image file.
        
        Parameters
        ----------
        
        """
        im = Im.fromarray(np.uint8(grid))
        im.save(self.name+"_"+str(i)+"."+self.img_typ)

    def validation(self, boundary = np.asarray([[100.,100.],[100.,300.],[300.,300.],[300.,100.]]), tar_area = 20,
                    seed = np.asarray([200.,200.]), piv_order = 1, label = None):
        
        self.img_seq = self._seqload()
        self.seq = Sequence(self.img_seq)
        self.seq.mesh_adaptivity_setup(max_iterations_adaptivity = 1, verbose = True)
        self.seq.mesh_geometry_setup(area = tar_area, roi = boundary, hls = None, obj = None, sed = seed, manual = False)
        self.seq.mesh_piv_setup(p_0 = np.zeros(6*piv_order))
        self.seq.mesh()
        self._comparison()
        self.labels.append(label)
        
    def _comparison(self):
        series = np.zeros((self.seq_len-1, 2)) # Standard error array.
        base = np.zeros((len(self.seq.meshes[0].pts), 2)) # Displacement pre-image update array.
        prevref = 0
        for i in range(len(self.seq.miref)):
            if self.seq.miref[i] != prevref:
                base = ob_warp_a
            ap_warp_a = np.zeros((len(self.seq.meshes[i].pts), 2))
            ap_warp_a = self._warp(i, self.seq.meshes[i].pts, map = True) #ap_warp_a = self._warp(self.seq.miref[i], i+1, self.seq.meshes[i].pts, map = True)
            ob_warp_a = np.zeros((len(self.seq.meshes[i].pts), 2))
            for n in range(len(self.seq.meshes[i].subsets)):
                ob_warp_a[n] = np.asarray([self.seq.meshes[i].subsets[n].u, self.seq.meshes[i].subsets[n].v]) + base[n]
            series[i,1] = np.std(np.sqrt(np.sum((ap_warp_a-ob_warp_a)**2, axis=1)))
            series[i,0] = self.mult[i+1] #-self.mult[self.seq.miref[i]]
        self.out.append(series)

    def _seqload(self):
        output = np.empty(self.seq_len, dtype=object)
        for i in range(self.seq_len):
            output[i] = Image(self.name+"_"+str(i)+"."+self.img_typ)
        return output

    def plotter(self, x, y):
        """Graphical output.
        
        Parameters
        ----------
        out: validator class objects"""

        formatting = np.asarray((["o", "r"],["^", "b"],["s", "g"],["+", "o"]))
        fig, ax = plt.subplots( figsize = (9,4.5))
        for i in range(len(self.out)): 
            ax.scatter(self.out[i][:,0], self.out[i][:,1], facecolors='none', edgecolors= formatting[i,1], marker = formatting[i,0], label=self.labels[i])
        plt.legend()
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_yscale("log")
        if self.typ == 1:
            ax.set_xscale("log")
        plt.tight_layout()
        fig.savefig(self.name+"_error."+self.img_typ)

print("===================================================u====================================================")
valid = Validator(name = "u", key = 0)
valid.isg(seq_len=6, mmin = 0, mmax = 5, typ = 0, comp = 0, comp_val = -1)
valid.validation(piv_order = 1, label = r"$1^{st}$ Order")
valid.validation(piv_order = 2, label = r"$2^{st}$ Order")
print(valid.out)
valid.plotter(x=r"Displacement, $u$ ($px$)",y=r"Standard error, $\rho_{px}$ ($px$)")
fig, ax = plt.subplots(figsize = (20,20))
col = ["r","g","b","m","y"]
for i in range(len(valid.seq.meshes)): 
    ax.scatter(valid.seq.meshes[i].pts[:,0], valid.seq.meshes[i].pts[:,1], color = col[i], s = 1)
plt.savefig("u_meshes.png", dpi = 500)

#print("===================================================dudx==================================================")
#valid1 = Validator(name = "dudx", key = 0)
#valid1.isg(seq_len=50, mmin = -3, mmax = -1, typ = 1, comp = 2, comp_val = -1)
#valid1.validation(piv_order = 1,  label = r"$1^{st}$ Order")
#valid1.validation(piv_order = 2,  label = r"$2^{st}$ Order")
#valid1.plotter(x=r"$1^{st}$ Order Strain Component, $du/dx$ ($px$)",y=r"Standard error, $\rho_{px}$ ($px$)")
#
#print("===================================================dudy==================================================")
#valid2 = Validator(name = "dudy", key = 0)
#valid2.isg(seq_len=50, mmin = -3, mmax = -1, typ = 1, comp = 4, comp_val = -1)
#valid2.validation(piv_order = 1, label = r"$1^{st}$ Order")
#valid2.validation(piv_order = 2, label = r"$2^{st}$ Order")
#valid2.plotter(x=r"$1^{st}$ Order Strain Component, $du/dy$ ($px$)",y=r"Standard error, $\rho_{px}$ ($px$)")
#a = np.logspace(-5, -3, 30, endpoint = True)
#for i in range(len(a)):
#    print(i ,a[i])
#print("===================================================d2udx2================================================")
#valid3 = Validator(name = "d2udx2", key = 0)
#valid3.isg(seq_len=30, mmin = -5, mmax = -3, typ = 1, comp = 6, comp_val = -1)
#valid3.validation(piv_order = 1, label = r"$1^{st}$ Order")
#valid3.validation(piv_order = 2, label = r"$2^{st}$ Order")
#valid3.plotter(x=r"$2^{nd}$ Order Strain Component, $d^2u/dx^2$ ($px$)",y=r"Standard error, $\rho_{px}$ ($px$)")    
#
#print("===================================================d2udxdy===============================================")
#valid4 = Validator(name = "d2udxdy", key = 0)
#valid4.isg(seq_len=30, mmin = -5, mmax = -3, typ = 1, comp = 8, comp_val = -1)
#valid4.validation(piv_order = 1, label = r"$1^{st}$ Order")
#valid4.validation(piv_order = 2, label = r"$2^{st}$ Order")
#valid4.plotter(x=r"$2^{nd} Order Strain Component, $d^2u/dxdy$ ($px$)",y=r"Standard error, $\rho_{px}$ ($px$)")    
#
#print("===================================================d2udy2================================================")
#valid5 = Validator(name = "d2udy2", key = 0)
#valid5.isg(seq_len=30, mmin = -5, mmax = -3, typ = 1, comp = 10, comp_val = -1)
#valid5.validation(piv_order = 1, label = r"$1^{st}$ Order")
#valid5.validation(piv_order = 2, label = r"$2^{st}$ Order")
#valid5.plotter(x=r"$2^{nd}$ Order Shear Strain, $d^2u/dy^2$ ($px$)",y=r"Standard error, $\rho_{px}$ ($px$)")    
