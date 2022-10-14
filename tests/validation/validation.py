from PIL import Image as Im
from geopyv.image import Image
from geopyv.templates import Circle
from geopyv.sequence import Sequence
import numpy as np 
from scipy import special as sp
import matplotlib.pyplot as plt 

from geopyv.geometry.exclusions import circular_exclusion, circular_exclusion_list

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

    def isg(self, image_size = 1001, img_typ = "jpg", speckle_no = 4*7640, rho = 4/np.sqrt(8), seq_len = 50, mmin = 0, mmax = 1, typ = 0, comp = 0, comp_val = 1, a = 0, b = 0, origin = np.asarray([0.0,0.0]), noise = np.asarray([[0.0, 0.0],[0,0]])):
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
            0 - u 
            1 - v or rotation
            2 - dudx sinusoidal 
            3 - dvdx or bar
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
            print("Image {}".format(i))
            T_p = self._warp(i, self.I_p)
            grid = self._t2g(i, T_p)
            self._create(i, grid)

    def _mult(self, cut=None):
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
        if type(cut)==int:
            self.pm = self.pm[:cut]
            self.noisem = self.noisem[:cut]
            self.am = self.am[:cut]
            self.bm = self.bm[:cut]

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
            warp[:,0] =  delta[:,0] * np.cos(self.am[i]) + delta[:,1] * np.sin(self.am[i]) - delta[:,0] 
            warp[:,1] =  - delta[:,0] * np.sin(self.am[i]) + delta[:,1] * np.cos(self.am[i]) - delta[:,1] 
        elif self.key*self.comp == 2: # Sinusoidal. 
            warp[:,0] = (self.am[i])*np.sin(2*np.pi*pt[:,0]/(self.bm[i]))*np.sin(2*np.pi*pt[:,1]/(self.bm[i]))
        
        if map == True:
            return warp
        else:
            return warp+self.I_p
    
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
        im.close()

    def validation(self, target_nodes = 1681, boundary = np.asarray([[300.,300.],[300.,700.],[700.,700.],[700.,300.]]), exclusions = [],
                    seed_coord = np.asarray([500.,500.]), template = Circle(30), order = 1, adaptive_iterations = 0, label = None):
        
        self.img_seq = self._seqload()
        self.seq = Sequence(img_sequence = self.img_seq, target_nodes = target_nodes, boundary = boundary, exclusions = exclusions)
        self.seq.solve(seed_coord = seed_coord, max_iterations = 50, max_norm = 1e-5, tolerance = 0.7, template = template, order = order, adaptive_iterations = 0, alpha = 0.25, beta = 4, size_lower_bound = 1)
        self._comparison()
        self.labels.append(label)
        
    def _comparison(self):
        series = np.zeros((self.seq_len-1, 2)) # Standard error array.
        base = np.zeros((len(self.seq.meshes[0].nodes), 2)) # Displacement pre-image update array.
        for i in range(len(self.seq.meshes)):
            if self.seq.update_register[i] == 1: # If reference updated...
                    base = ob_warp_a # ... record previous 
            ap_warp_a = np.zeros((len(self.seq.meshes[i].nodes), 2))
            ap_warp_a = self._warp(i, self.seq.meshes[i].nodes, map = True) #ap_warp_a = self._warp(self.seq.f_img_index[i], i+1, self.seq.meshes[i].nodes, map = True)
            ob_warp_a = np.zeros((len(self.seq.meshes[i].nodes), 2))
            for n in range(len(self.seq.meshes[i].subsets)):
                ob_warp_a[n] = np.asarray([self.seq.meshes[i].subsets[n].u, self.seq.meshes[i].subsets[n].v]) + base[n]
            series[i,1] = np.std(np.sqrt(np.sum((ap_warp_a-ob_warp_a)**2, axis=1)))
            series[i,0] = self.mult[i+1] #-self.mult[self.seq.f_img_index[i]]
        self.out.append(series)

    def seq_info_load(self, image_size = 1001, img_typ = "jpg", speckle_no = 4*7640, rho = 4/np.sqrt(8), seq_len = 50, mmin = 0, mmax = 1, typ = 0, comp = 0, comp_val = 1, a = 0, b = 0, origin = np.asarray([0.0,0.0]), noise = np.asarray([[0.0, 0.0],[0,0]]), cut = None):

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
        self.cut = cut

        self.p = np.zeros(12)
        self.p[comp] = self.comp_val
        self._mult(cut)
        if type(self.cut) == int:
            self.seq_len = self.cut

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

        marker = ["o", "^", "o", "^", "o", "^"]
        colour = ["red", "red", "blue", "blue", "yellow", "yellow"]
        fig, ax = plt.subplots( figsize = (9,4.5))
        for i in range(len(self.out)): 
            ax.scatter(self.out[i][:,0], self.out[i][:,1], color= colour[i], marker = marker[i], label=self.labels[i])
        plt.legend()
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_yscale("log")
        if self.typ == 1:
            ax.set_xscale("log")
        plt.show()
        plt.tight_layout()
        fig.savefig(self.name+"_error."+self.img_typ)

valid = Validator(name = "u", key = 0)
valid.isg(image_size = 401, speckle_no = 7640, seq_len=2, mmin = 0, mmax = 1, typ = 0, comp = 0, comp_val = -1)

#print("===================================================u====================================================")
#valid = Validator(name = "u", key = 0)
##valid.isg(seq_len=40, mmin = 0, mmax = 1, typ = 0, comp = 0, comp_val = -1)
#valid.seq_info_load(seq_len=40, mmin = 0, mmax = 1, typ = 0, comp = 0, comp_val = -1)
#valid.validation(order = 1, label = r"$1^{st}$ Order, 25px dia.", template = Circle(25))
#valid.validation(order = 2, label = r"$2^{st}$ Order, 25px dia.", template = Circle(25))
#valid.validation(order = 1, label = r"$1^{st}$ Order, 50px dia.", template = Circle(50))
#valid.validation(order = 2, label = r"$2^{st}$ Order, 50px dia.", template = Circle(50))
#valid.validation(order = 1, label = r"$1^{st}$ Order, 100px dia.", template = Circle(100))
#valid.validation(order = 2, label = r"$2^{st}$ Order, 100px dia.", template = Circle(100))
#valid.plotter(x=r"Displacement, $u$ ($px$)",y=r"Standard error, $\rho_{px}$ ($px$)")

#print("===================================================dudx==================================================")
#valid1 = Validator(name = "dudx", key = 0)
#valid1.isg(seq_len=40, mmin = -6, mmax = -1, typ = 1, comp = 2, comp_val = -1)
#valid1.validation(order = 1, label = r"$1^{st}$ Order, 25px dia.", template = Circle(25))
#valid1.validation(order = 2, label = r"$2^{st}$ Order, 25px dia.", template = Circle(25))
#valid1.validation(order = 1, label = r"$1^{st}$ Order, 50px dia.", template = Circle(50))
#valid1.validation(order = 2, label = r"$2^{st}$ Order, 50px dia.", template = Circle(50))
#valid1.validation(order = 1, label = r"$1^{st}$ Order, 100px dia.", template = Circle(100))
#valid1.validation(order = 2, label = r"$2^{st}$ Order, 100px dia.", template = Circle(100))
#valid1.plotter(x=r"$1^{st}$ Order Strain Component, $du/dx$ ($px$)",y=r"Standard error, $\rho_{px}$ ($px$)")
#
#print("===================================================dudy==================================================")
#valid2 = Validator(name = "dudy", key = 0)
#valid2.isg(seq_len=40, mmin = -6, mmax = -1, typ = 1, comp = 4, comp_val = -1)
##valid2.seq_info_load(seq_len=40, mmin = -3, mmax = -1, typ = 1, comp = 4, comp_val = -1)
#valid2.validation(order = 1, label = r"$1^{st}$ Order, 25px dia.", template = Circle(25))
#valid2.validation(order = 2, label = r"$2^{st}$ Order, 25px dia.", template = Circle(25))
#valid2.validation(order = 1, label = r"$1^{st}$ Order, 50px dia.", template = Circle(50))
#valid2.validation(order = 2, label = r"$2^{st}$ Order, 50px dia.", template = Circle(50))
#valid2.validation(order = 1, label = r"$1^{st}$ Order, 100px dia.", template = Circle(100))
#valid2.validation(order = 2, label = r"$2^{st}$ Order, 100px dia.", template = Circle(100))
#valid2.plotter(x=r"$1^{st}$ Order Strain Component, $du/dy$ ($px$)",y=r"Standard error, $\rho_{px}$ ($px$)")
#
#print("===================================================d2udx2================================================")
#valid3 = Validator(name = "d2udx2", key = 0)
##valid3.isg(seq_len=40, mmin = -6, mmax = -3, typ = 1, comp = 6, comp_val = 1)
#valid3.seq_info_load(seq_len=40, mmin = -6, mmax = -3, typ = 1, comp = 6, comp_val = 1, cut = 35)
#valid3.validation(order = 1, label = r"$1^{st}$ Order, 25px dia.", template = Circle(25))
#valid3.validation(order = 2, label = r"$2^{st}$ Order, 25px dia.", template = Circle(25))
#valid3.validation(order = 1, label = r"$1^{st}$ Order, 50px dia.", template = Circle(50))
#valid3.validation(order = 2, label = r"$2^{st}$ Order, 50px dia.", template = Circle(50))
#valid3.validation(order = 1, label = r"$1^{st}$ Order, 100px dia.", template = Circle(100))
#valid3.validation(order = 2, label = r"$2^{st}$ Order, 100px dia.", template = Circle(100))
#valid3.plotter(x=r"$2^{nd}$ Order Strain Component, $d^2u/dx^2$ ($px$)",y=r"Standard error, $\rho_{px}$ ($px$)")    
#print("===================================================d2udx2================================================")
#valid3 = Validator(name = "d2udx2", key = 0)
##valid3.isg(seq_len=40, mmin = -6, mmax = -3, typ = 1, comp = 6, comp_val = 1)
#valid3.seq_info_load(seq_len=40, mmin = -6, mmax = -3, typ = 1, comp = 6, comp_val = 1, cut = 32)
#valid3.validation(order = 1, label = r"$1^{st}$ Order, 20px dia.", template = Circle(20))
#valid3.validation(order = 2, label = r"$2^{st}$ Order, 20px dia.", template = Circle(20))
#valid3.validation(order = 1, label = r"$1^{st}$ Order, 30px dia.", template = Circle(30))
#valid3.validation(order = 2, label = r"$2^{st}$ Order, 30px dia.", template = Circle(30))
#valid3.validation(order = 1, label = r"$1^{st}$ Order, 40px dia.", template = Circle(40))
#valid3.validation(order = 2, label = r"$2^{st}$ Order, 40px dia.", template = Circle(40))
#valid3.validation(order = 1, label = r"$1^{st}$ Order, 50px dia.", template = Circle(50))
#valid3.validation(order = 2, label = r"$2^{st}$ Order, 50px dia.", template = Circle(50))
#valid3.validation(order = 1, label = r"$1^{st}$ Order, 60px dia.", template = Circle(60))
#valid3.validation(order = 2, label = r"$2^{st}$ Order, 60px dia.", template = Circle(60))
#valid3.validation(order = 1, label = r"$1^{st}$ Order, 70px dia.", template = Circle(70))
#valid3.validation(order = 2, label = r"$2^{st}$ Order, 70px dia.", template = Circle(70))
#valid3.validation(order = 1, label = r"$1^{st}$ Order, 80px dia.", template = Circle(80))
#valid3.validation(order = 2, label = r"$2^{st}$ Order, 80px dia.", template = Circle(80))
#valid3.validation(order = 1, label = r"$1^{st}$ Order, 90px dia.", template = Circle(90))
#valid3.validation(order = 2, label = r"$2^{st}$ Order, 90px dia.", template = Circle(90))
#valid3.validation(order = 1, label = r"$1^{st}$ Order, 100px dia.", template = Circle(100))
#valid3.validation(order = 2, label = r"$2^{st}$ Order, 100px dia.", template = Circle(100))
#
#marker = ["o", "^"]
#colour = ["red", "blue"]
#subset_dia = range(20,101,10)
#out = np.asarray(valid3.out)
#s_1_10 = out[::2, 10, 1]
#s_2_10 = out[1::2, 10, 1]
#
#s_1_20 = out[::2, 20, 1]
#s_2_20 = out[1::2, 20, 1]
#s_1_30 = out[::2, 30, 1]
#s_2_30 = out[1::2, 30, 1]
#
#fig, ax = plt.subplots( figsize = (9,4.5))
#ax.scatter(subset_dia, s_1_10, color = "red", marker = "o", label = r"$1^{st}$ Order")
#ax.scatter(subset_dia, s_2_10, color = "blue", marker = "^", label = r"$2^{nd}$ Order")
#plt.legend()
#ax.set_xlabel(r"Subset size, s ($px$)")
#ax.set_ylabel(r"Standard error, $\rho_{px}$ ($px$)")
#plt.show()
#plt.tight_layout()
#fig.savefig("subset_size_error_10.png")
#fig, ax = plt.subplots( figsize = (9,4.5))
#ax.scatter(subset_dia, s_1_20, color = "red", marker = "o", label = r"$1^{st}$ Order")
#ax.scatter(subset_dia, s_2_20, color = "blue", marker = "^", label = r"$2^{nd}$ Order")
#plt.legend()
#ax.set_xlabel(r"Subset size, s ($px$)")
#ax.set_ylabel(r"Standard error, $\rho_{px}$ ($px$)")
#plt.show()
#plt.tight_layout()
#fig.savefig("subset_size_error_20.png")
#fig, ax = plt.subplots( figsize = (9,4.5))
#ax.scatter(subset_dia, s_1_30, color = "red", marker = "o", label = r"$1^{st}$ Order")
#ax.scatter(subset_dia, s_2_30, color = "blue", marker = "^", label = r"$2^{nd}$ Order")
#plt.legend()
#ax.set_xlabel(r"Subset size, s ($px$)")
#ax.set_ylabel(r"Standard error, $\rho_{px}$ ($px$)")
#plt.show()
#plt.tight_layout()
#fig.savefig("subset_size_error_30.png")

#valid3.plotter(x=r"$2^{nd}$ Order Strain Component, $d^2u/dx^2$ ($px$)",y=r"Standard error, $\rho_{px}$ ($px$)")    
#print("===================================================d2udxdy===============================================")
#valid4 = Validator(name = "d2udxdy", key = 0)
##valid4.isg(seq_len=40, mmin = -5, mmax = -3, typ = 1, comp = 8, comp_val = -1)
#valid4.seq_info_load(seq_len=40, mmin = -5, mmax = -3, typ = 1, comp = 8, comp_val = 1, cut = 35)
#valid4.validation(order = 1, label = r"$1^{st}$ Order, 25px dia.", template = Circle(25))
#valid4.validation(order = 2, label = r"$2^{st}$ Order, 25px dia.", template = Circle(25))
#valid4.validation(order = 1, label = r"$1^{st}$ Order, 50px dia.", template = Circle(50))
#valid4.validation(order = 2, label = r"$2^{st}$ Order, 50px dia.", template = Circle(50))
#valid4.validation(order = 1, label = r"$1^{st}$ Order, 100px dia.", template = Circle(100))
#valid4.validation(order = 2, label = r"$2^{st}$ Order, 100px dia.", template = Circle(100))
#valid4.plotter(x=r"$2^{nd} Order Strain Component, $d^2u/dxdy$ ($px$)",y=r"Standard error, $\rho_{px}$ ($px$)")    
#
#print("===================================================d2udy2================================================")
#valid5 = Validator(name = "d2udy2", key = 0)
##valid5.isg(seq_len=40, mmin = -5, mmax = -3, typ = 1, comp = 10, comp_val = -1)
#valid4.seq_info_load(seq_len=40, mmin = -5, mmax = -3, typ = 1, comp = 10, comp_val = 1, cut = 35)
#valid5.validation(order = 1, label = r"$1^{st}$ Order, 25px dia.", template = Circle(25))
#valid5.validation(order = 2, label = r"$2^{st}$ Order, 25px dia.", template = Circle(25))
#valid5.validation(order = 1, label = r"$1^{st}$ Order, 50px dia.", template = Circle(50))
#valid5.validation(order = 2, label = r"$2^{st}$ Order, 50px dia.", template = Circle(50))
#valid5.validation(order = 1, label = r"$1^{st}$ Order, 100px dia.", template = Circle(100))
#valid5.validation(order = 2, label = r"$2^{st}$ Order, 100px dia.", template = Circle(100))
#valid5.plotter(x=r"$2^{nd}$ Order Shear Strain, $d^2u/dy^2$ ($px$)",y=r"Standard error, $\rho_{px}$ ($px$)")    

#print("===================================================Rotation================================================")
#valid6 = Validator(name = "Rotation", key = 1)
##valid6.isg(seq_len=40, mmin = -4, mmax = 1, typ = 1, comp = 1, a = np.pi*100/180, comp_val = 1, origin = np.asarray([200,200]))
#valid6.seq_info_load(seq_len=40, mmin = -4, mmax = 1, typ = 1, comp = 1, a = np.pi*100/180, comp_val = 1, origin = np.asarray([500,500]), cut = 29)
#valid6.validation(order = 1, label = r"$1^{st}$ Order, 25px dia.", template = Circle(25))
#valid6.validation(order = 2, label = r"$2^{st}$ Order, 25px dia.", template = Circle(25))
#valid6.validation(order = 1, label = r"$1^{st}$ Order, 50px dia.", template = Circle(50))
#valid6.validation(order = 2, label = r"$2^{st}$ Order, 50px dia.", template = Circle(50))
#valid6.validation(order = 1, label = r"$1^{st}$ Order, 100px dia.", template = Circle(100))
#valid6.validation(order = 2, label = r"$2^{st}$ Order, 100px dia.", template = Circle(100))
#valid6.plotter(x=r"Rotation, $\theta$ ($^o$)",y=r"Standard error, $\rho_{px}$ ($px$)") 


#fig, ax = plt.subplots( figsize = (9,4.5))
#for i in range(len(valid5.seq.meshes[0].subsets)):
#    x = np.ones(len(valid5.seq.meshes)+1)
#    y = np.ones(len(valid5.seq.meshes)+1)
#    x *= valid5.seq.meshes[0].subsets[i].f_coord[0]
#    y *= valid5.seq.meshes[0].subsets[i].f_coord[1]
#    for j in range(len(valid5.seq.meshes)):
#        x[j+1] += valid5.seq.meshes[j].subsets[i].u
#        y[j+1] += valid5.seq.meshes[j].subsets[i].v
#    ax.plot(x,y)
#fig.savefig("map.jpg")

#fig, ax = plt.subplots(figsize = (20,20))
#col = ["r","g","b","m","y"]
#for i in range(len(valid.seq.meshes)): 
#    ax.scatter(valid.seq.meshes[i].nodes[:,0], valid.seq.meshes[i].nodes[:,1], color = col[i], s = 10)
#plt.savefig("u_meshes.png", dpi = 500)

