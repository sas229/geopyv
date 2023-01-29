.. _Image:

Image
-----

The images used with `geopyv` are first converted to grayscale and then pre-filtered using a 5 by 5 Gaussian
smoothing kernel with a standard deviation of 1.1, which is done in order to reduce bias errors as demonstrated by 
:cite:t:`pan2013bias`. Digital images, consisting of pixel intensity arrays, are discrete representations of a continuous field,
hence, image intensity interpolation is a necessary operation in all iterative local DIC algorithms. In `geopyv`, bi-quintic 
B-spline interpolation is adopted, which although computationally expensive, minimises bias errors as demonstrated by 
:cite:t:`schreier2002systematic`. 

.. todo::
    
    - Show transformation of a subset from colour to grayscale to Guassian pre-filtered.

Computation of B-spline coefficients
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The B-spline coefficents are pre-computed for all images using recursive 1D deconvolution. The grayscale reference image :math:`f`, is first padded with a replicated border (default of 20 pixels) to create the padded reference image :math:`f_{p}` of size :math:`(i,j)` pixels. The quintic B-spline kernel :math:`\mathbf{b}` is:
    
.. math::
    :label: k
    
    \mathbf{k} = \begin{bmatrix} 1/120 &  13/60 & 11/20 & 13/60 & 1/120 & 0 \end{bmatrix}

A null vector :math:`\mathbf{b}_{x} = 0_{1,j}` is then created, into which the quintic B-spline kernel :math:`\mathbf{b}` is inserted as follows:
    
.. math::
    :label: null_1
    
    \mathbf{b}_{x\left( 0:3 \right)} = \mathbf{k}_{\left( 3:5 \right)} \text{ and } \mathbf{b}_{\left(j-3:j \right)} = \mathbf{k}_{\left( 0:3 \right)}
    
The Fast Fourier Transform (FFT) of this padded kernel vector is used to divide the FFT of the :math:`i`-th row of the padded grayscale image :math:`\mathbf{I}_{p}`, after which the inverse FFT is taken for each row:
    
.. math::
    :label: C_FFT
    
    \mathbf{C}_{\left( i, : \right)} = F^{-1}\left[ \frac{ F\left[ f_{p\left( i, : \right)} \right] }{ F\left[ \mathbf{b}_{x} \right] } \right]
    
where :math:`F` and :math:`F^{-1}` represent the FFT and inverse FFT, respectively. \
    
A second null vector :math:`\mathbf{b}_{y} = 0_{1,i}` is then created, into which the B-spline kernel is inserted as follows:

.. math::
    :label: null_2
    
    \mathbf{b}_{y \left( 0:3 \right)} = \mathbf{b}_{\left( 3:5 \right)} \text{ and } \mathbf{b}_{y \left( i-3:i \right)} = \mathbf{b}_{\left( 0:3 \right)}
    
The FFT of this padded kernel vector is used to divide the FFT of the :math:`j`-th column of the B-spline coefficient matrix :math:`\mathbf{C}`,
after which the inverse FFT is taken for each column to yield the final matrix of B-spline coefficients:

.. math::
    :label: C_kernel
    
    \mathbf{C}_{\left( :, j \right)} = F^{-1}\left[ \frac{ F\left[ \mathbf{C}_{\left( :, j \right)} \right] }{ F\left[ \mathbf{b}_{y} \right] } \right]

This array of coefficients is computed by :py:meth:`~geopyv.image.Image._get_C`. The same technique is used to calculate the B-spline coefficients for the target image :math:`g`. 

.. warning::
    
    The padded image must be at least as large as the quintic B-spline kernel in order for this method to function correctly (i.e. i > 5, j > 5). 

Precomputation of interpolant array
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The bi-quintic B-spline kernel :math:`\mathbf{Q}` is:
        
.. math::
    :label: Q
    
    \mathbf{Q} = \begin{bmatrix}
                    1/120 &  13/60 & 11/20 & 13/60 & 1/120 &     0 \\
                    -1/24 &  -5/12 &      0 &  5/12 &  1/24 &     0 \\
                    1/12 &    1/6 &   -1/2 &   1/6 &  1/12 &     0 \\
                    -1/12 &    1/6 &      0 &  -1/6 &  1/12 &     0 \\
                    1/24 &   -1/6 &    1/4 &  -1/6 &  1/24 &     0 \\
                    -1/120 &   1/24 &  -1/12 & -1/12 & -1/24 & 1/120 
                    \end{bmatrix} 
                    
and :math:`\mathbf{Q}^T` is its transpose. 

The :math:`\mathbf{C}_\left(i-2:i+3, j-2:j+3\right)` matrix is a subset of the B-spline coefficient matrix :math:`\mathbf{C}` computed by :py:meth:`~geopyv.image.Image._get_C`, where :math:`i` and :math:`j` are the image coordinates:
    
.. math::
    :label: C

    \mathbf{C}_\left(i-2:i+3, j-2:j+3\right) = \begin{bmatrix}
                    \mathbf{C}_\left(i-2,j-2 \right) &  \mathbf{C}_\left(i-1,j-2 \right) & \mathbf{C}_\left(i,j-2 \right) & \mathbf{C}_\left(i+1,j-2 \right) & \mathbf{C}_\left(i+2,j-2 \right) & \mathbf{C}_\left(i+3,j-2 \right) \\
                    \mathbf{C}_\left(i-2,j-1 \right) &  \mathbf{C}_\left(i-1,j-1 \right) & \mathbf{C}_\left(i,j-1 \right) & \mathbf{C}_\left(i+1,j-1 \right) & \mathbf{C}_\left(i+2,j-1 \right) & \mathbf{C}_\left(i+3,j-1 \right) \\
                    \mathbf{C}_\left(i-2,j \right)   &  \mathbf{C}_\left(i-1,j \right)   & \mathbf{C}_\left(i,j \right)   & \mathbf{C}_\left(i+1,j \right)   & \mathbf{C}_\left(i+2,j \right)   & \mathbf{C}_\left(i+3,j \right)   \\
                    \mathbf{C}_\left(i-2,j+1 \right) &  \mathbf{C}_\left(i-1,j+1 \right) & \mathbf{C}_\left(i,j+1 \right) & \mathbf{C}_\left(i+1,j+1 \right) & \mathbf{C}_\left(i+2,j+1 \right) & \mathbf{C}_\left(i+3,j+1 \right) \\
                    \mathbf{C}_\left(i-2,j+2 \right) &  \mathbf{C}_\left(i-1,j+2 \right) & \mathbf{C}_\left(i,j+2 \right) & \mathbf{C}_\left(i+1,j+2 \right) & \mathbf{C}_\left(i+2,j+2 \right) & \mathbf{C}_\left(i+3,j+2 \right) \\
                    \mathbf{C}_\left(i-2,j+3 \right) &  \mathbf{C}_\left(i-1,j+3 \right) & \mathbf{C}_\left(i,j+3 \right) & \mathbf{C}_\left(i+1,j+3 \right) & \mathbf{C}_\left(i+2,j+3 \right) & \mathbf{C}_\left(i+3,j+3 \right) 
                \end{bmatrix} 

The following matrix is pre-computed from these quantities for all pixels in the image by :py:meth:`~geopyv.image.Image._get_QCQT`:

.. math::
    :label: QCQT

    \mathbf{Q} \cdot \mathbf{C}_{\left(\left\lfloor x \right\rfloor-2:\left\lfloor x \right\rfloor+3, \left\lfloor y \right\rfloor-2:\left\lfloor y \right\rfloor+3\right)} \cdot \mathbf{Q}^T

.. note::
    
    This pre-computation - although computationally efficient when used in repeated image intensity interpolation - requires a significant amount of memory, particularly for large images.

Image intensity interpolation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In general, the intensity of arbitrary image coordinates are estimated using bi-quintic B-spline image intensity interpolation. First, the sub-pixel component of the position of each point in the subset is computed as follows from the current coordinates:

.. math::
    :label: delta_x_y

    \begin{array}{c}
    \delta x=x-\lfloor x\rfloor \\
    \delta y=y-\lfloor y\rfloor
    \end{array}

where :math:`\lfloor x\rfloor` and :math:`\lfloor y\rfloor` are the floor of the coordinates :math:`x` and :math:`y`. The interpolated pixel intensity at the current sub-pixel coordinate :math:`(x, y)` in the reference image :math:`f`, defined as :math:`f(x, y)`, is then calculated by performing the following operation:

.. math::
    :label: f_x_y

    f(x, y)=\left[\begin{array}{llllll}
    1 & \delta y & \delta y^{2} & \delta y^{3} & \delta y^{4} & \delta y^{5}
    \end{array}\right] \cdot \mathbf{Q} \cdot \mathbf{C}_{f(\lfloor x\rfloor-2:\lfloor x\rfloor+3,\lfloor y\rfloor-2:\lfloor y\rfloor+3)} \cdot \mathbf{Q}^T \cdot\left[\begin{array}{c}
    1 \\
    \delta x \\
    \delta x^{2} \\
    \delta x^{3} \\
    \delta x^{4} \\
    \delta x^{5}
    \end{array}\right]

where :math:`\mathbf{Q} \cdot \mathbf{C}_{f} \cdot \mathbf{Q}^T` is precomputed for the entirety of image :math:`f` by :py:meth:`~geopyv.image.Image._get_QCQT`. The same method is used to interpolate pixel intensitites for both the reference image :math:`f` and the target image :math:`g`.