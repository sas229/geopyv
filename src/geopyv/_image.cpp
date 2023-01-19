#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Eigen>
#include <iostream>
#include <math.h>

// ----------------------------------
// geopyv image module C++ extensions
// ----------------------------------

using namespace Eigen;

MatrixXd _QCQT(const Ref<const MatrixXd> &Q, const Ref<const MatrixXd> &QT, const Ref<const MatrixXd> &C, const Ref<const Vector2d> &image_gs_shape, const int &border)
{
    // Define variables.
    int rows = image_gs_shape(0);
    int cols = image_gs_shape(1);
    MatrixXd QCQT(rows*6, cols*6);

    // Perform matrix multiplication to pre-compute QK_B_Kt.
    #pragma omp parallel for
    for(int j = 0; j < cols;  j++)
    {
        for(int i = 0; i < rows; i++)
        {
            int ind_row = i+border-2;
            int ind_col = j+border-2;
            QCQT.block(i*6,j*6,6,6) = Q*C.block(ind_row, ind_col, 6, 6)*QT;
        }
    }
    return QCQT;
}

// ----------------
// Python interface
// ----------------

namespace py = pybind11;

PYBIND11_MODULE(_image_extensions,m)
{
m.doc() = "C++ image module extensions for geopyv.";
m.def("_QCQT", &_QCQT, py::return_value_policy::reference_internal, "C++ extension to pre-compute Q_C_QT matrix.");
}