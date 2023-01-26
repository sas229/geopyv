#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Eigen>
#include <math.h>
#include <iostream>
#include <stdexcept>

// -----------------------------------
// geopyv subset module C++ extensions
// -----------------------------------

using namespace Eigen;
using namespace std;

MatrixXd _f_coords(
    const Ref<const VectorXd> &coord,
    const Ref<const MatrixXd> &template_coords
     )
{
    // Define variables.
    int n = template_coords.rows();
    MatrixXd f_coords(n, 2);

    // Compute the reference coordinates using the subset template.
    for(int i = 0; i < n;  i++)
    {
        f_coords(i,0) = template_coords(i,0) + coord(0);
        f_coords(i,1) = template_coords(i,1) + coord(1);
    }
    return f_coords;
}

double _Delta_f(
    const Ref<const VectorXd> &f,
    double &f_m
    )
{
    // Define variables.
    int n = f.rows(), i;
    double sum_delta_f_sq = 0, Delta_f;

    // Compute the square root of the sum of delta squared.
    for(int i = 0; i < n;  i++)
    {
        sum_delta_f_sq += pow((f(i)-f_m), 2);
    }
    Delta_f = sqrt(sum_delta_f_sq);

    return Delta_f;
}

double _f_m(const Ref<const VectorXd> &f)
{
    return f.mean();
}

MatrixXd _grad(
    const Ref<const MatrixXd> &coords,
    const Ref<const MatrixXd> &QCQT
    )
{
    // Define variables.
    int n = coords.rows(), x_floor, y_floor;
    double delta_x, delta_y, one = 1.0, zero = 0.0;
    typedef Matrix<double, 6, 1> Vector6d;
    Vector6d delta_x_vec(6), delta_y_vec(6);
    MatrixXd grad(n,2);
    
    // Compute interpolated intensities.
    for(int i = 0; i < n;  i++)
    {   
        // Calculate floor and sub-pixel components.
        x_floor = std::floor(coords(i,0));
        y_floor = std::floor(coords(i,1));
        delta_x = coords(i,0) - x_floor;
        delta_y = coords(i,1) - y_floor;

        // Gradient in x direction.
        delta_x_vec(0) = zero;
        delta_x_vec(1) = one;
        delta_x_vec(2) = zero;
        delta_x_vec(3) = zero;
        delta_x_vec(4) = zero;
        delta_x_vec(5) = zero;
        delta_y_vec(0) = one;
        delta_y_vec(1) = delta_y;
        delta_y_vec(2) = pow(delta_y,2);
        delta_y_vec(3) = pow(delta_y,3);
        delta_y_vec(4) = pow(delta_y,4);
        delta_y_vec(5) = pow(delta_y,5);
        grad(i,0) = (delta_y_vec.transpose()*QCQT.block(y_floor*6, x_floor*6, 6, 6))*delta_x_vec;

        // Gradient in y direction.
        delta_x_vec(0) = one;
        delta_x_vec(1) = delta_x;
        delta_x_vec(2) = pow(delta_x,2);
        delta_x_vec(3) = pow(delta_x,3);
        delta_x_vec(4) = pow(delta_x,4);
        delta_x_vec(5) = pow(delta_x,5);
        delta_y_vec(0) = zero;
        delta_y_vec(1) = one;
        delta_y_vec(2) = zero;
        delta_y_vec(3) = zero;
        delta_y_vec(4) = zero;
        delta_y_vec(5) = zero;
        grad(i,1) = (delta_y_vec.transpose()*QCQT.block(y_floor*6, x_floor*6, 6, 6))*delta_x_vec;
        
    }
    return grad;
}

double _SSSIG(
    const Ref<const MatrixXd> &coords,
    const Ref<const MatrixXd> &grad
    )
{

    // Define variables.
    int n = coords.rows(), x_floor, y_floor;
    double SSSIG = 0, dx, dy;

    // Compute SSSIG.
    for(int i = 0; i < n;  i++)
    {
        x_floor = std::floor(coords(i,0));
        y_floor = std::floor(coords(i,1));
        dx = grad(i,0);
        dy = grad(i,1);
        SSSIG += 0.5*(pow(dx,2) + pow(dy,2));
    }
    return SSSIG;
}

VectorXd _intensity(
    const Ref<const MatrixXd> &coords,
    const Ref<const MatrixXd> &QCQT
    )
{
    // Define variables.
    int n = coords.rows(), x_floor, y_floor;
    double delta_x, delta_y, one = 1.0;
    VectorXd intensity(n);
    typedef Matrix<double, 6, 1> Vector6d;
    Vector6d delta_x_vec(6), delta_y_vec(6);

    // Compute interpolated intensities.
    for(int i = 0; i < n;  i++)
    {
        x_floor = std::floor(coords(i,0));
        y_floor = std::floor(coords(i,1));
        if ((x_floor < 0 || x_floor > QCQT.cols() / 6) || (y_floor < 0 || y_floor > QCQT.rows() / 6))
        {
            throw std::invalid_argument("Warp strayed outside of image boundary.");
        }
        delta_x = coords(i,0) - x_floor;
        delta_y = coords(i,1) - y_floor;
        delta_x_vec(0) = one;
        delta_y_vec(0) = one;
        for(int j = 1; j < 6;  j++) {
            delta_x_vec(j) = delta_x_vec(j - 1) * delta_x;
            delta_y_vec(j) = delta_y_vec(j - 1) * delta_y;
        }
        intensity(i) = (delta_y_vec.transpose()*QCQT.block(y_floor*6, x_floor*6, 6, 6))*delta_x_vec;
    }
    return intensity;
}

VectorXd _g_coord(
    const Ref<const VectorXd> &f_coord,
    const Ref<const VectorXd> &p
    )
{
    // Define variables.
    VectorXd g_coord(2);

    // Compute g coordinate.
    g_coord(0) = f_coord(0) + p(0);
    g_coord(1) = f_coord(1) + p(1);
    
    return g_coord;
}

double _sigma_intensity(
    const Ref<const VectorXd> &f,
    double &f_m
    )
{

    // Define variables.
    int n = f.rows();
    double sum_delta_f = 0, sigma;

    // Compute the standard deviation of the subset intensity.
    for(int i = 0; i < n;  i++)
    {
        sum_delta_f += pow((f(i)-f_m), 2);
    }
    sigma = pow((sum_delta_f/n), 0.5);
    return sigma;
}

MatrixXd _g_coords(
    const Ref<const VectorXd> &coord,
    const Ref<const VectorXd> &p,
    const Ref<const MatrixXd> &f_coords
    )
{
    // Define variables.
    int n = f_coords.rows();
    int m = p.size();
    double x_c = coord(0), y_c = coord(1), x, y, Delta_x, Delta_y;
    MatrixXd g_coords(n, 2);

    // Compute the target coordinates using the reference coordinates and the warp vector.
    for(int i = 0; i < n;  i++)
    {   
        x = f_coords(i,0);
        y = f_coords(i,1);
        Delta_x = x - x_c;
        Delta_y = y - y_c;
        if (m <=7){
            double u = p(0), v = p(1), u_x = p(2), v_x = p(3), u_y = p(4), v_y = p(5);
            g_coords(i,0) = x + u + u_x*Delta_x + u_y*Delta_y;
            g_coords(i,1) = y + v + v_x*Delta_x + v_y*Delta_y;
        }
        else if (m > 7){
            double u = p(0), v = p(1), u_x = p(2), v_x = p(3), u_y = p(4), v_y = p(5);
            double u_xx = p(6), v_xx = p(7), u_xy = p(8), v_xy = p(9), u_yy = p(10), v_yy = p(11);
            g_coords(i,0) = x + u + u_x*Delta_x + u_y*Delta_y + 0.5*u_xx*pow(Delta_x,2) 
                + u_xy*Delta_x*Delta_y + 0.5*u_yy*pow(Delta_y,2); 
            g_coords(i,1) = y + v + v_x*Delta_x + v_y*Delta_y + 0.5*v_xx*pow(Delta_x,2)
                + v_xy*Delta_x*Delta_y + 0.5*v_yy*pow(Delta_y,2); 
        }
        
    }
    return g_coords;
}

double _Delta_g(
    const Ref<const VectorXd> &g,
    double &g_m
    )
{
    // Define variables.
    int n = g.rows(), i;
    double sum_delta_g_sq = 0, Delta_g;

    // Compute the square root of the sum of delta squared.
    for(int i = 0; i < n;  i++)
    {
        sum_delta_g_sq += pow((g(i)-g_m), 2);
    }
    Delta_g = sqrt(sum_delta_g_sq);

    return Delta_g;
}

double _g_m(const Ref<const VectorXd> &g)
{
    return g.mean();
}

MatrixXd _sdi(
    const Ref<const VectorXd> &coord,
    const Ref<const MatrixXd> &coords,
    const Ref<const MatrixXd> &grad,
    const Ref<const VectorXd> &p
    )
{
    // Check order of warp function.
    int k = p.size();
    int m;
    if (k <=7)
    {
        m = 6;
    } else if (k > 7)
    {
        m = 12;
    }
    
    // Define variables.
    int n = coords.rows();
    double Delta_x, Delta_y;
    MatrixXd sdi(n,m);

    // Compute the steepest descent images.
    for(int i = 0; i < n;  i++)
    {
        Delta_x = coords(i,0) - coord(0);
        Delta_y = coords(i,1) - coord(1);
        sdi(i,0) = grad(i,0);
        sdi(i,1) = grad(i,1);
        sdi(i,2) = sdi(i,0)*Delta_x;
        sdi(i,3) = sdi(i,1)*Delta_x;
        sdi(i,4) = sdi(i,0)*Delta_y;
        sdi(i,5) = sdi(i,1)*Delta_y;
        if (m == 12)
        {
            sdi(i,6) = sdi(i,0)*0.5*pow(Delta_x,2);
            sdi(i,7) = sdi(i,1)*0.5*pow(Delta_x,2);
            sdi(i,8) = sdi(i,0)*Delta_x*Delta_y;
            sdi(i,9) = sdi(i,1)*Delta_x*Delta_y;
            sdi(i,10) = sdi(i,0)*0.5*pow(Delta_y,2);
            sdi(i,11) = sdi(i,1)*0.5*pow(Delta_y,2);
        }
    }
    return sdi;
}

MatrixXd _hessian(
    const Ref<const MatrixXd> &sdi
    )
{
    // Define variables.
    int m = sdi.cols(), n = sdi.cols();
    VectorXd sdi_i, sdi_j, sdi_dot_sdi;
    MatrixXd hessian(m,n);

    // Compute the Gauss-Newton approximation to the Hessian.
    // Summate in the upper diagonal and then copy to lower diagonal.
    for(int i = 0; i < m;  i++)
    {
        for(int j = i; j < n;  j++)
        {
            sdi_i = sdi.col(i);
            sdi_j = sdi.col(j);
            sdi_dot_sdi = sdi_i.transpose()*sdi_j;
            hessian(i,j) = sdi_dot_sdi.sum();
            hessian(j,i) = hessian(i,j);
        }
    }

    return hessian;
}

VectorXd _Delta_p_ICGN(
    const Ref<const MatrixXd> &hessian,
    const Ref<const VectorXd> &f,
    const Ref<const VectorXd> &g,
    const double &f_m,
    const double &g_m,
    const double &Delta_f,
    const double &Delta_g,
    const Ref<const MatrixXd> &sdi
    )
{
    // Define variables.
    int m = sdi.cols(), n = f.rows();
    VectorXd grad_znssd = VectorXd::Zero(m);
    MatrixXd inv_hessian(m,m);
    VectorXd Delta_p(m);
    
    // Compute gradient of correlation coefficient.
    for(int j = 0; j < m;  j++)
    {
        for(int i = 0; i < n;  i++)
        {
        grad_znssd(j) += sdi(i,j)*((f(i)-f_m)-((Delta_f/Delta_g)*(g(i)-g_m)));
        }
    }

    // Invert hessian and calculate the deformation parameter vector increment.
    inv_hessian = hessian.inverse();
    Delta_p = -inv_hessian*grad_znssd;

    return Delta_p;
}

VectorXd _Delta_p_FAGN(
    const Ref<const MatrixXd> &hessian,
    const Ref<const VectorXd> &f,
    const Ref<const VectorXd> &g,
    const double &f_m,
    const double &g_m,
    const double &Delta_f,
    const double &Delta_g,
    const Ref<const MatrixXd> &sdi
    )
{
    // Define variables.
    int m = sdi.cols(), n = f.rows();
    VectorXd grad_znssd = VectorXd::Zero(m);
    MatrixXd inv_hessian(m,m);
    VectorXd Delta_p(m);
    
    // Compute gradient of correlation coefficient.
    for(int j = 0; j < m;  j++)
    {
        for(int i = 0; i < n;  i++)
        {
        grad_znssd(j) += sdi(i,j)*(((f(i)-f_m)*(Delta_g/Delta_f))-(g(i)-g_m));
        }
    }

    // Invert hessian and calculate the deformation parameter vector increment.
    inv_hessian = hessian.inverse();
    Delta_p = inv_hessian*grad_znssd;

    return Delta_p;
}

VectorXd _p_new_ICGN(
    const Ref<const VectorXd> &p,
    const Ref<const VectorXd> &Delta_p
    )
{
    //Define variables.
    int k = p.size();
    VectorXd p_new = VectorXd::Zero(k);
    if (k == 6){
        // Define variables for a first-order subset.
        int m = 3, n =3;
        MatrixXd W_old(m,n), W_delta(m,n), W_new(m,n), inv_W_delta(m,n);
        double u = p(0), v = p(1), u_x = p(2), v_x = p(3), u_y = p(4), v_y = p(5);
        double Delta_u = Delta_p(0), Delta_v = Delta_p(1), Delta_u_x = Delta_p(2);
        double Delta_v_x = Delta_p(3), Delta_u_y = Delta_p(4), Delta_v_y = Delta_p(5);

        // Assemble homogenous warp function and warp function increment.
        W_old << 1+u_x, u_y, u, v_x, 1+v_y, v, 0, 0, 1;
        W_delta << 1+Delta_u_x, Delta_u_y, Delta_u, Delta_v_x, 1+Delta_v_y, Delta_v, 0, 0, 1;

        // Compute new warp function.
        inv_W_delta = W_delta.inverse();
        W_new = W_old*inv_W_delta;

        // Allocate the new warp vector.
        p_new(0) = W_new(0,2);
        p_new(1) = W_new(1,2);
        p_new(2) = W_new(0,0)-1;
        p_new(3) = W_new(1,0);
        p_new(4) = W_new(0,1);
        p_new(5) = W_new(1,1)-1;

    }
    else if (k == 12){
        // Define variables for a second-order subset.
        int m = 6, n =6;
        MatrixXd W_old(m,n), W_delta(m,n), W_new(m,n), inv_W_delta(m,n);
        double u = p(0), v = p(1), u_x = p(2), v_x = p(3), u_y = p(4), v_y = p(5);
        double u_xx = p(6), v_xx = p(7), u_xy = p(8), v_xy = p(9), u_yy = p(10), v_yy = p(11);
        double Delta_u = Delta_p(0), Delta_v = Delta_p(1), Delta_u_x = Delta_p(2);
        double Delta_v_x = Delta_p(3), Delta_u_y = Delta_p(4), Delta_v_y = Delta_p(5);
        double Delta_u_xx = Delta_p(6), Delta_v_xx = Delta_p(7), Delta_u_xy = Delta_p(8);
        double Delta_v_xy = Delta_p(9), Delta_u_yy = Delta_p(10), Delta_v_yy = Delta_p(11); 
        double S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, S13, S14, S15, S16, S17, S18;
        double Delta_S1, Delta_S2, Delta_S3, Delta_S4, Delta_S5, Delta_S6, Delta_S7, Delta_S8, Delta_S9;
        double Delta_S10, Delta_S11, Delta_S12, Delta_S13, Delta_S14, Delta_S15, Delta_S16, Delta_S17, Delta_S18;

        // Calculate the homogenous warp function terms.
        S1 = (2*u_x) + (u_x*u_x) + (u*u_xx);
        S2 = (2*u*u_xy) + (2*(1+u_x)*u_y);
        S3 = (u_y*u_y) + (u*u_yy);
        S4 = 2*u*(1+u_x);
        S5 = 2*u*u_y;
        S6 = u*u;
        S7 = 0.5*(v*u_xx) + (2*(1+u_x)*v_x) + (u*v_xx);
        S8 = (u_y*v_x) + (u_x*v_y) + (v*u_xy) + (u*v_xy) + v_y + u_x;
        S9 = 0.5*((v*u_yy) + (2*u_y*(1+v_y)) + (u*v_yy));
        S10 = v + (v*u_x) + (u*v_x);
        S11 = u + (v*u_y) + (u*v_y);
        S12 = u*v;
        S13 = (v_x*v_x) + (v*v_xx);
        S14 = (2*v*v_xy) + (2*v_x*(1+v_y));
        S15 = (2*v_y) + (v_y*v_y) + (v*v_yy);
        S16 = 2*v*v_x;
        S17 = 2*v*(1+v_y);
        S18 = v*v;
        
        // Calculate the homogenous warp function increment terms.
        Delta_S1 = (2*Delta_u_x) + (Delta_u_x*Delta_u_x) + (Delta_u*Delta_u_xx);
        Delta_S2 = (2*Delta_u*Delta_u_xy) + (2*(1+Delta_u_x)*Delta_u_y);
        Delta_S3 = (Delta_u_y*Delta_u_y) + (Delta_u*Delta_u_yy);
        Delta_S4 = 2*Delta_u*(1+Delta_u_x);
        Delta_S5 = 2*Delta_u*Delta_u_y;
        Delta_S6 = Delta_u*Delta_u;
        Delta_S7 = 0.5*(Delta_v*Delta_u_xx) + (2*(1+Delta_u_x)*Delta_v_x) + (Delta_u*Delta_v_xx);
        Delta_S8 = (Delta_u_y*Delta_v_x) + (Delta_u_x*Delta_v_y) + (Delta_v*Delta_u_xy) + (Delta_u*Delta_v_xy) + Delta_v_y + Delta_u_x;
        Delta_S9 = 0.5*((Delta_v*Delta_u_yy) + (2*Delta_u_y*(1+Delta_v_y)) + (Delta_u*Delta_v_yy));
        Delta_S10 = Delta_v + (Delta_v*Delta_u_x) + (Delta_u*Delta_v_x);
        Delta_S11 = Delta_u + (Delta_v*Delta_u_y) + (Delta_u*Delta_v_y);
        Delta_S12 = Delta_u*Delta_v;
        Delta_S13 = (Delta_v_x*Delta_v_x) + (Delta_v*Delta_v_xx);
        Delta_S14 = (2*Delta_v*Delta_v_xy) + (2*Delta_v_x*(1+Delta_v_y));
        Delta_S15 = (2*Delta_v_y) + (Delta_v_y*Delta_v_y) + (Delta_v*Delta_v_yy);
        Delta_S16 = 2*Delta_v*Delta_v_x;
        Delta_S17 = 2*Delta_v*(1+Delta_v_y);
        Delta_S18 = Delta_v*Delta_v;
        
        // Assemble homogenous warp function and warp function increment.
        W_old << 1+S1, S2, S3, S4, S5, S6, S7, 1+S8, S9, S10, S11, S12, S13, S14, 1+S15, S16, S17, S18, 0.5*u_xx, u_xy, 0.5*u_yy, 1+u_x, u_y, u, 0.5*v_xx, v_xy, 0.5*v_yy, v_x, 1+v_y, v, 0, 0, 0, 0, 0, 1;
        W_delta << 1+Delta_S1, Delta_S2, Delta_S3, Delta_S4, Delta_S5, Delta_S6, Delta_S7, 1+Delta_S8, Delta_S9, Delta_S10, Delta_S11, Delta_S12, Delta_S13, Delta_S14, 1+Delta_S15, Delta_S16, Delta_S17, Delta_S18, 0.5*Delta_u_xx, Delta_u_xy, 0.5*Delta_u_yy, 1+Delta_u_x, Delta_u_y, Delta_u, 0.5*Delta_v_xx, Delta_v_xy, 0.5*Delta_v_yy, Delta_v_x, 1+Delta_v_y, Delta_v, 0, 0, 0, 0, 0, 1;
        
        // Compute new warp function.
        inv_W_delta = W_delta.inverse();
        W_new = W_old*inv_W_delta;

        // Allocate the new warp vector.
        p_new(0) = W_new(3,5);
        p_new(1) = W_new(4,5);
        p_new(2) = W_new(3,3)-1;
        p_new(3) = W_new(4,3);
        p_new(4) = W_new(3,4);
        p_new(5) = W_new(4,4)-1;
        p_new(6) = W_new(3,0)*2;
        p_new(7) = W_new(4,0)*2;
        p_new(8) = W_new(3,1);
        p_new(9) = W_new(4,1);
        p_new(10) = W_new(3,2)*2;
        p_new(11) = W_new(4,2)*2;
    }
        
    return p_new;
}

double _norm(
    const Ref<const VectorXd> &Delta_p,
    const double &size
    )
{
    // Define variables.
    int n = Delta_p.size();
    double norm;
    
    // Compute norm after Gao et al. (2015) where size is a representative dimension for the subset size.
    if (n <= 7)
    {
        norm = pow((pow(Delta_p(0),2) + pow(Delta_p(1),2) + 
        pow((Delta_p(2)*size),2) + pow((Delta_p(3)*size),2) + 
        pow((Delta_p(4)*size),2) + pow((Delta_p(5)*size),2)), 0.5);
    } 
    else if (n > 7)
    {
        norm = pow((pow(Delta_p(0),2) + pow(Delta_p(1),2) + 
        pow((Delta_p(2)*size),2) + pow((Delta_p(3)*size),2) + 
        pow((Delta_p(4)*size),2) + pow((Delta_p(5)*size),2) +
        pow((0.5*Delta_p(6)*pow(size,2)),2) +
        pow((0.5*Delta_p(7)*pow(size,2)),2) +
        pow((0.5*Delta_p(8)*pow(size,2)),2) +
        pow((0.5*Delta_p(9)*pow(size,2)),2) +
        pow((0.5*Delta_p(10)*pow(size,2)),2) +
        pow((0.5*Delta_p(11)*pow(size,2)),2)), 0.5);
    }
    return norm;
}

double _ZNSSD(
    const Ref<const VectorXd> &f,
    const Ref<const VectorXd> &g,
    const double &f_m,
    const double &g_m,
    const double &Delta_f,
    const double &Delta_g
    )
{
    // Define variables.
    int n = f.rows(), i;
    double znssd = 0;

    // Compute the normalised sum of square differences.
    for(int i = 0; i < n;  i++)
    {
        znssd += pow((((f(i)-f_m)/Delta_f)-((g(i)-g_m)/Delta_g)),2);
    }

    return znssd;
}

std::vector<MatrixXd> _init_reference(
    const Ref<const VectorXd> &f_coord,
    const Ref<const MatrixXd> &template_coords,
    const Ref<const MatrixXd> &f_QCQT
    )
{
    // Define variables.
    int n = template_coords.rows();
    double f_m, Delta_f, SSSIG, sigma;
    VectorXd f(n), quality(2), constants(2);
    MatrixXd f_coords(n,2), grad_f(n,2);

    // Compute reference quantitities.
    f_coords = _f_coords(f_coord, template_coords);
    f = _intensity(f_coords, f_QCQT);
    f_m = f.mean();
    Delta_f = _Delta_f(f, f_m);
    grad_f = _grad(f_coords, f_QCQT);
    SSSIG = _SSSIG(f_coords, grad_f);
    sigma = _sigma_intensity(f, f_m);
    constants << f_m, Delta_f;
    quality << SSSIG, sigma;
    
    // Generate output container.
    std::vector<Eigen::MatrixXd> output;
    output.push_back(f_coords);
    output.push_back(f);
    output.push_back(constants);
    output.push_back(grad_f);
    output.push_back(quality);   

    return output;
}

std::vector<MatrixXd> _solve_ICGN(
    const Ref<const VectorXd> &f_coord,
    const Ref<const MatrixXd> &f_coords,
    const Ref<const VectorXd> &f,
    const double &f_m,
    const double &Delta_f,
    const Ref<const MatrixXd> &grad_f,
    const Ref<const MatrixXd> &f_QCQT,
    const Ref<const MatrixXd> &g_QCQT,
    const Ref<const VectorXd> &p_0,
    const double &max_norm,
    const int &max_iterations
    )
{
    // Define variables.
    int n = f_coords.rows();
    int m = p_0.size();
    int iteration = 1;
    double norm = 1, C_LS, C_ZNCC, size = pow(double(n),0.5);
    MatrixXd sdi(n,m), hessian(m,m);
    VectorXd g_coord(2), p(m), g(n), Delta_p(m), p_new(m), constants(2);
    MatrixXd g_coords(n,2);
    MatrixXd convergence = MatrixXd::Zero(4, max_iterations+1);
    double g_m, Delta_g;

    // Compute reference quantitities.
    sdi = _sdi(f_coord, f_coords, grad_f, p);
    hessian = _hessian(sdi);

    // Iterate to solution.
    p = p_0;
    try{
        while (iteration < max_iterations && norm > max_norm){
            g_coord = _g_coord(f_coord, p);
            g_coords = _g_coords(f_coord, p, f_coords);
            g = _intensity(g_coords, g_QCQT);
            g_m = g.mean();
            Delta_g = _Delta_g(g, g_m);
            Delta_p = _Delta_p_ICGN(hessian, f, g, f_m, g_m, Delta_f, Delta_g, sdi);
            p_new = _p_new_ICGN(p, Delta_p);
            norm = _norm(Delta_p, size);
            C_LS = _ZNSSD(f, g, f_m, g_m, Delta_f, Delta_g);
            C_ZNCC = 1-(C_LS/2);
            convergence(0,iteration-1) = iteration;
            convergence(1,iteration-1) = norm;
            convergence(2,iteration-1) = C_ZNCC;
            convergence(3,iteration-1) = C_LS;
            iteration += 1;
            p = p_new;
        }
    }
    catch(const std::invalid_argument& e){
        std::vector<Eigen::MatrixXd> output;
        return output;
    }
    constants << g_m, Delta_g;

    // Generate output container.
    std::vector<Eigen::MatrixXd> output;
    output.push_back(g_coords);
    output.push_back(g);
    output.push_back(constants);
    output.push_back(convergence);
    output.push_back(p); 

    return output;
}

std::vector<MatrixXd> _solve_FAGN(
    const Ref<const VectorXd> &f_coord,
    const Ref<const MatrixXd> &f_coords,
    const Ref<const VectorXd> &f,
    const double &f_m,
    const double &Delta_f,
    const Ref<const MatrixXd> &grad_f,
    const Ref<const MatrixXd> &f_QCQT,
    const Ref<const MatrixXd> &g_QCQT,
    const Ref<const VectorXd> &p_0,
    const double &max_norm,
    const int &max_iterations
    )
{
    // Define variables.
    int n = f_coords.rows();
    int m = p_0.size();
    int iteration = 1;
    double norm = 1, C_LS, C_ZNCC, size = pow(double(n),0.5);
    MatrixXd sdi(n,m), hessian(m,m);
    VectorXd g_coord(2), p(m), g(n), Delta_p(m), p_new(m), constants(2);
    MatrixXd g_coords(n,2), grad_g(n,2);
    MatrixXd convergence = MatrixXd::Zero(4, max_iterations+1);
    double g_m, Delta_g;

    // Iterate to solution.
    p = p_0;
    try{
        while (iteration < max_iterations && norm > max_norm){
            g_coord = _g_coord(f_coord, p);
            g_coords = _g_coords(f_coord, p, f_coords);
            g = _intensity(g_coords, g_QCQT);
            g_m = g.mean();
            Delta_g = _Delta_g(g, g_m);
            grad_g = _grad(g_coords, g_QCQT);
            sdi = _sdi(g_coord, g_coords, grad_g, p);
            hessian = _hessian(sdi);
            Delta_p = _Delta_p_FAGN(hessian, f, g, f_m, g_m, Delta_f, Delta_g, sdi);
            p_new = p + Delta_p;
            norm = _norm(Delta_p, size);
            C_LS = _ZNSSD(f, g, f_m, g_m, Delta_f, Delta_g);
            C_ZNCC = 1-(C_LS/2);
            convergence(0,iteration-1) = iteration;
            convergence(1,iteration-1) = norm;
            convergence(2,iteration-1) = C_ZNCC;
            convergence(3,iteration-1) = C_LS;
            iteration += 1;
            p = p_new;
        }
    }
    catch(const std::invalid_argument& e){
        std::vector<Eigen::MatrixXd> output;
        return output;
    }
    constants << g_m, Delta_g;

    // Generate output container.
    std::vector<Eigen::MatrixXd> output;
    output.push_back(g_coords);
    output.push_back(g);
    output.push_back(constants);
    output.push_back(convergence);
    output.push_back(p); 

    return output;
}

// ----------------
// Python interface
// ----------------

namespace py = pybind11;

PYBIND11_MODULE(_subset_extensions,m)
{

py::register_exception<invalid_argument>(m,"ValueError");
m.doc() = "C++ subset module extensions for geopyv.";
m.def("_init_reference", &_init_reference, py::return_value_policy::reference_internal, "C++ extension to initialise the reference subset.");
m.def("_solve_ICGN", &_solve_ICGN, py::return_value_policy::reference_internal, "C++ extension to solve the subset deformation using the ICGN method.");
m.def("_solve_FAGN", &_solve_FAGN, py::return_value_policy::reference_internal, "C++ extension to solve the subset deformation using the FAGN method.");
}