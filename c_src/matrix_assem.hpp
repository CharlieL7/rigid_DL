#ifndef MATRIX_ASSEMBLY_HPP
#define MATRIX_ASSEMBLY_HPP

#include "Eigen/Dense"

using Eigen::Matrix;
using Eigen::Map;
using Eigen::Dynamic;
using Eigen::RowMajor;

typedef Matrix<double, 3, 3, RowMajor> Mat33;
typedef Matrix<double, 1, 3> Mat13;
typedef Matrix<double, 3, 1> Mat31;

// 6-point quadrature
const double a = 0.816847572980459;
const double b = 0.445948490915965;
const double c = 0.108103018168070;
const double d = 0.091576213509771;
const double omega1 = 0.109951743655322;
const double omega2 = 0.223381589678011;
const Matrix<double, 6 ,2, RowMajor> PARA_PTS = (
	Matrix<double, 6, 2>() << d, d, a, d, d, a, b, b, c, b, b, c
).finished();
const Matrix<double, 6, 1> W = (
	Matrix<double, 6, 1>() << omega1, omega1, omega1, omega2, omega2, omega2
).finished();
const int NUM_QUAD_PTS = 6;

Mat33 get_lin_tri_nodes(Matrix<double, Dynamic, 3> verts, Matrix<int, Dynamic, 3> faces, int face_num);

extern "C"
{
	void add_cp_le_DL_terms(double* K, double* nodes, double* verts, int* faces, int num_faces, double* normals, double* hs_arr);
}

Mat33 get_lin_tri_nodes(Matrix<double, Dynamic, 3> verts, Matrix<int, Dynamic, 3> faces, int face_num);

typedef std::function<double(double xi, double eta, Mat33 nodes)> func_scalar;
double int_over_tri_lin(func_scalar func, Mat33 nodes, double hs);

typedef std::function<Mat13(double xi, double eta, Mat33 nodes)> func_1d;
Mat13 int_over_tri_lin(func_1d func, Mat33 nodes, double hs);

typedef std::function<Mat33(double xi, double eta, Mat33 nodes)> func_2d;
Mat33 int_over_tri_lin(func_2d func, Mat33 nodes, double hs);

typedef std::function<Mat33(double xi, double eta, Mat33 nodes, Mat13 n, Mat13 x_0)> cp_le_quad_func;
Mat33 int_over_tri_lin(cp_le_quad_func func, Mat33 nodes, Mat13 n, Mat13 x_0, double hs);

Mat33 cp_le_DL_integrand(double xi, double eta, Mat33 nodes, Mat13 n, Mat13 x_0);

Mat33 stresslet_n(Mat13 x, Mat13 x_0, Mat13 n);

Mat13 pos_linear(double xi, double eta, Mat33 nodes);
#endif
