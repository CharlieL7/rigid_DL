/*
 * C++ version of the matrix assembly functions for speed.
 * Used with ctypes to call C++ code from Python
 */

#include <cmath>
#include <iostream>
#include <functional>
#include "Eigen/StdVector"
#include "matrix_assem.hpp"

using Eigen::RowMajor;

extern "C"
{
	void add_cp_le_DL_terms(double* K, double* nodes, double* verts, int* faces, int num_faces, double* normals, double* hs_arr)
	{
		/*
		 * Makes the double layer terms for a constant potential, linear elements mesh.
		 * This function needs all of the raw data arrays because passing a python object
		 * would not work
		 * Parameters:
		 * 	K: stiffness matrix; (3 * num_faces, 3 * num_faces) array
		 * 	nodes: mesh nodes, triangle centers; (face_num, 3) array
		 * 	verts: mesh verticies; (vert_num, 3) array
		 * 	faces: mesh faces, the three nodes of a face; (face_num ,3) array
		 *	normals: mesh face normal vectors; (face_num ,3) array
		 *	hs_arr: square mesh areas, triangle area would be 0.5 of these; (face_num,) array
		 */
		const double c_0 = 1. / (4. * M_PI);
		Map<Matrix<double, Dynamic, Dynamic>, RowMajor> K_map(K, 3*num_faces, 3*num_faces); // using map here to not have double copy
		// consider transposing all of these to use column major representation
		Map<Matrix<double, Dynamic, 3, RowMajor>> _nodes(nodes, num_faces, 3);
		Map<Matrix<double, Dynamic, 3, RowMajor>> _verts(verts, num_faces, 3);
		Map<Matrix<int, Dynamic, 3, RowMajor>> _faces(faces, num_faces, 3);
		Map<Matrix<double, Dynamic, 3, RowMajor>> _normals(normals, num_faces, 3);
		Map<Matrix<double, Dynamic, 1>> _hs_arr(hs_arr, num_faces, 1);
		for (int face_num = 0; face_num < num_faces; ++face_num)
		{
			const Mat33 face_nodes = get_lin_tri_nodes(_verts, _faces, face_num);
			const Mat13 face_n = _normals.row(face_num);
			const double face_hs = _hs_arr(face_num);
			for (int src_num = 0; src_num < num_faces; ++src_num)
			{
				const Mat13 node = _nodes.row(src_num);
				if (face_num != src_num)
				{
					Mat33 sub_mat = int_over_tri_lin(&cp_le_DL_integrand, face_nodes, face_n, node, face_hs);
					K_map.block<3, 3>(3*src_num, 3*face_num) += c_0 * sub_mat;
					K_map.block<3, 3>(3*src_num, 3*src_num) -= c_0 * sub_mat;
				} //do nothing if face_num == src_num
			}
		}
		for (int src_num = 0; src_num < num_faces; ++src_num)
		{
			K_map.block<3, 3>(3*src_num, 3*src_num) += c_0 * -4 * M_PI * Mat33::Identity();
		}
	}
}

Mat33 get_lin_tri_nodes(Matrix<double, Dynamic, 3> verts, Matrix<int, Dynamic, 3> faces, int face_num)
{
	/*
	 * Gets the positions of the triangle nodes at face_num
	 * Returns:
	 * 	Triangle nodes as 3x3 matrix with rows as node positions
	 */
	Mat33 nodes;
	Matrix<int, 1, 3> face = faces.row(face_num);
	for (int i = 0; i < 3; ++i)
	{
		nodes.row(i) = verts.row(face(i));
	}
	return nodes;
}

// Not sure if this can be done with templates rather than overloading
// would somehow need to get the return type of a template function pointer
double int_over_tri_lin(func_scalar func, Mat33 nodes, double hs)
{
	/*
	 * Gaussian quadrature over a linear triangle for scalar values
	 * Parameters:
	 * 	func: double returning function that takes (xi, eta, nodes)
	 * 	nodes: 3 triangle nodes as row vectors
	 * 	hs: triangle area
	 * Returns:
	 * 	Integrated double value
	 */ 
	Matrix<double, NUM_QUAD_PTS, 1> f;
	for (int i = 0; i < PARA_PTS.rows(); ++i)
	{
		double xi = PARA_PTS(i, 0);
		double eta = PARA_PTS(i, 1);
		f(i) = func(xi, eta, nodes);
	}
	double ret = 0.5 * hs * f.transpose() * W;
	return ret;
}

Mat13 int_over_tri_lin(func_1d func, Mat33 nodes, double hs)
{
	/*
	 * Gaussian quadrature over a linear triangle for vector values
	 * Parameters:
	 * 	func: vector returning function that takes (xi, eta, nodes)
	 * 	nodes: 3 triangle nodes as row vectors
	 * 	hs: triangle area
	 * Returns:
	 * 	Integrated vector value
	 */ 
	Matrix<double, NUM_QUAD_PTS, 3> f;
	for (int i = 0; i < PARA_PTS.rows(); ++i)
	{
		double xi = PARA_PTS(i, 0);
		double eta = PARA_PTS(i, 1);
		f.row(i) = func(xi, eta, nodes);
	}
	Mat13 ret = 0.5 * hs * f.transpose() * W;
	return ret;
}


Mat33 int_over_tri_lin(func_2d func, Mat33 nodes, double hs)
{
	/*
	 * Gaussian quadrature over a linear triangle for 2d matrix values
	 * Parameters:
	 * 	func: 2d matrix returning function that takes (xi, eta, nodes)
	 * 	nodes: 3 triangle nodes as column vectors
	 * 	hs: triangle area
	 * Returns:
	 * 	Integrated 2d matrix value
	 */ 
	// Need to make a vector of 2D matrices
	std::vector<Mat33, Eigen::aligned_allocator<Mat33>> f;
	for (int i = 0; i < PARA_PTS.rows(); ++i)
	{
		double xi = PARA_PTS(i, 0);
		double eta = PARA_PTS(i, 1);
		f.push_back(func(xi, eta, nodes));
	}
	Mat33 ret = Mat33::Zero();
	for (unsigned int i = 0; i < f.size(); ++i)
	{
		ret += f.at(i) * W(i);
	}
	ret *= 0.5 * hs;
	return ret;
}


typedef std::function<Mat33(double , double , Mat33 , Matrix<double, 1, 3>, Matrix<double, 1, 3>)> cp_le_quad_func;
Mat33 int_over_tri_lin(cp_le_quad_func func, Mat33 nodes, Matrix<double, 1, 3> n, Matrix<double, 1, 3> x_0, double hs)
{
	/*
	 * Gaussian quadrature over a linear triangle for 2d matrix values
	 * Parameters:
	 * documentation TODO
	 * Returns:
	 */ 
	// Need to make a vector of 2D matrices
	std::vector<Mat33, Eigen::aligned_allocator<Mat33>> f;
	for (int i = 0; i < PARA_PTS.rows(); ++i)
	{
		double xi = PARA_PTS(i, 0);
		double eta = PARA_PTS(i, 1);
		f.push_back(func(xi, eta, nodes, n, x_0));
	}
	Mat33 ret = Mat33::Zero();
	for (unsigned int i = 0; i < f.size(); ++i)
	{
		ret += f.at(i) * W(i);
	}
	ret *= 0.5 * hs;
	return ret;
}


Mat33 cp_le_DL_integrand(double xi, double eta, Mat33 nodes, Mat13 n, Mat13 x_0)
{
	Mat13 x = pos_linear(xi, eta, nodes);
	return stresslet_n(x, x_0, n);
}


Mat33 stresslet_n(Mat13 x, Mat13 x_0, Mat13 n)
{
	Mat13 x_hat = x - x_0;
	double r = x_hat.norm();
	Mat33 S_ij = -6. * (x_hat.transpose() * x_hat) * (x_hat * n.transpose()) / (pow(r, 5));
	return S_ij;
}


Mat13 pos_linear(double xi, double eta, Mat33 nodes)
{
    Mat13 x = (1. - xi - eta) * nodes.row(0) + xi * nodes.row(1) + eta * nodes.row(2);
    return x;
}
