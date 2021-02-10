/*
 * C version of the matrix assembly functions for speed.
 * Used with ctypes to call C code from Python
 */

#include <cmath>
#include <iostream>
#include "Eigen/Dense"

using Eigen::Matrix;
using Eigen::Map;
using Eigen::Dynamic;


double* add_cp_le_DL_terms(double* K, double* nodes, double* verts, int* faces, int num_faces, double* normals, double* hs_arr)
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
	double c_0 = 1. / (4. * M_PI);
	Map<Matrix<double, Dynamic, Dynamic>> K_map(K, 3*num_faces, 3*num_faces); // using map here to not have double copy
	Matrix<double, Dynamic, 3> _nodes(nodes, num_faces); // does this copy the data or just map?
	Matrix<double, Dynamic, 3> _verts(verts, num_faces);
	Matrix<int, Dynamic, 3> _faces(faces, num_faces);
	Matrix<double, Dynamic, 3> _normals(normals, num_faces);
	Matrix<double, Dynamic, 1> _hs_arr(hs_arr, num_faces);
	for (int face_num = 0; face_num < num_faces; ++face_num)
	{
		Matrix<double, 3, 3> face_nodes = get_lin_tri_nodes(_verts, _faces, face_num);
		Matrix<double, 3, 1> face_n = _normals.block<3, 1>(face_num, 0);
		double face_hs = _hs_arr(face_num);
		for (int src_num = 0; src_num < num_faces; ++src_num)
		{
			Matrix<double, 3, 1> node = _nodes.block<3, 1>(src_num, 0);
			if (face_num != src_num)
			{
				Matric<double, 3, 3> sub_mat = int_over_tri_lin(); // TODO
				K_map.block<3, 3>(3*src_num, 3*face_num) += c_0 * sub_mat;
				K_map.block<3, 3>(3*src_num, 3*src_num) -= c_0 * sub_mat;
			} //do nothing if face_num == src_num
		}
	}
	for (int src_num = 0; src_num < num_faces; ++src_num)
	{
		K_map.block<3, 3>(3*src_num, 3*src_num) += c_0 * -4 * M_PI * Matrix<double, 3, 3>::Identity();
	}
}

Matrix<double, 3, 3> get_lin_tri_nodes(Matrix<double, Dynamic, 3> verts, Matrix<int, Dynamic, 3> faces, int face_num)
{
	//TODO
}
