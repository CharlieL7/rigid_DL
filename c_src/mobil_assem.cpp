/*
 * C version of the matrix assembly functions for speed.
 * Used with ctypes to call C code from Python
 */

#include <cmath>
#include <iostream>
#include <Eigen/Dense>

double * add_cp_le_DL_terms(double* K, double* nodes, int* faces, double* normals, double* hs_arr)
{
	/*
	 * Makes the double layer terms for a constant potential, linear elements mesh.
	 * This function needs all of the raw data arrays because passing a python object
	 * would not work
	 * Parameters:
	 * 	K: stiffness matrix; (3 * face_num, 3 * face_num) array
	 * 	nodes: mesh nodes, triangle centers; (face_num, 3) array
	 * 	faces: mesh faces, the three nodes of a face; (face_num ,3) array
	 *	normals: mesh face normal vectors; (face_num ,3) array
	 *	hs_arr: square mesh areas, triangle area would be 0.5 of these; (face_num,) array
	 */
	double c_0 = 1. / (4. * M_PI);


