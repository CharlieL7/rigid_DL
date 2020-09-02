"""
Unit tests for the gaussian quadrature module
"""

import unittest
import numpy as np
import rigid_DL.geometric as geo
import rigid_DL.gauss_quad as gq

class TestGeometrc(unittest.TestCase):

    def test_pos_linear(self):
        nodes = np.identity(3)
        pos = geo.pos_linear(0., 0., nodes)
        self.assertTrue(np.array_equal(pos, np.array([1., 0., 0.])))
        nodes = np.array([
            [0., 1., 0.,],
            [0., 0., 1.,],
            [0., 0., 0.,]
        ])
        pos = geo.pos_linear(0.4, 0.2, nodes)
        self.assertTrue(np.array_equal(pos, np.array([0.4, 0.2, 0.])))
    
    """
    def test_pos_quadratic(self):

    def test_dphi_dxi_quadratic(self):

    def test_dphi_deta_quadratic(self):

    def test_calc_abg(self):

    def test_stresslet(self):

    def test_stresslet_n(self):

    def test_inertia_func_linear(self):

    def test_inertia_func_quadratic(self):

    def test_shape_func_linear(self):
    """

class TestGaussQuadrature(unittest.TestCase):

    def test_lin_ele_no_func(self):
        def in_func(xi, eta, nodes):
            return 1.
        nodes = np.identity(3)
        hs = 1.
        ret = gq.int_over_tri_lin(in_func, nodes, hs)
        self.assertEqual(ret, 0.5)

    def test_lin_ele_cons_func(self):
        nodes = np.identity(3)
        hs = 1.
        ret = gq.int_over_tri_lin(geo.pos_linear, nodes, hs)
        print(ret)
        self.assertTrue(np.array_equal(ret, np.array([1/8, 1/8, 1/8])))
