"""
Unit tests for the gaussian quadrature module
"""

import unittest
import numpy as np
import rigid_DL.geometric as geo
import rigid_DL.gauss_quad as gq

class TestGaussQuadrature(unittest.TestCase):

    def test_lin_ele_cons_func(self):
        def in_func(xi, eta, nodes):
            return 1.
        nodes = np.identity(3)
        hs = 1.
        ret = gq.int_over_tri_lin(in_func, nodes, hs)
        self.assertEqual(ret, 0.5)

    def test_lin_ele_lin_func(self):
        nodes = np.array([
            [0., 1., 0.,],
            [0., 0., 1.,],
            [0., 0., 0.,]
        ])
        hs = 1.0
        ret = gq.int_over_tri_lin(geo.pos_linear, nodes, hs)
        self.assertTrue(np.allclose(ret, np.array([1/6, 1/6, 0.])))
