"""
Unit tests for the gaussian quadrature module
"""

import unittest
import numpy as np
import gauss_quad as gq

class TestGaussQuadrature(unittest.TestCase):

    def test_lin_tri(self):
        def in_func(xi, eta, nodes):
            return 1.
        nodes = np.identity(3)
        hs = 1.
        ret = gq.int_over_tri_lin(in_func, nodes, hs)
        self.assertEqual(ret, 0.5)
