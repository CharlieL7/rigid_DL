"""
Unit tests for the geometric functions module
"""

import unittest
import numpy as np
import rigid_DL.geometric as geo

class TestGeometric(unittest.TestCase):

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
        self.assertTrue(np.allclose(pos, np.array([0.4, 0.2, 0.])))


    def test_calc_abg(self):
        nodes = np.array([
            [0., 1., 0., 0.5, 0.5, 0.],
            [0., 0., 1., 0., 0.5, 0.5],
            [0., 0., 0., 0., 0., 0.]
        ])
        (a, b, g) = geo.calc_abg(nodes)
        self.assertAlmostEqual(a, 1/2.)
        self.assertAlmostEqual(b, 1/2.)
        self.assertAlmostEqual(g, 1/2.)


    def test_pos_quadratic(self):
        # flat tri
        nodes = np.array([
            [0., 1., 0., 0.5, 0.5, 0.],
            [0., 0., 1., 0., 0.5, 0.5],
            [0., 0., 0., 0., 0., 0.]
        ])
        pos = geo.pos_quadratic(0.2, 0.73, nodes)
        self.assertTrue(np.allclose(pos, np.array([0.2, 0.73, 0.])))


        # curved tri
        nodes = np.array([
            [0., 1., 0., 0.5, 0.5, 0.],
            [0., 0., 1., 0., 0.5, 0.5],
            [0., 0., 0., 0.3, 0.3, 0.]
        ])
        pos = geo.pos_quadratic(1/3., 1/3., nodes)
        self.assertTrue(np.allclose(pos, np.array([0.33333333, 0.33333333, 0.26666667])))


    def test_dphi_dxi_quadratic(self):
        # flat tri
        nodes = np.array([
            [0., 1., 0., 0.5, 0.5, 0.],
            [0., 0., 1., 0., 0.5, 0.5],
            [0., 0., 0., 0., 0., 0.]
        ])
        dphi_dxi = geo.dphi_dxi_quadratic(1/3., 1/3., nodes)
        self.assertTrue(
            np.allclose(
                dphi_dxi,
                np.array([-1/3, 1/3., 0., 0., 4/3., -4/3])
            )
        )


    def test_dphi_deta_quadratic(self):
        # flat tri
        nodes = np.array([
            [0., 1., 0., 0.5, 0.5, 0.],
            [0., 0., 1., 0., 0.5, 0.5],
            [0., 0., 0., 0., 0., 0.]
        ])
        dphi_deta = geo.dphi_deta_quadratic(1/3., 1/3., nodes)
        self.assertTrue(
            np.allclose(
                dphi_deta,
                np.array([-1/3., 0., 1/3., -4/3., 4/3., 0.])
            )
        )


    def test_stresslet(self):
        x = np.array([1., 0., 0.])
        x_0 = np.array([0., 0., 0.])
        T = geo.stresslet(x, x_0)
        self.assertTrue(
            np.allclose(
                T,
                np.array(
                    [[[-6, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0]],

                     [[0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0]],

                     [[0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0]]]
                )
            )
        )

        x = np.array([0.2, -1., 0.5])
        T = geo.stresslet(x, x_0)
        self.assertTrue(
            np.allclose(
                T,
                np.array(
                    [[[-0.0253961, 0.12698048, -0.06349024],
                      [0.12698048, -0.6349024, 0.3174512],
                      [-0.06349024, 0.3174512, -0.1587256]],

                     [[0.12698048, -0.6349024, 0.3174512],
                      [-0.6349024, 3.17451201, -1.58725601],
                      [0.3174512, -1.58725601, 0.793628]],

                     [[-0.06349024, 0.3174512, -0.1587256],
                      [0.3174512, -1.58725601, 0.793628],
                      [-0.1587256, 0.793628, -0.396814]]]
                )
            )
        )


    def test_stresslet_n(self):
        x = np.array([1., 0., 0.])
        x_0 = np.array([0., 0., 0.])
        n = np.array([0., 1., 0.])
        S = geo.stresslet_n(x, x_0, n)
        self.assertTrue(
            np.allclose(
                S,
                np.zeros((3, 3))
            )
        )

        x = np.array([.3, 0.1, -1])
        x_0 = np.array([0., 0., 0.])
        n = np.array([0., 1., 0.])
        S = geo.stresslet_n(x, x_0, n)
        self.assertTrue(
            np.allclose(
                S,
                np.array(
                    [[-0.04255122, -0.01418374, 0.14183741],
                     [-0.01418374, -0.00472791, 0.04727914],
                     [0.14183741, 0.04727914, -0.47279137]]
                )
            )
        )
