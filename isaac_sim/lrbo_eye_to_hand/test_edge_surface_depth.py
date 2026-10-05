import unittest
import numpy as np
from edge_surface_depth import fit_edge_surface_depth


class SurfaceDepthTests(unittest.TestCase):
    def setUp(self):
        self.v, self.u = np.indices((100, 120))
        self.depth = 1 / (1.3 + .001 * self.u + .004 * self.v)
        self.mask = (self.v >= 43) & (self.v <= 50) & (self.u >= 18) & (self.u <= 102)
        self.edge = [[20, 40], [100, 40]]

    def test_original_ray_not_inward_depth(self):
        ray, info = fit_edge_surface_depth(self.depth, self.mask, *self.edge)
        expected = 1 / (1.3 + .001 * np.array([20,100]) + .004 * 40)
        np.testing.assert_allclose(ray(self.edge), expected, atol=1e-10)
        self.assertGreater(abs(ray(self.edge)[0]-self.depth[47,20]), .01)
        self.assertLess(info['surface_rms_m'], 1e-10)
        self.assertNotEqual(*ray(self.edge))

    def test_noise_and_outliers(self):
        rng = np.random.default_rng(42)
        depth = self.depth + rng.normal(0, .0003, self.depth.shape)
        outliers = self.mask & (rng.random(self.depth.shape) < .15)
        depth[outliers] += .04
        ray, info = fit_edge_surface_depth(depth, self.mask, *self.edge)
        np.testing.assert_allclose(ray(self.edge), self.depth[40,[20,100]], atol=.001)
        self.assertGreater(info['surface_inlier_ratio'], .75)

    def test_inconsistent_surface_rejected(self):
        depth = self.depth.copy()
        depth[self.mask] += np.random.default_rng(1).uniform(-.06,.06,np.sum(self.mask))
        with self.assertRaises(ValueError):
            fit_edge_surface_depth(depth,self.mask,*self.edge)

    def test_missing_endpoint_support_rejected(self):
        mask = self.mask & (self.u > 45) & (self.u < 75)
        with self.assertRaises(ValueError):
            fit_edge_surface_depth(self.depth,mask,*self.edge)

    def test_invalid_depth_rejected(self):
        with self.assertRaises(ValueError):
            fit_edge_surface_depth(np.full_like(self.depth,np.nan),self.mask,*self.edge)


if __name__ == '__main__':
    unittest.main()
