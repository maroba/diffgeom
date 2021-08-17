import unittest
import sympy as sp
from diffgeom import Manifold


class TestManifold(unittest.TestCase):

    def test_init_manifold_stores_coords_and_metric(self):
        # Arrange
        coords = sp.symbols('x, y')
        metric = sp.diag(1, 1)
        # Act
        manifold = Manifold(metric, coords)
        # Assert
        self.assertEqual(manifold.coords, coords)
        self.assertEqual(manifold.metric, metric)
