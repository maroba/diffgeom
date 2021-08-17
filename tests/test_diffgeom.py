import unittest
import sympy as sp
from diffgeom import Manifold, Tensor


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

    def test_euclidean_plane_in_cartesian_coords_has_vanishing_christoffel(self):
        # Arrange
        coords = sp.symbols('x, y')
        metric = sp.diag(1, 1)
        # Act
        plane = Manifold(metric, coords)
        # Assert
        self.assertEqual(len(plane.gammas), 0)

    def test_euclidean_plane_in_polar_coords_has_nonvanishing_christoffel(self):
        # Arrange
        r, phi = sp.symbols('r, phi')
        metric = sp.diag(1, r**2)
        # Act
        plane = Manifold(metric, coords=(r, phi)) # zeroth coord == r, first coord == phi
        # Assert
        self.assertEqual(len(plane.gammas), 3)
        self.assertEqual(plane.gammas[0, 1, 1], -r)
        self.assertEqual(plane.gammas[1, 0, 1], 1/r)
        self.assertEqual(plane.gammas[1, 1, 0], 1 / r)


class TestTensor(unittest.TestCase):

    def test_init_tensor_with_values_stores_values(self):
        # Arrange
        r, phi = sp.symbols('r, phi')
        metric = sp.diag(1, r**2)
        plane = Manifold(metric, coords=(r, phi))

        # Act
        A = Tensor(plane, 'ulu', {(0, 1, 0): r**4, (1, 0, 1): 1/r**2})

        # Assert
        self.assertEqual(len(A), 2)
        self.assertEqual(A[0, 1, 0], r**4)
        self.assertEqual(A[1, 0, 1], 1 / r**2)

    def test_set_tensor_component_stores_value(self):
        # Arrange
        r, phi = sp.symbols('r, phi')
        metric = sp.diag(1, r**2)
        plane = Manifold(metric, coords=(r, phi))
        A = Tensor(plane, 'ulu', {(0, 1, 0): r ** 4, (1, 0, 1): 1 / r ** 2})

        # Act
        A[1, 1, 1] = r**3

        # Assert
        self.assertEqual(A[1, 1, 1], r**3)

    def test_init_tensor_without_values_is_zero_tensor(self):

        # Arrange
        r, phi = sp.symbols('r, phi')
        metric = sp.diag(1, r ** 2)
        plane = Manifold(metric, coords=(r, phi))
        # Act
        A = Tensor(plane, 'ulu')

        # Assert
        self.assertEqual(len(A), 0)
