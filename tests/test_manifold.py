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
        self.assertEqual(manifold.metric[0, 0], metric[0, 0])
        self.assertEqual(manifold.metric[1, 1], metric[1, 1])

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

    def test_transform_cartesian_to_polar_changes_coords_and_metric(self):

        # Arrange
        x, y = sp.symbols('x, y')
        r, phi = sp.symbols('r, phi')
        plane_cartesian = Manifold(metric=sp.diag(1, 1), coords=(x, y))

        # Act
        plane_polar = plane_cartesian.transform((r, phi), {x: r*sp.cos(phi), y: r*sp.sin(phi)})

        # Assert
        assert plane_polar.coords == (r, phi)
        assert len(plane_polar.metric) == 2
        assert plane_polar.metric[0, 0] == 1
        assert plane_polar.metric[1, 1] == r ** 2
        assert plane_polar.metric[r, r] == 1
        assert plane_polar.metric[phi, phi] == r ** 2

    def test_geodesic_equations_plane_cartesian(self):
        # Arrange
        x, y = sp.symbols('x, y')
        plane = Manifold(metric=sp.diag(1, 1), coords=(x, y))

        # Act
        eqs, s = plane.geodesic_equations()

        # Assert
        assert eqs[0] == sp.Eq(sp.Derivative(x, s, s), 0)
        assert eqs[1] == sp.Eq(sp.Derivative(y, s, s), 0)

    def test_geodesic_equations_plane_polar(self):
        # Arrange
        r, phi = sp.symbols('r, phi')
        plane = Manifold(metric=sp.diag(1, r**2), coords=(r, phi))

        # Act
        eqs, s = plane.geodesic_equations()

        # Assert
        assert eqs[0] == sp.Eq(sp.Derivative(r, s, s), -r*sp.Derivative(phi, s)**2)
        assert eqs[1] == sp.Eq(sp.Derivative(phi, s, s), 2*sp.Derivative(r, s)*sp.Derivative(phi,s)/r)
