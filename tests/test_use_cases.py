import unittest
import sympy as s
from diffgeom import *


class TestUseCases(unittest.TestCase):

    def test_radial_fall_to_black_hole(self):
        coords = t, r, theta, phi = s.symbols('t, r, theta, phi')
        M = s.symbols('M')
        metric = s.diag(-(1+2*M/r), 1/(1+2*M/r), r**2, r**2*s.sin(theta)**2)
        spacetime = Manifold(metric, coords)

        eqs, tau = spacetime.geodesic_equations()
        pass

    def test_from_robertson_walker_to_einstein_eqs(self):
        t, r, theta, phi, k = s.symbols('t, r, theta, phi, k')
        R, rho, p = s.symbols('R rho p', cls=s.Function)
        g = s.diag(-1, R(t)**2 / (1-k*r), R(t)**2, R(t)**2 * s.sin(theta)**2)
        universe = Manifold(g, (t, r, theta, phi))

        u = Tensor(universe, 'u', {0: 1})
        T = (rho(t) + p(t)) * u * u + p(t) * universe.metric_inv
        conservation_eq = TensorEquation(T.diff().contract(1, 2), 0)
        conservation_eq.lhs.simplify()

        pass



