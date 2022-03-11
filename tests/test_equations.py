import unittest

import sympy as sp

from diffgeom.equations import ODESystem


class TestODESystem(unittest.TestCase):

    def test_find_order_2(self):

        # Arrange
        f, t = sp.symbols('f, t')
        eqs = [sp.Eq(sp.Derivative(f, t), sp.Derivative(f, t, t) + 4)]

        # Act
        odes = ODESystem(eqs)

        # Assert
        assert odes.order == 2


