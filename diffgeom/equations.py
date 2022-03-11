from sympy import preorder_traversal
import sympy


class TensorEquation(object):

    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs


class ODESystem(object):

    def __init__(self, sympy_eqs):
        self.sympy_eqs = sympy_eqs

    @property
    def order(self):
        max_derivs = []
        for eq in self.sympy_eqs:
            max_deriv = 0
            for arg in preorder_traversal(eq):
                if arg.func == sympy.Derivative:
                    axis = arg.args[1][0]
                    order = arg.args[1][1]
                    if order > max_deriv:
                        max_deriv = order
            max_derivs.append(max_deriv)

        return max(max_derivs)


    def _highest_deriv(self, eq):
        pass
