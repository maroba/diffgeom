from abc import ABC

import sympy as sp

from diffgeom.indexed import Christoffel
from diffgeom import Tensor, Vector, OneForm


class Manifold(object):
    """
    Class representing a general (pseudo)Riemannian manifold with given coordinate system
    """

    def __init__(self, metric, coords):
        """

        Parameters
        ----------
        metric: sympy matrix
            matrix with the metric coefficients in given coordinate system

        coords: tuple of sympy symbols
            the coordinates

        Examples
        --------

        >>> import sympy
        >>> x, y = sympy.symbols('x, y')
        >>> plane = Manifold(sympy.diag(1, 1), (x, y))

        """
        self._metric = metric
        self.coords = coords
        self.gammas = Christoffel(coords, metric)

    @property
    def metric(self):
        """
        Returns the metric tensor of the manifold.

        Returns
        -------
            diffgeom.Tensor of rank 2 (lower)
        """
        return Tensor(self, 'll', {(i, j): self._metric[i, j] for i in range(self.dims)
                                                              for j in range(self.dims)
                                   }, latex_head='g')

    @property
    def metric_inv(self):
        """
        Returns the inverse of the metric tensor of the manifold.

        Returns
        -------
            diffgeom.Tensor of rank 2 (upper)
        """
        inv = self._metric.inv()
        return Tensor(self, 'uu', {(i, j): inv[i, j] for i in range(self.dims)
                                   for j in range(self.dims)
                                   })

    @property
    def dims(self):
        """
        Returns the number of space dimensions

        Returns
        -------
            int
        """
        return len(self.coords)

    def transform(self, new_coords, trans):
        """
        Performs coordinate transformation of the given coordinate system.

        Parameters
        ----------
        new_coords: tuple of sympy symbols
            the new coordinates

        trans: dict
            The coordinate transformation given by the OLD coordinates in terms of the NEW ones.
            Keys: the old coordinates
            Values: the old coordinates in terms of the new ones

        Returns
        -------
            a new Manifold object with the new coordinate system
        """
        jacobian = []
        for chi in trans.values():
            row = []
            for x in new_coords:
                row.append(sp.diff(chi, x))
            jacobian.append(row)
        jacobian = sp.simplify(sp.Matrix(jacobian).inv())
        new_metric = sp.simplify(jacobian * self._metric.subs(trans) * jacobian.transpose()).inv()

        return Manifold(metric=new_metric, coords=tuple(new_coords))

    def geodesic_equations(self, parameter=r'\tau'):
        eqs = []
        s = sp.symbols(parameter)
        if s in self.coords:
            raise Exception('Name collision between curve parameter s and coordinate name s')
        for i, xi in enumerate(self.coords):
            lhs = sp.Derivative(xi, s, s)
            rhs = 0
            for j, xj in enumerate(self.coords):
                for k, xk in enumerate(self.coords):
                    rhs += self.gammas[i, j, k] * sp.Derivative(xj, s) * sp.Derivative(xk, s)
            eqs.append(sp.Eq(lhs, rhs))
        return eqs, s

    def four_velocity(self, name='U'):
        tau = sp.symbols(r'\tau')
        return Vector(self, {
            k: sp.Derivative(x, tau) for k, x in enumerate(self.coords)
        }, latex_head=name)

    def parallel_transport_equations(self):
        raise NotImplementedError


class Sphere(Manifold):
    """
    Two-dimensional sphere in spherical coordinates.
    """

    def __init__(self, unit=False):
        r, ph, th = sp.symbols('r, phi, theta')
        if unit:
            r = 1
        super(Sphere, self).__init__(sp.diag(r ** 2, r ** 2 * sp.sin(th) ** 2), (th, ph))


class Minkowski(Manifold):
    """
    Flat spacetime in cartesian coordinates.
    """

    def __init__(self):
        t, x, y, z = sp.symbols('t, x, y, z')
        super(Minkowski, self).__init__(sp.diag(-1, 1, 1, 1), (t, x, y, z))


class Schwarzschild(Manifold):
    """
    Spacetime with Schwarzschild metric.
    """

    def __init__(self):
        coords = t, r, theta, phi = sp.symbols('t, r, theta, phi')
        M = sp.symbols('M')
        super(Schwarzschild, self).__init__(sp.diag(-(1+2*M/r),
                                                    1/(1+2*M/r),
                                                    r**2,
                                                    r**2*sp.sin(theta)**2), coords)

#
# TODO: Kruskal-Szekeres, Kerr, Robertson-Walker
#
