import sympy as sp

from diffgeom.indexed import Christoffel
from diffgeom import Tensor


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
                                   })

    @property
    def metric_inv(self):
        """
        Returns the inverse of the metric tensor of the manifold.

        Returns
        -------
            diffgeom.Tensor of rank 2 (upper)
        """
        inv = self._metric.inv()
        return Tensor(self, 'll', {(i, j): inv[i, j] for i in range(self.dims)
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
