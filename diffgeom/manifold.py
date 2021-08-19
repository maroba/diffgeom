import sympy as sp

from diffgeom.indexed import Christoffel
from diffgeom import Tensor


class Manifold(object):

    def __init__(self, metric, coords):
        self._metric = metric
        self.coords = coords
        self.gammas = Christoffel(coords, metric)

    @property
    def metric(self):
        return Tensor(self, 'll', {(i, j): self._metric[i, j] for i in range(self.dims)
                                                              for j in range(self.dims)
                                   })

    @property
    def metric_inv(self):
        inv = self._metric.inv()
        return Tensor(self, 'll', {(i, j): inv[i, j] for i in range(self.dims)
                                   for j in range(self.dims)
                                   })

    @property
    def dims(self):
        return len(self.coords)

    def transform(self, new_coords, trans):
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

    def __init__(self, unit=False):
        r, ph, th = sp.symbols('r, phi, theta')
        if unit:
            r = 1
        super(Sphere, self).__init__(sp.diag(r ** 2, r ** 2 * sp.sin(th) ** 2), (th, ph))


class Minkowski(Manifold):

    def __init__(self):
        t, x, y, z = sp.symbols('t, x, y, z')
        super(Minkowski, self).__init__(sp.diag(-1, 1, 1, 1), (t, x, y, z))


class Schwarzschild(Manifold):

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
