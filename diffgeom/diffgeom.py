import sympy as sp
from itertools import product


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


class IndexedObject(object):

    def __init__(self, values=None, names=None, latex_head=None, idx_pos=None):
        self.values = {}
        if latex_head is not None:
            self.latex_head = latex_head
        else:
            self.latex_head = '\\Box'

        if values is not None:
            for key, value in values.items():
                if isinstance(key, int):
                    key = key,
                if not isinstance(key, tuple):
                    key = tuple(key)
                self.values[key] = value

        if names is None:
            self.names_map = {}
        else:
            self.names_map = {n: k for k, n in enumerate(names)}

        self.idx_pos = idx_pos


    def __len__(self):
        return len(self.values)

    def __getitem__(self, m_idx):
        return self.values.get(self._translate_multi_idx(m_idx), 0)

    def __setitem__(self, m_idx, value):
        self.values[self._translate_multi_idx(m_idx)] = value
        if value == 0:
            del self.values[self._translate_multi_idx(m_idx)]

    def _translate_multi_idx(self, m_idx):
        result = []
        if not hasattr(m_idx, '__len__'):
            m_idx = (m_idx, )
        for idx in m_idx:
            if isinstance(idx, int):
                result.append(idx)
            else:
                result.append(self.names_map[idx])
        return tuple(result)

    def simplify(self):
        for key, value in self.values.items():
            self[key] = sp.simplify(value)

    def _repr_latex_(self):

        head = self.latex_head
        from sympy.printing.latex import latex
        latex_str = ''

        for m_idx, value in self.values.items():
            latex_value = latex(value, mode='plain')
            latex_str += head
            for k, pos in enumerate(self.idx_pos):
                if pos == 'u':
                    latex_str += '^{' +  str(m_idx[k]) + '}\\,'
                else:
                    latex_str += '_{' +  str(m_idx[k]) + '}\\,'
            latex_str += ' = ' + latex_value + '\\\\\n'
        return '$$\\displaystyle %s $$' % latex_str


class Christoffel(IndexedObject):

    def __init__(self, coords, metric):
        super(Christoffel, self).__init__(names=coords, latex_head='\\Gamma', idx_pos='ull')

        x = coords
        g = metric
        n = len(x)
        g_inv = g.inv()
        for mu in range(n):
            for alpha in range(n):
                for beta in range(n):
                    gamma = 0
                    for nu in range(n):
                        gamma += g_inv[mu, nu] * (sp.diff(g[alpha, nu], x[beta]) +
                                                  sp.diff(g[nu, beta], x[alpha]) -
                                                  sp.diff(g[alpha, beta], x[nu])
                                                  )
                    if gamma != 0:
                        self[(mu, alpha, beta)] = gamma / 2


class Tensor(IndexedObject):

    def __init__(self, manifold, idx_pos, values=None, latex_head=None):
        super(Tensor, self).__init__(values, names=manifold.coords, idx_pos=idx_pos, latex_head=latex_head)
        self.manifold = manifold

    def __add__(self, other):
        self.guard_is_compatible_with(other)
        result = Tensor(self.manifold, self.idx_pos)

        for m_idx, value in self.values.items():
            result[m_idx] = value

        for m_idx, value in other.values.items():
            result[m_idx] += value

        return result

    def __sub__(self, other):
        self.guard_is_compatible_with(other)
        result = Tensor(self.manifold, self.idx_pos)

        for m_idx, value in self.values.items():
            result[m_idx] = value

        for m_idx, value in other.values.items():
            result[m_idx] -= value

        return result

    def __mul__(self, other):
        result = Tensor(self.manifold, self.idx_pos + other.idx_pos)

        for m_idx in result.multi_indices:
            m_idx_left = m_idx[:self.rank]
            m_idx_right = m_idx[self.rank:]
            result[m_idx] = self[m_idx_left] * other[m_idx_right]

        return result

    @property
    def rank(self):
        return len(self.idx_pos)

    @property
    def coords(self):
        return self.manifold.coords

    @property
    def dims(self):
        return self.manifold.dims

    def guard_is_compatible_with(self, other):
        if self.idx_pos != other.idx_pos:
            raise IncompatibleIndexPositionException

    def contract(self, idx1, idx2):
        if idx2 < idx1:
            idx1, idx2 = idx2, idx1
        idx_pos = list(self.idx_pos)
        del idx_pos[idx2]
        del idx_pos[idx1]
        idx_pos = ''.join(idx_pos)

        result = Tensor(self.manifold, idx_pos, latex_head=self.latex_head)

        for m_idx in self.multi_indices:
            if m_idx[idx1] != m_idx[idx2]:
                continue
            value = self[m_idx]
            m_idx = list(m_idx)
            del m_idx[idx2]
            del m_idx[idx1]
            result[m_idx] += value

        return result

    @property
    def multi_indices(self):
        return product(*tuple(list(range(self.dims)) for _ in range(self.rank)))

    def lower_index(self, idx):
        if self.idx_pos[idx] == 'l':
            raise IncompatibleIndexPositionException('Index already downstairs.')
        g = self.manifold.metric
        idx_pos = list(self.idx_pos)
        idx_pos[idx] = 'l'
        result = Tensor(self.manifold, ''.join(idx_pos), latex_head=self.latex_head)

        dims = self.manifold.dims
        for m_idx in self.multi_indices:

            for sigma in range(dims):
                m_idx_var = list(m_idx)
                m_idx_var[idx] = sigma
                result[m_idx] += g[m_idx[idx], sigma] * self[m_idx_var]

        return result

    def raise_index(self, idx):
        if self.idx_pos[idx] == 'u':
            raise IncompatibleIndexPositionException('Index already upstairs.')
        g_inv = self.manifold.metric_inv

        idx_pos = list(self.idx_pos)
        idx_pos[idx] = 'u'
        result = Tensor(self.manifold, ''.join(idx_pos), latex_head=self.latex_head)

        dims = self.manifold.dims
        for m_idx in self.multi_indices:

            for sigma in range(dims):
                m_idx_var = list(m_idx)
                m_idx_var[idx] = sigma
                result[m_idx] += g_inv[m_idx[idx], sigma] * self[m_idx_var]

        return result

    def diff(self):
        result = Tensor(self.manifold, self.idx_pos + 'l')

        for m_idx in self.multi_indices:
            for sigma in range(self.dims):

                x = self.manifold.coords[sigma]
                value = sp.diff(self[m_idx], x)
                for idx in range(self.rank):
                    if self.idx_pos[idx] == 'u':
                        for alpha in range(self.dims):
                            gamma = self.manifold.gammas[(m_idx[idx], alpha, sigma)]
                            m_idx_var = list(m_idx)
                            m_idx_var[idx] = alpha
                            value += self[tuple(m_idx_var)] * gamma
                    else:
                        for alpha in range(self.dims):
                            gamma = self.manifold.gammas[(alpha, m_idx[idx], sigma)]
                            m_idx_var = list(m_idx)
                            m_idx_var[idx] = alpha
                            value -= self[tuple(m_idx_var)] * gamma

                result[tuple(list(m_idx) + [sigma])] = value

        result.latex_head = '\\nabla ' + result.latex_head
        return result


class RiemannTensor(Tensor):

    def __init__(self, manifold):
        super(RiemannTensor, self).__init__(manifold, 'ulll', latex_head='R')
        C = self.manifold.gammas
        x = self.manifold.coords
        dims = len(x)
        for alpha in range(dims):
            for beta in range(dims):
                for mu in range(dims):
                    for nu in range(dims):
                        self[alpha, beta, mu, nu] += sp.diff(C[alpha, beta, nu], x[mu]) \
                                                        - sp.diff(C[alpha, beta, mu], x[nu])
                        for sigma in range(dims):
                            self[alpha, beta, mu, nu] += C[alpha, sigma, mu] * C[sigma, beta, nu]
                            self[alpha, beta, mu, nu] -= C[alpha, sigma, nu] * C[sigma, beta, mu]

        self.simplify()


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


class IncompatibleIndexPositionException(Exception):
    pass
