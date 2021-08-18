import sympy as sp
from itertools import product


class Manifold(object):

    def __init__(self, metric, coords):
        self.metric = metric
        self.coords = coords
        self.gammas = Christoffel(coords, metric)


class IndexedObject(object):

    def __init__(self, values=None, names=None):
        if values is None:
            self.values = {}
        else:
            self.values = dict(values)
        if names is None:
            self.names_map = {}
        else:
            self.names_map = {n: k for k, n in enumerate(names)}

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


class Christoffel(IndexedObject):

    def __init__(self, coords, metric):
        super(Christoffel, self).__init__(names=coords)

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

    def __init__(self, manifold, idx_pos, values=None):
        super(Tensor, self).__init__(values, names=manifold.coords)
        self.manifold = manifold
        self.idx_pos = idx_pos

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

        result = Tensor(self.manifold, idx_pos)

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
        return product(*tuple(list(range(len(self.manifold.coords))) for _ in range(self.rank)))


class IncompatibleIndexPositionException(Exception):
    pass
