import sympy as sp


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
            self.values = values
        if names is None:
            self.names_map = {}
        else:
            self.names_map = {n: k for k, n in enumerate(names)}

    def __len__(self):
        return len(self.values)

    def __getitem__(self, m_idx):
        return self.values[self._translate_multi_idx(m_idx)]

    def __setitem__(self, m_idx, value):
        self.values[self._translate_multi_idx(m_idx)] = value

    def _translate_multi_idx(self, m_idx):
        result = []
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


