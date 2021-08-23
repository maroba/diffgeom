import unittest
import sympy as sp
from diffgeom import Manifold, Tensor, IncompatibleIndexPositionException, RiemannTensor, Sphere, Minkowski


class TestIndexedObject(unittest.TestCase):

    def test_indexedobject_access_component_by_name(self):
        # Arrange
        r, phi = sp.symbols('r, phi')
        metric = sp.diag(1, r ** 2)
        plane = Manifold(metric, coords=(r, phi))
        A = Tensor(plane, 'ulu', {(0, 1, 0): r**4, (1, 0, 1): 1/r**2})

        # Act
        actual_get = A[r, phi, r]
        A[r, r, r] = 5
        actual_set = A[0, 0, 0]

        # Assert
        self.assertEqual(actual_get, r**4)
        self.assertEqual(actual_get, A[0, 1, 0])
        self.assertEqual(actual_set, 5)
        self.assertEqual(plane.gammas[phi, r, phi], 1 / r)


class TestTensor(unittest.TestCase):

    def test_init_tensor_with_values_stores_values(self):
        # Arrange
        r, phi = sp.symbols('r, phi')
        metric = sp.diag(1, r**2)
        plane = Manifold(metric, coords=(r, phi))

        # Act
        A = Tensor(plane, 'ulu', {(0, 1, 0): r**4, (1, 0, 1): 1/r**2})

        # Assert
        self.assertEqual(len(A), 2)
        self.assertEqual(A[0, 1, 0], r**4)
        self.assertEqual(A[1, 0, 1], 1 / r**2)

    def test_set_tensor_component_stores_value(self):
        # Arrange
        r, phi = sp.symbols('r, phi')
        metric = sp.diag(1, r**2)
        plane = Manifold(metric, coords=(r, phi))
        A = Tensor(plane, 'ulu', {(0, 1, 0): r ** 4, (1, 0, 1): 1 / r ** 2})

        # Act
        A[1, 1, 1] = r**3

        # Assert
        self.assertEqual(A[1, 1, 1], r**3)

    def test_init_tensor_without_values_is_zero_tensor(self):

        # Arrange
        r, phi = sp.symbols('r, phi')
        metric = sp.diag(1, r ** 2)
        plane = Manifold(metric, coords=(r, phi))
        # Act
        A = Tensor(plane, 'ulu')

        # Assert
        self.assertEqual(len(A), 0)

    def test_add_tensors_of_same_structure_gives_sum(self):
        # Arrange
        r, phi = sp.symbols('r, phi')
        metric = sp.diag(1, r ** 2)
        plane = Manifold(metric, coords=(r, phi))
        A = Tensor(plane, 'ulu')
        B = Tensor(plane, 'ulu')
        A[0, 1, 1] = 1
        B[0, 1, 1] = 2
        B[0, 0, 1] = 3

        # Act
        C = A + B

        # Assert
        self.assertEqual(len(C), 2)
        self.assertEqual(C[0, 1, 1], 3)
        self.assertEqual(C[0, 0, 1], 3)

    def test_sub_tensors_of_same_structure_gives_sum(self):
        # Arrange
        r, phi = sp.symbols('r, phi')
        metric = sp.diag(1, r ** 2)
        plane = Manifold(metric, coords=(r, phi))
        A = Tensor(plane, 'ulu')
        B = Tensor(plane, 'ulu')
        A[0, 1, 1] = 1
        B[0, 1, 1] = 1
        B[0, 0, 1] = 3

        # Act
        C = A - B

        # Assert
        self.assertEqual(len(C), 1)
        self.assertEqual(C[0, 0, 1], -3)

    def test_add_or_sub_incompatible_tensors_raises_exception(self):
        # Arrange
        r, phi = sp.symbols('r, phi')
        metric = sp.diag(1, r ** 2)
        plane = Manifold(metric, coords=(r, phi))
        A = Tensor(plane, 'uuu')
        B = Tensor(plane, 'lll')

        # Act / Assert
        with self.assertRaises(IncompatibleIndexPositionException):
            A + B

        with self.assertRaises(IncompatibleIndexPositionException):
            A - B

    def test_tensor_product_returns_new_tensor(self):
        # Arrange
        r, phi = sp.symbols('r, phi')
        metric = sp.diag(1, r ** 2)
        plane = Manifold(metric, coords=(r, phi))
        A = Tensor(plane, 'ulu')
        B = Tensor(plane, 'lu')

        # Act
        C = A * B

        # Assert
        self.assertEqual(C.rank, 5)

    def test_multiply_tensor_with_scalar(self):
        # Arrange
        r, phi = sp.symbols('r, phi')
        rho = sp.symbols('rho')
        metric = sp.diag(1, r ** 2)
        plane = Manifold(metric, coords=(r, phi))
        A = Tensor(plane, 'ulu', {0: 1, 1: 2})
        B = rho * A

        assert B[0] == rho * A[0]
        assert B[1] == rho * A[1]

    def test_multiply_tensor_with_scalar_reverse(self):
        # Arrange
        r, phi = sp.symbols('r, phi')
        rho = sp.symbols('rho')
        metric = sp.diag(1, r ** 2)
        plane = Manifold(metric, coords=(r, phi))
        A = Tensor(plane, 'ulu', {0: 1, 1: 2})
        B = A * rho

        assert B[0] == rho * A[0]
        assert B[1] == rho * A[1]

    def test_tensor_contraction_returns_new_tensor_of_lower_rank(self):
        # Arrange
        minkowski = Minkowski()
        coords = minkowski.coords
        x = Tensor(minkowski, 'u', values={k: c for k, c in enumerate(coords)})
        eta = Tensor(minkowski, 'll', {(k, k): minkowski.metric[k, k] for k in range(4)})

        # Act
        prod = eta * x
        x_low = prod.contract(1, 2)

        # Assert
        self.assertEqual(prod.rank, 3)
        self.assertEqual(x_low.rank, 1)
        self.assertEqual(x_low.idx_pos, 'l')
        self.assertEqual(x_low[0], -coords[0])
        for k in range(1, 4):
            self.assertEqual(x_low[k], coords[k])

    def test_lowering_index_returns_lowered_tensor(self):
        # Arrange
        coords = sp.symbols('t, x, y, z')
        metric = sp.diag(-1, 1, 1, 1)
        minkowski = Manifold(metric, coords)
        xx = Tensor(minkowski, 'uu', values={(k,k): c for k, c in enumerate(coords)})

        # Act
        xx_low = xx.lower_index(0)

        # Assert
        self.assertEqual(xx_low.idx_pos, 'lu')
        self.assertEqual(xx_low[0, 0], -coords[0])
        self.assertEqual(xx_low[1, 1], coords[1])

    def test_raising_index_returns_raised_tensor(self):
        # Arrange
        minkowski = Minkowski()
        coords = minkowski.coords
        xx = Tensor(minkowski, 'll', values={(k,k): c for k, c in enumerate(coords)})
        xx[0, 0] *= -1

        # Act
        xx_high = xx.raise_index(0)

        # Assert
        self.assertEqual(xx_high.idx_pos, 'ul')
        self.assertEqual(xx_high[0, 0], coords[0])
        self.assertEqual(xx_high[1, 1], coords[1])

    def test_diff_tensor(self):
        r, ph, th = sp.symbols('r, phi, theta')
        sphere = Manifold(sp.diag(r ** 2, r ** 2 * sp.sin(th) ** 2), (th, ph))

        A = Tensor(sphere, 'ul', {(0, 0): th**2+ph**3,
                                    (0, 1): th**3 + ph**2,
                                    (1, 0): th**2 * ph**3,
                                    (1, 1): th**3 * ph**2
                                    })

        nabla_A = A.diff()
        C = sphere.gammas

        assert nabla_A[(th, th, th)] == sp.diff(A[th, th], th)
        assert nabla_A[(th, th, ph)] ==  sp.diff(A[th, th], ph) + A[th, th] * C[th, th, ph] + A[ph, th] * C[th, ph, ph] \
                                        - A[th, th] * C[th, th, ph] - A[th, ph] * C[ph, th, ph]
        assert nabla_A[(th, th, ph)] != sp.diff(A[th, th], ph)


class TestRiemann(unittest.TestCase):

    def test_euclidean_plan_has_vanishing_riemann(self):

        r, phi = sp.symbols('r phi')
        metric = sp.diag(1, r**2)
        plane = Manifold(metric, (r, phi))

        riemann = RiemannTensor(plane)
        assert len(riemann) == 0

    def test_sphere_has_nonvanishing_riemann(self):

        sphere = Sphere()
        th = sphere.coords[0]
        R = RiemannTensor(sphere)

        assert len(R) > 0
        assert R[0, 1, 0, 1] == sp.sin(th)**2
        assert R[0, 1, 1, 0] == -sp.sin(th) ** 2
        assert R[1, 0, 1, 0] == 1
        assert R[1, 0, 0, 1] == -1