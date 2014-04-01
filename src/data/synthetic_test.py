import unittest as ut
from synthetic import whiten, Random
from numpy import testing, random, cov, eye, asmatrix, diag

class TestSequenceFunctions(ut.TestCase):

    def setUp(self):
        pass

    def test_whiten(self):
        P = 100
        N = 10000
        X = random.randn(P, N)
        W = whiten(X)
        C = asmatrix(cov(W))
        C = C * diag(1/diag(C))
        testing.assert_allclose(C, asmatrix(eye(P)), atol=1e-2, rtol=1e-2)

if __name__ == '__main__':
    ut.main()