from unittest import TestCase
import numpy
from chainer      import cuda, Variable
from chainer.cuda import to_gpu
from chainer.gradient_check import assert_allclose, numerical_grad
import chainer.functions as F

cuda.init()

class _TestDot(TestCase):
    def check_forward(self, x1_data, x2_data):
        x1 = Variable(x1_data)
        x2 = Variable(x2_data)
        y = self.op(x1, x2)
        if isinstance(y.data, cuda.GPUArray):
            self.assertTrue(hasattr(y.data.gpudata, 'device'))
        assert_allclose(self.forward_answer, y.data)

    def forward_cpu(self):
        self.check_forward(self.x1, self.x2)

    def test_dot_forward_cpu(self):
        self.forward_cpu()

    def forward_gpu(self):
        self.check_forward(to_gpu(self.x1), to_gpu(self.x2))

    def test_dot_forward_gpu(self):
        self.forward_gpu()

    def check_backward(self, x1_data, x2_data, y_grad, atol):
        x1 = Variable(x1_data)
        x2 = Variable(x2_data)
        y = self.op(x1, x2)
        y.grad = y_grad
        y.backward()

        func = y.creator
        f = lambda: func.forward((x1.data, x2.data))
        gx1, gx2 = numerical_grad(f, (x1.data, x2.data), (y.grad,))
        assert_allclose(gx1, x1.grad, atol=atol)
        assert_allclose(gx2, x2.grad, atol=atol)

    def backward_cpu(self, atol=1e-2):
        self.check_backward(self.x1, self.x2, self.gy, atol)

    def test_dot_backward_cpu(self):
        self.backward_cpu()

    def backward_gpu(self, atol=1e-2):
        self.check_backward(to_gpu(self.x1), to_gpu(self.x2), to_gpu(self.gy), atol)

    def test_dot_backward_gpu(self):
        self.backward_gpu()

class TestDotMatrixMatrix(_TestDot):
    def setUp(self):
        self.x1 = numpy.random.uniform(.5, 1, (2, 5)).astype(numpy.float32)
        self.x2 = numpy.random.uniform(.5, 1, (5,10)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (2,10)).astype(numpy.float32)
        self.op = lambda x, y: F.dot(x, y)
        self.forward_answer = numpy.dot(self.x1, self.x2)

class TestDotMatrixTMatrix(_TestDot):
    def setUp(self):
        self.x1 = numpy.random.uniform(.5, 1, (5, 2)).astype(numpy.float32)
        self.x2 = numpy.random.uniform(.5, 1, (5,10)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (2,10)).astype(numpy.float32)
        self.op = lambda x, y: F.dot(x, y, transa=True)
        self.forward_answer = numpy.dot(self.x1.T, self.x2)

class TestDotMatrixMatrixT(_TestDot):
    def setUp(self):
        self.x1 = numpy.random.uniform(.5, 1, (2, 5)).astype(numpy.float32)
        self.x2 = numpy.random.uniform(.5, 1, (10,5)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (2,10)).astype(numpy.float32)
        self.op = lambda x, y: F.dot(x, y, transb=True)
        self.forward_answer = numpy.dot(self.x1, self.x2.T)

class TestDotMatrixTMatrixT(_TestDot):
    def setUp(self):
        self.x1 = numpy.random.uniform(.5, 1, (5, 2)).astype(numpy.float32)
        self.x2 = numpy.random.uniform(.5, 1, (10,5)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (2,10)).astype(numpy.float32)
        self.op = lambda x, y: F.dot(x, y, transa=True, transb=True)
        self.forward_answer = numpy.dot(self.x1.T, self.x2.T)

class TestDotVectorVectorT(_TestDot):
    def setUp(self):
        self.x1 = numpy.random.uniform(.5, 1, (5,)).astype(numpy.float32)
        self.x2 = numpy.random.uniform(.5, 1, (5,)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (1, 1)).astype(numpy.float32)
        self.op = lambda x, y: F.dot(x, y, transb=True)
        self.forward_answer = numpy.dot(self.x1, self.x2)

class TestDotVectorTVector(_TestDot):
    def setUp(self):
        self.x1 = numpy.random.uniform(.5, 1, (5,)).astype(numpy.float32)
        self.x2 = numpy.random.uniform(.5, 1, (5,)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (5, 5)).astype(numpy.float32)
        self.op = lambda x, y: F.dot(x, y, transa=True)
        self.forward_answer = numpy.dot(self.x1.reshape(5, 1), self.x2.reshape(1, 5))
