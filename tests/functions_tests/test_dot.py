from unittest import TestCase
import numpy
from chainer      import cuda, Variable
from chainer.cuda import to_gpu
from chainer.gradient_check import assert_allclose, numerical_grad
import chainer.functions as F

cuda.init()

class _TestDot(TestCase):
    def check_forward(self, op, x1_data, x2_data):
        x1 = Variable(x1_data)
        x2 = Variable(x2_data)
        y = op(x1, x2)
        if isinstance(y.data, cuda.GPUArray):
            self.assertTrue(hasattr(y.data.gpudata, 'device'))
        assert_allclose(numpy.dot(self.x1, self.x2), y.data)

    def forward_cpu(self, op):
        self.check_forward(op, self.x1, self.x2)

    def test_dot_forward_cpu(self):
        self.forward_cpu(lambda x, y: F.dot(x, y))

    def forward_gpu(self, op):
        self.check_forward(op, to_gpu(self.x1), to_gpu(self.x2))

    def test_dot_forward_gpu(self):
        self.forward_gpu(lambda x, y: F.dot(x, y))

    def check_backward(self, op, x1_data, x2_data, y_grad, atol):
        x1 = Variable(x1_data)
        x2 = Variable(x2_data)
        y = op(x1, x2)
        y.grad = y_grad
        y.backward()

        func = y.creator
        f = lambda: func.forward((x1.data, x2.data))
        gx1, gx2 = numerical_grad(f, (x1.data, x2.data), (y.grad,))
        assert_allclose(gx1, x1.grad, atol=atol)
        assert_allclose(gx2, x2.grad, atol=atol)

    def backward_cpu(self, op, atol=1e-5):
        self.check_backward(op, self.x1, self.x2, self.gy, atol)

    def test_dot_backward_cpu(self):
        self.backward_cpu(lambda x, y: F.dot(x, y))

    def backward_gpu(self, op, atol=1e-5):
        self.check_backward(op, to_gpu(self.x1), to_gpu(self.x2), to_gpu(self.gy), atol)

    def test_dot_backward_gpu(self):
        self.backward_gpu(lambda x, y: F.dot(x, y))

class TestDotMatrix(_TestDot):
    def setUp(self):
        self.x1 = numpy.random.uniform(.5, 1, (2, 1)).astype(numpy.float32)
        self.x2 = numpy.random.uniform(.5, 1, (1, 3)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)

class TestDotVector(_TestDot):
    def setUp(self):
        self.x1 = numpy.random.uniform(.5, 1, (3,)).astype(numpy.float32)
        self.x2 = numpy.random.uniform(.5, 1, (3,)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, ()).astype(numpy.float32)

