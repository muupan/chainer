import unittest

import numpy
import six

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer.testing import attr


if cuda.available:
    cuda.init()


class TestSoftmax(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)

    def check_forward(self, x_data, use_cudnn=True):
        x = chainer.Variable(x_data)
        y = functions.softmax(x, use_cudnn)

        y_expect = numpy.exp(self.x)
        for i in six.moves.range(y_expect.shape[0]):
            y_expect[i] /= y_expect[i].sum()

        gradient_check.assert_allclose(y_expect, y.data)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.cudnn
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    @attr.gpu
    def test_forwrad_gpu_no_cudnn(self):
        self.check_forward(cuda.to_gpu(self.x), False)

    def check_backward(self, x_data, gy_data, use_cudnn=True):
        x = chainer.Variable(x_data)
        y = functions.softmax(x, use_cudnn)
        y.grad = gy_data
        y.backward()

        func = y.creator
        f = lambda: func.forward((x.data,))
        gx, = gradient_check.numerical_grad(f, (x.data,), (y.grad,), eps=1e-2)

        gradient_check.assert_allclose(gx, x.grad)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.cudnn
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    @attr.gpu
    def test_backward_gpu_no_cudnn(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy), False)
