from unittest import TestCase

from chainer.optimizers import SGD
from chainer.testing import attr

from linear_model import LinearModel


class TestSGD(TestCase):

    def setUp(self):
        self.optimizer = SGD(0.1)
        self.model = LinearModel(self.optimizer)

    def test_linear_model_cpu(self):
        self.assertGreater(self.model.accuracy(False), 0.7)

    @attr.gpu
    def test_linear_model_gpu(self):
        self.assertGreater(self.model.accuracy(True), 0.7)
