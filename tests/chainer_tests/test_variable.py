import unittest

import numpy as np

import chainer
from chainer import cuda
from chainer import testing
from chainer.testing import attr

import six


class Constant(chainer.Function):

    def __init__(self, outputs):
        self.outputs = outputs

    def forward_cpu(self, inputs):
        return self.outputs

    def forward_gpu(self, inputs):
        return tuple(map(cuda.to_gpu, self.outputs))

    def backward_cpu(self, inputs, grad_outputs):
        return tuple(map(np.zeros_like, inputs))

    def backward_gpu(self, inputs, grad_outputs):
        return tuple(map(cuda.zeros_like, inputs))


def constant(xs, value):
    return Constant(value)(*xs)


class TestVariable(unittest.TestCase):

    def setUp(self):
        self.x = np.random.uniform(-1, 1, 10).astype(np.float32)
        self.a = np.random.uniform(0.1, 10, 10).astype(np.float32)
        self.c = np.arange(10).reshape(2, 5).astype(np.float32)

    def test_repr(self):
        x = chainer.Variable(self.x, name='x')
        self.assertEqual(repr(x), '<variable x>')

    def test_str(self):
        x = chainer.Variable(self.x, name='x')
        self.assertEqual(str(x), 'x')

    def check_len(self, gpu):
        x = self.x
        if gpu:
            x = cuda.to_gpu(x)
        x = chainer.Variable(x)
        self.assertEqual(len(x), 10)

    def test_len_cpu(self):
        self.check_len(False)

    @attr.gpu
    def test_len_gpu(self):
        self.check_len(True)

    def check_label(self, expected, gpu):
        c = self.c
        if gpu:
            c = cuda.to_gpu(c)
        c = chainer.Variable(c)
        self.assertEqual(c.label, expected)

    def test_label_cpu(self):
        self.check_label('(2, 5), float32', False)

    @attr.gpu
    def test_label_gpu(self):
        self.check_label('(2, 5), float32', True)

    def check_backward(self, inputs, intermediates, outputs, retain_grad):
        for o in outputs:
            o.backward(retain_grad)

        self.assertTrue(all([x.grad is not None for x in inputs]))
        if retain_grad:
            self.assertTrue(all([x.grad is not None for x in intermediates]))
        else:
            self.assertTrue(all([x.grad is None for x in intermediates]))
        self.assertTrue(any([x.grad is not None for x in outputs]))

    # length is number of edges. So, # of Variables created is length+1
    def create_linear_chain(self, length, gpu):
        if gpu:
            x = chainer.Variable(cuda.to_gpu(self.x))
        else:
            x = chainer.Variable(self.x)
        ret = [x]
        for i in six.moves.range(length):
            ret.append(constant((ret[i], ), (self.a, )))
        if gpu:
            ret[-1].grad = cuda.zeros_like(ret[-1].data)
        else:
            ret[-1].grad = np.zeros_like(ret[-1].data)
        return ret

    def test_backward_cpu(self):
        ret = self.create_linear_chain(2, False)
        self.check_backward((ret[0], ), (ret[1], ), (ret[2], ), False)

    @attr.gpu
    def test_backward_gpu(self):
        ret = self.create_linear_chain(2, True)
        self.check_backward((ret[0], ), (ret[1], ), (ret[2], ), False)

    def test_backward_cpu_retain_grad(self):
        ret = self.create_linear_chain(2, False)
        self.check_backward((ret[0], ), (ret[1], ), (ret[2], ), True)

    @attr.gpu
    def test_backward_gpu_retain_grad(self):
        ret = self.create_linear_chain(2, True)
        self.check_backward((ret[0], ), (ret[1], ), (ret[2], ), True)

    def test_unchain_backward_cpu(self):
        ret = self.create_linear_chain(3, False)
        ret[1].unchain_backward()
        self.check_backward((ret[1], ), (ret[2], ), (ret[3], ), False)

    @attr.gpu
    def test_unchain_backward_gpu(self):
        ret = self.create_linear_chain(3, True)
        ret[1].unchain_backward()
        self.check_backward((ret[1], ), (ret[2], ), (ret[3], ), False)

    def test_unchain_backward_cpu_retain_grad(self):
        ret = self.create_linear_chain(3, False)
        ret[1].unchain_backward()
        self.check_backward((ret[1], ), (ret[2], ), (ret[3], ), False)

    @attr.gpu
    def test_unchain_backward_gpu_retain_grad(self):
        ret = self.create_linear_chain(3, False)
        ret[1].unchain_backward()
        self.check_backward((ret[1], ), (ret[2], ), (ret[3], ), False)

    def test_invalid_value_type(self):
        with self.assertRaises(AssertionError):
            chainer.Variable(1)

    def test_grad_type_check_pass(self):
        a = chainer.Variable(np.empty((3,), dtype=np.float32))
        a.grad = np.ndarray((3,), dtype=np.float32)

    def test_grad_type_check_type(self):
        a = chainer.Variable(np.empty((), dtype=np.float32))
        with self.assertRaises(TypeError):
            a.grad = np.float32()

    @attr.gpu
    def test_grad_type_check_type_cpu_gpu_mixture(self):
        a = chainer.Variable(np.empty((3,), dtype=np.float32))
        with self.assertRaises(TypeError):
            a.grad = cuda.empty((3,), dtype=np.float32)

    def test_grad_type_check_dtype(self):
        a = chainer.Variable(np.empty((3,), dtype=np.float32))
        with self.assertRaises(TypeError):
            a.grad = np.empty((3,), dtype=np.float64)

    def test_grad_type_check_shape(self):
        a = chainer.Variable(np.empty((3,), dtype=np.float32))
        with self.assertRaises(ValueError):
            a.grad = np.empty((2,), dtype=np.float32)

    def test_to_cpu_from_cpu(self):
        a = chainer.Variable(np.zeros(3, dtype=np.float32))
        a.grad = np.ones_like(a.data)
        b = a.data
        gb = a.grad
        c = b.copy()
        gc = gb.copy()
        a.to_cpu()
        self.assertIs(a.data, b)
        self.assertIs(a.grad, gb)
        np.testing.assert_array_equal(a.data, c)
        np.testing.assert_array_equal(a.grad, gc)

    @attr.gpu
    def test_to_cpu(self):
        a = chainer.Variable(cuda.cupy.zeros(3, dtype=np.float32))
        a.grad = cuda.cupy.ones_like(a.data)
        a.to_cpu()
        np.testing.assert_array_equal(a.data, np.zeros(3, dtype=np.float32))
        np.testing.assert_array_equal(a.grad, np.ones(3, dtype=np.float32))

    @attr.gpu
    def test_to_gpu_from_gpu(self):
        cp = cuda.cupy
        a = chainer.Variable(cp.zeros(3, dtype=np.float32))
        a.grad = cuda.cupy.ones_like(a.data)
        b = a.data
        gb = a.grad
        c = b.copy()
        gc = gb.copy()
        a.to_gpu()
        self.assertIs(a.data, b)
        self.assertIs(a.grad, gb)
        cp.testing.assert_array_equal(a.data, c)
        cp.testing.assert_array_equal(a.grad, gc)

    @attr.gpu
    def test_to_gpu(self):
        cp = cuda.cupy
        a = chainer.Variable(np.zeros(3, dtype=np.float32))
        a.grad = np.ones(3, dtype=np.float32)
        a.to_gpu()
        cp.testing.assert_array_equal(a.data, cp.zeros(3, dtype=np.float32))
        cp.testing.assert_array_equal(a.grad, cp.ones(3, dtype=np.float32))

    def check_zerograd(self, a_data, fill=False):
        xp = cuda.get_array_module(a_data)
        a = chainer.Variable(a_data)
        if fill:
            a.grad = xp.full_like(a_data, np.nan)

        a.zerograd()
        self.assertIsNot(a.grad, None)
        g_expect = xp.zeros_like(a.data)
        xp.testing.assert_array_equal(a.grad, g_expect)

    def test_zerograd_cpu(self):
        self.check_zerograd(np.empty(3, dtype=np.float32))

    def test_zerograd_fill_cpu(self):
        self.check_zerograd(np.empty(3, dtype=np.float32), fill=True)

    @attr.multi_gpu(2)
    def test_zerograds_multi_gpu(self):
        cupy = cuda.cupy
        with cuda.get_device(1):
            a = chainer.Variable(cupy.empty(3, dtype=np.float32))
        a.zerograd()
        self.assertIsNot(a.grad, None)
        self.assertEqual(int(a.grad.device), 1)
        with cuda.get_device(1):
            g_expect = cupy.zeros_like(a.data)
            cupy.testing.assert_array_equal(a.grad, g_expect)

    @attr.multi_gpu(2)
    def test_zerograds_fill_multi_gpu(self):
        cupy = cuda.cupy
        with cuda.get_device(1):
            a = chainer.Variable(cupy.empty(3, dtype=np.float32))
            a.grad = cupy.empty_like(a.data)
        a.zerograd()
        self.assertEqual(int(a.grad.device), 1)
        with cuda.get_device(1):
            g_expect = cupy.zeros_like(a.data)
            cupy.testing.assert_array_equal(a.grad, g_expect)

    @attr.gpu
    def test_zerograd_gpu(self):
        self.check_zerograd(cuda.cupy.empty(3, dtype=np.float32))

    @attr.gpu
    def test_zerograd_fill_gpu(self):
        self.check_zerograd(cuda.cupy.empty(3, dtype=np.float32), fill=True)

    def check_copydata(self, data1, data2, expect):
        xp = cuda.get_array_module(data1)
        v = chainer.Variable(data1)
        w = chainer.Variable(data2)
        v.copydata(w)
        xp.testing.assert_array_equal(v.data, expect)

    def test_copydata_cpu_to_cpu(self):
        self.check_copydata(np.zeros(3, dtype=np.float32),
                            np.ones(3, dtype=np.float32),
                            np.ones(3, dtype=np.float32))

    @attr.gpu
    def test_copydata_cpu_to_gpu(self):
        cp = cuda.cupy
        self.check_copydata(cp.zeros(3, dtype=np.float32),
                            np.ones(3, dtype=np.float32),
                            cp.ones(3, dtype=np.float32))

    @attr.gpu
    def test_copydata_gpu_to_gpu(self):
        cp = cuda.cupy
        self.check_copydata(cp.zeros(3, dtype=np.float32),
                            cp.ones(3, dtype=np.float32),
                            cp.ones(3, dtype=np.float32))

    @attr.gpu
    def test_copydata_gpu_to_cpu(self):
        cp = cuda.cupy
        self.check_copydata(np.zeros(3, dtype=np.float32),
                            cp.ones(3, dtype=np.float32),
                            np.ones(3, dtype=np.float32))

    @attr.multi_gpu(2)
    def test_copydata_gpu_to_another_gpu(self):
        cp = cuda.cupy
        with cuda.get_device(0):
            data1 = cp.zeros(3, dtype=np.float32)
            expect = cp.ones(3, dtype=np.float32)
        with cuda.get_device(1):
            data2 = cp.ones(3, dtype=np.float32)
        self.check_copydata(data1, data2, expect)

    def check_addgrad(self, src, dst, expect):
        xp = cuda.get_array_module(dst)
        a = chainer.Variable(src)
        a.grad = src
        b = chainer.Variable(dst)
        b.grad = dst
        b.addgrad(a)
        xp.testing.assert_array_equal(b.grad, expect)

    def test_addgrad_cpu_to_cpu(self):
        self.check_addgrad(np.full(3, 10, dtype=np.float32),
                           np.full(3, 20, dtype=np.float32),
                           np.full(3, 30, dtype=np.float32))

    @attr.gpu
    def test_addgrad_cpu_to_gpu(self):
        cp = cuda.cupy
        self.check_addgrad(np.full(3, 10, dtype=np.float32),
                           cp.full(3, 20, dtype=np.float32),
                           cp.full(3, 30, dtype=np.float32))

    @attr.gpu
    def test_addgrad_gpu_to_gpu(self):
        cp = cuda.cupy
        self.check_addgrad(cp.full(3, 10, dtype=np.float32),
                           cp.full(3, 20, dtype=np.float32),
                           cp.full(3, 30, dtype=np.float32))

    @attr.gpu
    def test_addgrad_gpu_to_cpu(self):
        cp = cuda.cupy
        self.check_addgrad(cp.full(3, 10, dtype=np.float32),
                           np.full(3, 20, dtype=np.float32),
                           np.full(3, 30, dtype=np.float32))

    @attr.multi_gpu(2)
    def test_addgrad_gpu_to_another_gpu(self):
        cp = cuda.cupy
        with cuda.get_device(1):
            a = cp.full(3, 10, dtype=np.float32)
        with cuda.get_device(0):
            b = cp.full(3, 20, dtype=np.float32)
            c = cp.full(3, 30, dtype=np.float32)
        self.check_addgrad(a, b, c)


class TestVariableSetCreator(unittest.TestCase):
    class MockFunction(object):
        pass

    def setUp(self):
        self.x = np.random.uniform(-1, 1, (2, 5)).astype(np.float32)
        self.f = self.MockFunction()
        self.f.rank = 10

    def check_set_creator(self, x):
        x = chainer.Variable(x)
        x.set_creator(self.f)
        self.assertEqual(x.creator, self.f)
        self.assertEqual(x.rank, 11)

    def test_set_creator_cpu(self):
        self.check_set_creator(self.x)

    @attr.gpu
    def test_set_creator_gpu(self):
        self.check_set_creator(cuda.to_gpu(self.x))


testing.run_module(__name__, __file__)
