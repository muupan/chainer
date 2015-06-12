from chainer import Function
import numpy

class Dot(Function):
    """Dot product: matrix multiplication between 2D arrays or inner product
    between 1D arrays."""
    def forward_cpu(self, x):
        assert len(x[0].shape) == len(x[1].shape)
        return numpy.dot(x[0], x[1]),

    def backward_cpu(self, x, gy):
        gx0 = numpy.dot(gy[0], x[1].T).reshape(x[0].shape)
        gx1 = numpy.dot(x[0].T, gy[0]).reshape(x[1].shape)
        return gx0, gx1

def dot(x, y):
    """Compute dot product: matrix multiplication between 2D arrays or inner
    product between 1D arrays."""
    return Dot()(x,y)
