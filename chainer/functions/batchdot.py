from chainer import Function
import numpy

class BatchDot(Function):
    """Dot product in a batch manner: batch matrix multiplication between two
    arrays of 2D arrays or batch inner product between two arrays of 1D arrays
    """
    def _output_shape(self, x):
        x0, x1 = x
        batch_size = x0.shape[0]
        if len(x0.shape) == 3:
            # matrix multiplication
            return [batch_size, x0.shape[1], x1.shape[2]]
        elif len(x0.shape) == 2:
            # inner product
            return [batch_size]

    def forward_cpu(self, x):
        x0, x1 = x
        assert len(x0.shape) == len(x1.shape)
        assert x0.shape[0] == x1.shape[0]
        batch_size = x0.shape[0]
        shape = self._output_shape(x)
        ret = numpy.empty(shape)
        for i in xrange(batch_size):
            ret[i] = numpy.dot(x0[i], x1[i])
        return ret,

    def backward_cpu(self, x, gy):
        x0, x1 = x
        assert x0.shape[0] == x1.shape[0]
        batch_size = x0.shape[0]
        gx0 = numpy.empty(x0.shape)
        gx1 = numpy.empty(x1.shape)
        for i in xrange(batch_size):
            gx0[i] = numpy.dot(gy[0][i], x1[i].T).reshape(x0[0].shape)
            gx1[i] = numpy.dot(x0[i].T, gy[0][i]).reshape(x1[0].shape)
        return gx0, gx1

def batchdot(x, y):
    """Compute dot product in a batch manner: batch matrix multiplication
    between two arrays of 2D arrays or batch inner product between two arrays
    of 1D arrays"""
    return BatchDot()(x,y)

