from chainer import cuda, Function
import numpy

def _as_mat(x):
    # vectors are considered as column vectors
    return x.reshape(1, (len(x))) if len(x.shape) == 1 else x

def _as_trans_op(trans):
    return 't' if trans else 'n'

def _dot_cpu_notransout(x, y, transa=False, transb=False):
    x = _as_mat(x)
    y = _as_mat(y)
    if transa:
        x = x.T
    if transb:
        y = y.T
    return numpy.dot(x, y)

def _dot_cpu(x, y, transa=False, transb=False, transout=False):
    if transout:
        # (X Y)^T = Y^T X^T
        return _dot_cpu_notransout(y, x, transa=not transb, transb=not transa)
    else:
        return _dot_cpu_notransout(x, y, transa=transa, transb=transb)

def _dot_gpu_notransout(x, y, transa=False, transb=False):
    x = _as_mat(x)
    y = _as_mat(y)
    with cuda.using_cumisc():
        return cuda.culinalg.dot(x, y,
                transa=_as_trans_op(transa),
                transb=_as_trans_op(transb))

def _dot_gpu(x, y, transa=False, transb=False, transout=False):
    if transout:
        # (X Y)^T = Y^T X^T
        return _dot_gpu_notransout(y, x, transa=not transb, transb=not transa)
    else:
        return _dot_gpu_notransout(x, y, transa=transa, transb=transb)

class Dot(Function):
    """Dot product: matrix multiplication between 2D arrays or inner product
    between 1D arrays."""

    def __init__(self, transa=False, transb=False):
        self.transa = transa
        self.transb = transb

    def forward_cpu(self, x):
        assert len(x[0].shape) == len(x[1].shape)
        return _dot_cpu(x[0], x[1], transa=self.transa, transb=self.transb),

    def forward_gpu(self, x):
        assert len(x[0].shape) == len(x[1].shape)
        return _dot_gpu(x[0], x[1], transa=self.transa, transb=self.transb),

    def backward_cpu(self, x, gy):
        gx0 = _dot_cpu(gy[0], x[1], transb=not self.transb, transout=self.transa).reshape(x[0].shape)
        gx1 = _dot_cpu(x[0], gy[0], transa=not self.transa, transout=self.transb).reshape(x[1].shape)
        return gx0, gx1

    def backward_gpu(self, x, gy):
        with cuda.using_cumisc():
            gx0 = _dot_gpu(gy[0], x[1], transb=not self.transb, transout=self.transa).reshape(x[0].shape)
            gx1 = _dot_gpu(x[0], gy[0], transa=not self.transa, transout=self.transb).reshape(x[1].shape)
            return gx0, gx1

def dot(x, y, transa=False, transb=False):
    """Compute dot product: matrix multiplication between 2D arrays or inner
    product between 1D arrays."""
    return Dot(transa=transa, transb=transb)(x, y)
