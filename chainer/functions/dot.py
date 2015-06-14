from chainer import cuda, Function
import numpy

def _as_mat(x):
    # 1-D arrays are considered as column vectors
    return x.reshape((len(x), 1)) if len(x.shape) == 1 else x

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

def _dot_gpu_notransout(x, y, transa=False, transb=False, out=None):
    x = _as_mat(x)
    y = _as_mat(y)
    with cuda.using_cumisc():
        return cuda.culinalg.dot(x, y,
                transa=_as_trans_op(transa),
                transb=_as_trans_op(transb),
                out=out)

def _dot_gpu(x, y, transa=False, transb=False, transout=False, out=None):
    if transout:
        # (X Y)^T = Y^T X^T
        return _dot_gpu_notransout(y, x,
                transa=not transb, transb=not transa, out=out)
    else:
        return _dot_gpu_notransout(x, y, transa=transa, transb=transb, out=out)

class Dot(Function):
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
        gx0 = _dot_cpu(
                gy[0], x[1], transb=not self.transb, transout=self.transa
                ).reshape(x[0].shape)
        gx1 = _dot_cpu(
                x[0], gy[0], transa=not self.transa, transout=self.transb
                ).reshape(x[1].shape)
        return gx0, gx1

    def backward_gpu(self, x, gy):
        with cuda.using_cumisc():
            gx0 = _dot_gpu(
                    gy[0], x[1], transb=not self.transb, transout=self.transa
                    ).reshape(x[0].shape)
            gx1 = _dot_gpu(
                    x[0], gy[0], transa=not self.transa, transout=self.transb
                    ).reshape(x[1].shape)
            return gx0, gx1

def dot(x, y, transa=False, transb=False):
    """Compute dot product of two arrays.

    Args:
        x, y: Variables of 2-D or 1-D arrays.
            A 2-D array with shape (N, M) is considered as a NxM matrix.
            A 1-D array with shape (N,) is considered as a Nx1 matrix.
        transa (bool): If true, transpose x.
        transb (bool): If true, transpose y.

    Returns:
        ~chainer.Variable: Dot product of x and y as a 2-D array
    """
    return Dot(transa=transa, transb=transb)(x, y)

class BatchDot(Function):
    def __init__(self, transa=False, transb=False):
        self.transa = transa
        self.transb = transb

    def _output_shape(self, a, b):
        batch_size = a.shape[0]
        a_mat_shape = _as_mat(a[0]).shape
        b_mat_shape = _as_mat(b[0]).shape
        m = a_mat_shape[1] if self.transa else a_mat_shape[0]
        n = b_mat_shape[0] if self.transb else b_mat_shape[1]
        return (batch_size, m, n)

    def forward_cpu(self, x):
        x0, x1 = x
        assert x0.shape[0] == x1.shape[0]
        batch_size = x0.shape[0]
        shape = self._output_shape(x0, x1)
        ret = numpy.empty(shape)
        for i in xrange(batch_size):
            ret[i] = _dot_cpu(
                    x0[i], x1[i], transa=self.transa, transb=self.transb)
        return ret,

    def backward_cpu(self, x, gy):
        x0, x1 = x
        batch_size = x0.shape[0]
        gx0 = numpy.empty(x0.shape)
        gx1 = numpy.empty(x1.shape)
        for i in xrange(batch_size):
            gx0[i] = _dot_cpu(gy[0][i], x1[i],
                    transb=not self.transb, transout=self.transa
                    ).reshape(x0[0].shape)
            gx1[i] = _dot_cpu(x0[i], gy[0][i],
                    transa=not self.transa, transout=self.transb
                    ).reshape(x1[0].shape)
        return gx0, gx1

    def forward_gpu(self, x):
        x0, x1 = x
        assert x0.shape[0] == x1.shape[0]
        batch_size = x0.shape[0]
        shape = self._output_shape(x0, x1)
        ret = cuda.empty(shape)
        # TODO(muupan) batch computation for more efficiency
        for i in xrange(batch_size):
             _dot_gpu(x0[i], x1[i],
                     transa=self.transa, transb=self.transb, out=ret[i])
        return ret,

    def backward_gpu(self, x, gy):
        x0, x1 = x
        batch_size = x0.shape[0]
        gx0 = cuda.empty((batch_size,) + _as_mat(x0[0]).shape)
        gx1 = cuda.empty((batch_size,) + _as_mat(x1[0]).shape)
        # TODO(muupan) batch computation for more efficiency
        for i in xrange(batch_size):
            _dot_gpu(gy[0][i], x1[i],
                    transb=not self.transb, transout=self.transa, out=gx0[i])
            _dot_gpu(x0[i], gy[0][i],
                    transa=not self.transa, transout=self.transb, out=gx1[i])
        gx0 = gx0.reshape(x0.shape)
        gx1 = gx1.reshape(x1.shape)
        return gx0, gx1

def batchdot(x, y, transa=False, transb=False):
    """Compute dot product of two arrays in a batch manner.

    Args:
        x, y: Variables of 3-D or 2-D arrays.
            A 3-D array of shape (B, N, M) is considered as B NxM matrices.
            A 2-D array of shape (B, N,) is considered as B Nx1 matrices.
        transa (bool): If true, transpose each matrix in x.
        transb (bool): If true, transpose each matrix in y.

    Returns:
        ~chainer.Variable: Batch of dot products of two matrices from x and y in
            a 3-D array.
    """
    return BatchDot(transa=transa, transb=transb)(x, y)
