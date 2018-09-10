import numpy

import chainerx


# TODO(hvy): Define this function with other similar chainerx-numpy compatibility functions.
def _as_numpy_dtype(dtype):
    if isinstance(dtype, chainerx.dtype):
        return dtype.name
    return dtype


# TODO(sonots): Support subclassing
asanyarray = chainerx.asarray


def loadtxt(
        fname, dtype=float, comments='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0,
        encoding='bytes', device=None):
    if isinstance(dtype, chainerx.dtype):
        dtype = dtype.name
    return chainerx.array(numpy.loadtxt(
        fname, dtype=dtype, comments=comments, delimiter=delimiter, converters=converters, skiprows=skiprows, usecols=usecols,
        unpack=unpack, ndmin=ndmin, encoding=encoding), device=device)


# TODO(hvy): Optimize with pre-allocated memory using count for non-native devices.
def fromiter(iterable, dtype, count=-1, device=None):
    return chainerx.array(numpy.fromiter(iterable, dtype=_as_numpy_dtype(dtype), count=count), device=device)


def fromstring(string, dtype=float, count=-1, sep='', device=None):
    # sep should always be specified in numpy.fromstring since its default argument has been deprecated since 1.14.
    # https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.fromstring.html
    return chainerx.array(numpy.fromstring(string, dtype=_as_numpy_dtype(dtype), count=count, sep=sep), device=device)


def fromfile(file, dtype=float, count=-1, sep='', device=None):
    return chainerx.array(numpy.fromfile(file, dtype=_as_numpy_dtype(dtype), count=count, sep=sep), device=device)


def fromfunction(function, shape, **kwargs):
    dtype = kwargs.pop('dtype', float)
    device = kwargs.pop('device', None)
    return chainerx.array(numpy.fromfunction(function, shape, dtype=_as_numpy_dtype(dtype), **kwargs), device=device)