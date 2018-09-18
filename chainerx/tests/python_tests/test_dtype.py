import math

import numpy
import pytest

import chainerx
import chainerx.testing


def test_py_types():
    assert chainerx.bool is bool
    assert chainerx.int is int
    assert chainerx.float is float


def test_dtype_from_python_type():
    assert chainerx.dtype('bool') == chainerx.bool_
    assert chainerx.dtype('int') == chainerx.int64
    assert chainerx.dtype('float') == chainerx.float64
    assert chainerx.dtype(bool) == chainerx.bool_
    assert chainerx.dtype(int) == chainerx.int64
    assert chainerx.dtype(float) == chainerx.float64


@chainerx.testing.parametrize_dtype_specifier('dtype_spec', with_chainerx_dtypes=False)
def test_dtype_from_specifier(dtype_spec):
    assert chainerx.dtype(dtype_spec).name == numpy.dtype(dtype_spec).name


@pytest.mark.parametrize('dtype_symbol', chainerx.testing.all_dtypes)
def test_dtypes(dtype_symbol):
    dtype = getattr(chainerx, dtype_symbol)
    numpy_dtype = numpy.dtype(dtype_symbol)
    assert isinstance(dtype, chainerx.dtype)
    assert dtype.name == numpy_dtype.name
    assert dtype.char == numpy_dtype.char
    assert dtype.itemsize == numpy_dtype.itemsize
    assert dtype.kind == numpy_dtype.kind
    assert dtype.byteorder == numpy_dtype.byteorder
    assert dtype.str == numpy_dtype.str
    assert dtype.num == numpy_dtype.num
    assert chainerx.dtype(dtype.name) == dtype
    assert chainerx.dtype(dtype.char) == dtype
    assert chainerx.dtype(dtype) == dtype
    # From NumPy dtypes
    assert chainerx.dtype(numpy_dtype) == dtype


def test_eq():
    assert chainerx.int8 == chainerx.int8
    assert chainerx.dtype('int8') == chainerx.int8
    assert chainerx.dtype(chainerx.int8) == chainerx.int8
    assert not 8 == chainerx.int8
    assert not chainerx.int8 == 8
    assert not 'int8' == chainerx.int8
    assert not chainerx.int8 == 'int8'


def test_ne():
    assert chainerx.int32 != chainerx.int8
    assert chainerx.dtype('int32') != chainerx.int8
    assert chainerx.dtype(chainerx.int32) != chainerx.int8
    assert 32 != chainerx.int32
    assert chainerx.int8 != 32
    assert 'int32' != chainerx.int32
    assert chainerx.int8 != 'int32'


def test_implicity_convertible():
    chainerx.zeros(shape=(2, 3), dtype='int32')


@chainerx.testing.parametrize_dtype_specifier('dtype_spec')
@pytest.mark.parametrize('value', [
    -2,
    1,
    -1.5,
    2.3,
    True,
    False,
    numpy.array(1),
    float('inf'),
    float('nan'),
])
def test_type(dtype_spec, value):
    expected = chainerx.Scalar(value, dtype_spec)
    actual = chainerx.dtype(dtype_spec).type(value)
    assert isinstance(actual, chainerx.Scalar)
    assert actual.dtype == chainerx.dtype(dtype_spec)
    if math.isnan(expected):
        assert math.isnan(actual)
    else:
        assert expected == actual