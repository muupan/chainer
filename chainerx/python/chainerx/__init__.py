import chainerx.testing  # NOQA

from chainerx._core import *  # NOQA

from builtins import bool, int, float  # NOQA

from chainerx import _core

from chainerx.creation.from_data import asanyarray  # NOQA
from chainerx.creation.from_data import fromfile  # NOQA
from chainerx.creation.from_data import fromfunction  # NOQA
from chainerx.creation.from_data import fromiter  # NOQA
from chainerx.creation.from_data import fromstring  # NOQA
from chainerx.creation.from_data import loadtxt  # NOQA

_global_context = _core.Context()
_core.set_global_default_context(_global_context)