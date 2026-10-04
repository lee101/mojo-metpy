"""The FFI contract: the compiled library is what the tests actually exercise.

Two things are checked here that nothing else can check:

- the ctypes signature table is exactly what `src/capi.mojo` declares, so a
  kernel whose signature changes cannot be called through a stale table; and
- `dist/libmojo-metpy.so` is really loaded.  A wrapper that quietly fell back to
  pure Python would make every other test in this suite pass while proving
  nothing about the port.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np
import pytest

from mojometpy import _lib


def test_signature_table_matches_the_mojo_source():
    """`_SIGNATURES` is regenerated from `src/capi.mojo`, not hand-maintained."""
    assert _lib._SIGNATURES == _lib.parse_signatures()


def test_every_int_argument_is_64_bit():
    """Mojo's `Int` is 64-bit; a 32-bit ctypes arg truncates buffer addresses.

    This is not hypothetical: with `c_int` every address above 4 GiB is silently
    cut, which turns the kernel into a wild write.  Assert the width directly.
    """
    for name, (argtypes, restype) in _lib._SIGNATURES.items():
        for argtype in argtypes:
            assert argtype is not ctypes.c_int, name
            assert argtype in (ctypes.c_int64, ctypes.c_double), name
        assert restype in (None, ctypes.c_int64), name


def test_library_is_actually_loaded():
    lib = _lib.lib()
    assert isinstance(lib, ctypes.CDLL)
    path = os.path.realpath(lib._name)
    assert os.path.exists(path), path
    # the exports are real symbols in that object, not Python fallbacks
    for name in _lib.parse_signatures():
        assert hasattr(lib, name), name


def test_a_kernel_result_comes_from_the_compiled_code():
    """A round trip that no pure-Python fallback could reproduce.

    `normalize_window` writes in place into the buffer it is handed, so the
    array the caller passed in must come back normalised.
    """
    lib = _lib.lib()
    window = np.ascontiguousarray(np.array([1.0, 1.0, 2.0]))
    lib.mm_normalize_window(int(window.ctypes.data), window.size)
    np.testing.assert_allclose(window, window / window.sum(), rtol=1e-15)
    assert window.sum() == pytest.approx(1.0, rel=1e-15)


def test_a_corrupt_buffer_does_not_pass_silently():
    """The kernels write through raw addresses, so check they are not no-ops.

    If the library were not really loaded the sentinel would survive; a real
    kernel overwrites it.
    """
    lib = _lib.lib()
    out = np.full(4, np.nan)
    src = np.ascontiguousarray(np.array([1.0, 2.0, 3.0, 4.0]))
    lib.mm_mixing_ratio(
        int(src.ctypes.data),
        int(np.full(4, 100000.0).ctypes.data),
        0.6219569100577033,
        4,
        int(out.ctypes.data),
    )
    assert np.all(np.isfinite(out))