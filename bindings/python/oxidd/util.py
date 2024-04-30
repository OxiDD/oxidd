"""Primitives and utilities"""

__all__ = ["Assignment"]

from collections.abc import Iterator, Sequence
from typing import Optional, Union

from _oxidd import ffi as _ffi
from _oxidd import lib as _lib
from typing_extensions import Never, Self, overload, override

#: CFFI allocator that does not zero the newly allocated region
_alloc = _ffi.new_allocator(should_clear_after_alloc=False)


class Assignment(Sequence[Optional[bool]]):
    """Boolean Assignment returned by an FFI function"""

    _data: ...  #: Wrapped oxidd_assignment_t

    def __init__(self, _: Never):
        """Private constructor

        Assignments cannot be instantiated directly, they are only returned by
        FFI functions.
        """
        raise RuntimeError(
            "Assignments cannot be instantiated directly, they are only "
            "returned by FFI functions."
        )

    @classmethod
    def _from_raw(cls, raw) -> Self:
        """Create an assignment from a raw FFI object (``oxidd_assignment_t``)"""
        assignment = cls.__new__(cls)
        assignment._data = raw
        return assignment

    def __del__(self):
        _lib.oxidd_assignment_free(self._data)

    @override
    def __len__(self) -> int:
        return int(self._data.len)

    def _get_unchecked(self, index: int) -> Optional[bool]:
        """Get the element at ``index`` without bounds checking

        SAFETY: ``index`` must be in bounds (``0 <= index < len(self)``)
        """
        v = int(self._data.data[index])
        return bool(v) if v >= 0 else None

    @override
    @overload
    def __getitem__(self, index: int) -> Optional[bool]: ...

    @override
    @overload
    def __getitem__(self, index: slice) -> list[Optional[bool]]: ...

    @override
    def __getitem__(
        self, index: Union[int, slice]
    ) -> Union[Optional[bool], list[Optional[bool]]]:
        n = len(self)
        if isinstance(index, slice):
            start, stop, step = index.indices(n)
            return [self._get_unchecked(i) for i in range(start, stop, step)]

        i = index if index >= 0 else n + index
        if i < 0 or i >= n:
            raise IndexError("Assignment index out of range")
        return self._get_unchecked(i)

    @override
    def __iter__(self) -> Iterator[Optional[bool]]:
        return (self._get_unchecked(i) for i in range(len(self)))

    @override
    def __reversed__(self) -> Iterator[Optional[bool]]:
        return (self._get_unchecked(i) for i in range(len(self) - 1, -1, -1))
