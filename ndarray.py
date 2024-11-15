from __future__ import annotations
import numpy as np

class ndarray_base(np.ndarray):
    def __new__(cls, input_array: np.ndarray, *, dtype=None, shape=None) -> 'ndarray_base':
        obj: 'ndarray_base' = np.asarray(input_array, dtype=dtype).view(cls)
        expected_ndim: int = cls._expected_ndim()
        if obj.ndim != expected_ndim:
            raise ValueError(f"{cls.__name__} requires {expected_ndim} dimensions, but got {obj.ndim}.")
        if shape is not None and obj.shape != shape:
            raise ValueError(f"{cls.__name__} requires shape {shape}, but got {obj.shape}.")
        expected_dtype = cls._expected_dtype()
        if expected_dtype is not None and obj.dtype != expected_dtype:
            raise TypeError(f"{cls.__name__} requires dtype '{expected_dtype}', but got '{obj.dtype}'.")
        return obj

    @classmethod
    def _expected_ndim(cls) -> int:
        raise NotImplementedError("_expected_ndim must be implemented by subclasses.")

    @classmethod
    def _expected_dtype(cls):
        return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({super().__repr__()})"


class ndarray2d(ndarray_base):
    @classmethod
    def _expected_ndim(cls) -> int:
        return 2


class ndarray3d(ndarray_base):
    @classmethod
    def _expected_ndim(cls) -> int:
        return 3


class ndarray4d(ndarray_base):
    @classmethod
    def _expected_ndim(cls) -> int:
        return 4

# Example usage:

# 2D array
a: ndarray2d = ndarray2d(np.random.rand(5, 5))
#print(a)  # correct 

# 3D array
b: ndarray3d = ndarray3d(np.random.rand(2, 3, 4))
#print(b) # correct 

# 4D array
# c: ndarray4d = ndarray4d(np.random.rand(4, 5)) # incorrect 

