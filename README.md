# SuiteSparse-AMD

This package is the port of the SuiteSparse AMD (Approximate Minimum Degree) function. This is a Python C wrapper of the library from [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse).

This package currently only works with Numpy arrays and 2D lists.

## Future

- Work with PyTorch CPU Tensors

# Source Installation

```bash
pip install git+https://github.com/Marius-Juston/SuiteSparse.git
```


# Compile Source

```bash
python3 -m build --wheel --sdist
```