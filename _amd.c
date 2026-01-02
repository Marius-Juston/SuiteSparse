#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <math.h>

#include "amd.h"

static inline int read_numeric(const Py_buffer *view, const char *ptr, double *out) {
    if (strcmp(view->format, "f") == 0) {
        *out = *(float *) ptr;
    } else if (strcmp(view->format, "d") == 0) {
        *out = *(double *) ptr;
    } else if (strcmp(view->format, "b") == 0) {
        *out = *(int8_t *) ptr;
    } else if (strcmp(view->format, "B") == 0) {
        *out = *(uint8_t *) ptr;
    } else if (strcmp(view->format, "h") == 0) {
        *out = *(int16_t *) ptr;
    } else if (strcmp(view->format, "H") == 0) {
        *out = *(uint16_t *) ptr;
    } else if (strcmp(view->format, "i") == 0) {
        *out = *(int32_t *) ptr;
    } else if (strcmp(view->format, "I") == 0) {
        *out = *(uint32_t *) ptr;
    } else if (strcmp(view->format, "l") == 0) {
        *out = *(long *) ptr;
    } else if (strcmp(view->format, "q") == 0) {
        *out = *(int64_t *) ptr;
    } else if (strcmp(view->format, "?") == 0) {
        *out = *(char *) ptr;
    } else {
        PyErr_Format(PyExc_TypeError, "Unsupported buffer format '%s' is unrecognized", view->format);

        return 0;
    }

    return isfinite(*out);
}

static inline int copy_array(const int32_t *array, const size_t size, PyObject *list) {
    for (size_t i = 0; i < size; i++) {
        PyObject *idx = PyLong_FromLong(array[i]);

        if (!idx) {
            return 0;
        }

        PyList_SET_ITEM(list, i, idx);
    }

    return 1;
}

static PyObject *_amd(PyObject *self, PyObject *args) {
    PyObject *obj;
    Py_buffer view;

    if (!PyArg_ParseTuple(args, "O", &obj)) {
        return NULL;
    }

    if (PyObject_GetBuffer(obj, &view, PyBUF_FORMAT | PyBUF_STRIDED | PyBUF_ND) == -1) {
        return NULL;
    }

    if (view.ndim != 2) {
        PyErr_SetString(PyExc_ValueError, "Expected 2 dimensional arrays");
        PyBuffer_Release(&view);
        return NULL;
    }

    const Py_ssize_t rows = view.shape[0];
    const Py_ssize_t cols = view.shape[1];

    if (rows != cols) {
        PyErr_SetString(PyExc_ValueError, "Expected a square matrix with equal number of rows and columns");
        goto fail;
    }

    const int32_t n = (int32_t) rows;

    const char *data = view.buf;
    int32_t count = 0;
    double value;

    int32_t *Ap = PyMem_Calloc(rows + 1, sizeof(int32_t));
    if (!Ap) {
        PyErr_NoMemory();
        goto fail;
    }
    Ap[0] = 0;

    for (Py_ssize_t i = 0; i < rows; i++) {
        for (Py_ssize_t j = 0; j < cols; j++) {
            const char *ptr = data + i * view.strides[0]
                              + j * view.strides[1];

            if (!read_numeric(&view, ptr, &value)) {
                PyErr_SetString(PyExc_ValueError, "Unsupported dtype or NaN/Inf encountered");
                goto fail;
            }

            if (value != 0) {
                ++count;
            }
        }

        Ap[i + 1] = count;
    }

    int32_t *Ai = PyMem_Calloc(count, sizeof(int32_t));
    if (!Ai) {
        PyErr_NoMemory();
        goto fail;
    }

    size_t index = 0;

    for (Py_ssize_t i = 0; i < rows; i++) {
        for (Py_ssize_t j = 0; j < cols; j++) {
            const char *ptr = data + i * view.strides[0]
                              + j * view.strides[1];

            if (!read_numeric(&view, ptr, &value)) {
                PyErr_SetString(PyExc_ValueError, "Unsupported dtype or NaN/Inf encountered");
                goto fail;
            }

            if (value != 0) {
                Ai[index++] = (int32_t) j;
            }

            if (index >= count) {
                goto loop_exit;
            }
        }
    }

loop_exit:

    int *version = PyMem_Calloc(3, sizeof(int));
    if (!version) {
        PyErr_NoMemory();
        goto fail;
    }

    amd_version(version);


    double *Control = PyMem_Calloc(AMD_CONTROL, sizeof(double));
    if (!Control) {
        PyErr_NoMemory();
        goto fail;
    }

    amd_defaults(Control);
    amd_control(Control);

    double *Info = PyMem_Calloc(AMD_INFO, sizeof(double));
    if (!Info) {
        PyErr_NoMemory();
        goto fail;
    }
    int32_t *P = PyMem_Calloc(n, sizeof(int32_t));
    if (!P) {
        PyErr_NoMemory();
        goto fail;
    }

    const int32_t result = amd_order(n, Ap, Ai, P, Control, Info);

    amd_info(Info);

    PyObject *permutation = PyList_New(n);
    if (!permutation) {
        goto fail;
    }

    if (!copy_array(P, n, permutation)) {
        goto fail;
    }

    // PyObject *out = PyTuple_New(2);
    // if (!out) {
    //     goto fail;
    // }
    //
    // PyObject *csc = PyList_New(rows + 1);
    // if (!copy_array(Ap, rows + 1, csc)) {
    //     goto fail;
    // }
    //
    // PyObject *res = PyList_New(count);
    // if (!copy_array(Ai, count, res)) {
    //     goto fail;
    // }
    //
    // PyTuple_SET_ITEM(out, 0, csc);
    // PyTuple_SET_ITEM(out, 1, res);

    PyMem_Free(Ap);
    PyMem_Free(Ai);

    PyMem_Free(version);
    PyMem_Free(Control);
    PyMem_Free(Info);
    PyMem_Free(P);

    PyBuffer_Release(&view);


    return permutation;

fail:
    PyBuffer_Release(&view);
    return NULL;
}

static struct PyMethodDef methods[] = {
    {"amd", (PyCFunction) _amd, METH_VARARGS},
    {NULL, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_amd",
    NULL,
    -1,
    methods
};


PyMODINIT_FUNC PyInit__amd(void) {
    return PyModule_Create(&module);
}
