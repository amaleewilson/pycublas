#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ndarrayobject.h>
#include "cusgemm.h"


static char module_docstring[] =
  "This module wraps Nvidia's cuBLAS sgamm.";


static char cusgemm_docstring[] =
  "Calculate SGEMM using cuBLAS.";


static PyObject *cusgemm_cusgemm(PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
  {"cusgemm", cusgemm_cusgemm, METH_VARARGS, cusgemm_docstring},
  {NULL, NULL, 0, NULL}
};


PyMODINIT_FUNC init_cusgemm(void) {
    PyObject *m = Py_InitModule3("_cusgemm", module_methods, module_docstring);
    if (m == NULL)
        return;

    import_array();
}

static PyObject *cusgemm_cusgemm(PyObject *self, PyObject *args) {

  PyObject *a_obj, *b_obj;

  /* parse function input args */
  if (!PyArg_ParseTuple(args, "OO", &a_obj, &b_obj))
    return NULL;

  /* we expect input to be numpy arrays */
  PyObject *a_array = PyArray_FROM_OF(a_obj, NPY_ARRAY_F_CONTIGUOUS);
  PyObject *b_array = PyArray_FROM_OF(b_obj, NPY_ARRAY_F_CONTIGUOUS);

  /* array's dimensionality */
  int m = (int) PyArray_DIM(a_array, 0);
  int n = (int) PyArray_DIM(b_array, 1);
  int k = (int) PyArray_DIM(a_array, 1);

  /* TODO: check common dimension, so matrix multiplication is possible */

  /* get pointers to the raw data */
  float *a = (float*)PyArray_DATA(a_array);
  float *b = (float*)PyArray_DATA(b_array);

  /* allocate memory for results */
  float *c = (float *) malloc( m * n * sizeof(float));

  /* call cublas */
  cu_sgemm(a, b, c, m, n, k);


  /* return numpy array as a response */
  long int dims[2];
  dims[0] = m;
  dims[1] = n;

  PyArray_Descr* descr = PyArray_DescrFromType(NPY_FLOAT);
  PyObject *ret = PyArray_NewFromDescr(&PyArray_Type, descr, 2, dims, NULL, (void *) c, NPY_ARRAY_F_CONTIGUOUS, 0);
  return ret;
}
