#include <Python.h>
#include <numpy/arrayobject.h>
#include "addition.h"
#include <stdio.h>

static char module_docstring[] =
    "This module provides some simple addition functionality";

static char add_docstring[] =
    "Adds two integers together";

static char element_add_docstring[] =
	"Returns the elementwise sum of two arrays of equal size.";

static PyObject *add_addition(PyObject *self, PyObject *args);
static PyObject *element_add_addition(PyObject *self, PyObject *args);
static void print_array(int *array, int size);
static void test_element_add();

static PyMethodDef module_methods[] = {
    {"add", add_addition, METH_VARARGS, add_docstring},
    {"element_add", element_add_addition, METH_VARARGS, element_add_docstring},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC PyInit__addition(void)
{
    
    PyObject *module;
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_addition",
        module_docstring,
        -1,

        module_methods,
        NULL,
        NULL,
        NULL,
        NULL
    };
    module = PyModule_Create(&moduledef);
    if (!module) return NULL;

    /* Load `numpy` functionality. */
    import_array();

    return module;
}

static PyObject *add_addition(PyObject *self, PyObject *args)
{
    int num1;
    int num2;
    int sum;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "ii", &num1, &num2))
        return NULL;

    /* Call the external C function to compute the chi-squared. */
    sum = add(num1, num2);

    /* Build the output tuple */
    PyObject *ret = Py_BuildValue("i", sum);
    return ret;
}

static PyObject *element_add_addition(PyObject *self, PyObject *args) {
	int* array1;
	int* array2;
	PyObject* obj1;
	PyObject* obj2;
	PyArrayObject* cobraArray;
	int size;

	/* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "OO", &obj1, &obj2))
        return NULL;

    /* Interpret input as numpy arrays */
    PyObject* firstArray = PyArray_FROM_OTF(obj1, NPY_INT, NPY_IN_ARRAY);
	PyObject* secondArray = PyArray_FROM_OTF(obj2, NPY_INT, NPY_IN_ARRAY);

	/* Throw an exception if it didn't work */
	if (firstArray == NULL || secondArray == NULL) {
		Py_XDECREF(firstArray);
		Py_XDECREF(secondArray);
		return NULL;
	}	

	/* Ensuring the two arrays of the same size. */
	if ((int)PyArray_DIM(firstArray, 0) == 
			(int)PyArray_DIM(secondArray, 0)) {
		size = (int) PyArray_DIM(firstArray, 0);	// Arbitrary which we choose
		printf("%d\n", size);
	}
	else {
		// The two arrays are not of the same size. Throw an exception
		return NULL;
	}

	// Getting pointers to the data as C data types
	array1 = (int*)PyArray_DATA(firstArray);
	array2 = (int*)PyArray_DATA(secondArray);
	print_array(array2, size);
	print_array(array1, size);

	// Creating the python array
	npy_intp dimension[1] = {size};
    printf("About to build array\n");
    cobraArray = (PyArrayObject*) PyArray_FromDims(1, dimension, 'd');
    Py_INCREF(cobraArray);
    PyArray_ENABLEFLAGS((PyArrayObject*)cobraArray, NPY_ARRAY_OWNDATA);
	
    // Obtaining a C pointer to our python array
    int* result_array = (int*) cobraArray->data;

	// Calling the function
	element_add(array1, array2, result_array, size);

	print_array((int*)cobraArray->data, size);

	// Cleaning Up
	Py_DECREF(array1);
	Py_DECREF(array2);
	free(result_array);

	printf("Made It\n");
	return PyArray_Return(cobraArray);
}

static void print_array(int *array, int size) {
	int i = 0;
	printf("{");
	for(; i < size; i++) {
		printf("%d, ", array[i]);
	}
	printf("}\n");
}