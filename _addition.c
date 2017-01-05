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
	int* resultArray;
	PyObject* obj1;
	PyObject* obj2;
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

	// Calling the function
	resultArray = element_add(array1, array2, size);

	// Cleaning Up
	Py_DECREF(array1);
	Py_DECREF(array2);

	/* Building the output */
    npy_int dimension = {size}

	PyObject* pyArray = PyArray_SimpleNew(size, dimension, NPY_INT);
	Py_INCREF(pyArray);

	printf("Made It\n");
	// Building Output Array
	int* array_buffer = (int*)PyArray_DATA(pyArray);
	int i;
	for (i = 0; i < size; i++) {
		*array_buffer++ = *resultArray++;
		printf("%d, ", array_buffer[i]);
	}

	return pyArray;
}

static void print_array(int *array, int size) {
	int i = 0;
	printf("{");
	for(; i < size; i++) {
		printf("%d, ", array[i]);
	}
	printf("}\n");
}

static void test_element_add() {
	printf("Testing Elementwise Addition\n");
	int my_array[] = {1, 2, 3};
	int* final;

	print_array(my_array, 3);
	final = element_add(my_array, my_array, 3);
	print_array(final, 3);
	printf("Done Testing Elementwise Addition\n");
}