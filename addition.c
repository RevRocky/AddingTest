#include <stdlib.h>
#include <stdio.h>
#include "addition.h"

/*
 * A simple adding function. Meant to test making python
 * extensions in C
 */
int add(int num1, int num2) {
	return num1 + num2;
}

/*
 * Adds each correspoding element of two arrays into a third
 * array.
 * Result Array is a C pointer to a numpy array.
 */
void element_add(int* array1, int* array2, int* result_array, int size) {
	int i;
	for (i = 0; i < size; i++) {
		result_array[i] = array1[i] + array2[i];
		printf("%d\n", result_array[i]);
	}
}