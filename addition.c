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
 * Due to the calling environment we kno that the two arrays will
 * always be of the same size
 */
int* element_add(int* array1, int* array2, int size) {
	int* sumArray = malloc(size * sizeof(int));
	int i;

	for (i = 0; i < size; i++) {
		sumArray[i] = array1[i] + array2[i];
	}
	return sumArray;
}