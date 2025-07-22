#ifndef OTTERTENSORS_UTILITIES_H
#define OTTERTENSORS_UTILITIES_H
#include "ottertensors.h"
#include <stdio.h>
#include <stdlib.h>



void print_tensor(OtterTensor* t, int significant_digits);
void print_tensor_recursive(OtterTensor* t, int level, int ndims,int idx,int significant_digits);
OtterTensor* OT_copy(OtterTensor* a);
void OT_initialize_copy(OtterTensor* a, OtterTensor* copy);
OtterTensor* OT_Flatten(OtterTensor* t);
OtterTensor* OT_zeros(int* dims, int rank);
OtterTensor* OT_ones(int* dims, int rank);



#endif