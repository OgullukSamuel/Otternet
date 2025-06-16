#ifndef OTTER_TENSORS_OPERATIONS_H
#define OTTER_TENSORS_OPERATIONS_H
#include "ottertensors.h"
#include "ottertensors_utilities.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


OtterTensor* OT_tensors_sum(OtterTensor* a, OtterTensor* b) ;
void OT_ref_tensors_sum(OtterTensor* a, OtterTensor* b);

void OT_ref_tensors_substract(OtterTensor* a, OtterTensor* b);

OtterTensor* OT_tensors_substract(OtterTensor* a, OtterTensor* b);
OtterTensor* OT_Matrix_multiply(OtterTensor* a, OtterTensor* b);
OtterTensor* OT_dot(OtterTensor* a, OtterTensor* b);



OtterTensor* OT_scalar_add(OtterTensor* main, float lambda);
OtterTensor* OT_scalar_subtract(OtterTensor* main, float lambda) ;
OtterTensor* OT_scalar_multiply(OtterTensor* main,float lambda);


OtterTensor* OT_Transpose(OtterTensor* t);



#endif