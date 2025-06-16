#ifndef OTTERTENSORS_RANDOM_H
#define OTTERTENSORS_RANDOM_H
#include "ottertensors.h"
#include "ottertensors_utilities.h"
#include <stdlib.h>
#include <time.h>


OtterTensor* OT_random_uniform(int* dims, int rank,float min,float max);

int* OR_select_batch(int total_size, int batch_size) ;


#endif