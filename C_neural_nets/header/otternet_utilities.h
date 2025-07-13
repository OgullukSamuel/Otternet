#ifndef OTTERNET_UTILITIES_H
#define OTTERNET_UTILITIES_H
#include "../header/ottertensors.h"
#include "../header/ottertensors_utilities.h"
#include "../header/ottertensors_operations.h"
#include "../header/ottertensors_random.h"
#include "../header/ottermath.h"
#include "../header/OtterLayers.h"



int get_layer_type(void* layer);
int find_index(Otterchain** list, int size, Otterchain* target);
int argmin(int* distances, int size);
void rankify(int* input, int* output, int size);












#endif