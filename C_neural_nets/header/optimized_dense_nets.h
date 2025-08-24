/* #ifndef OPTIMIZED_DENSE_NETS_H
#define OPTIMIZED_DENSE_NETS_H

#include "ottertensors.h"
#include "otternet_optimizers.h"
#include "ottertensors_utilities.h"
#include "ottertensors_operations.h"
#include "ottertensors_random.h"

#include "ottermath.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>



typedef struct Dense_network {
    Dense_layer** layers;
    int num_layers;
    int optimizer;
    char* error_function;
    float learning_rate;
    float* optimizer_params; // Parameters for the optimizer, if needed
}Dense_network;

Dense_network* ON_initialise_network(int* dense_layers, int num_layers, char** activation_functions);
void ON_compile_Dense_network(Dense_network* network, char* optimizer, char* error_function, float learning_rate, float* optimizer_params);















#endif */