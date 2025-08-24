#ifndef OTTERNET_H
#define OTTERNET_H

#include "ottertensors.h"
#include "otternet_optimizers.h"
#include "OtterLayers.h"
#include "OtterActivation.h"
#include "ottertensors_utilities.h"
#include "ottertensors_operations.h"
#include "ottertensors_random.h"
#include "otternet_utilities.h"
#include "ottermath.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>


typedef struct Otterchain Otterchain;




typedef struct Otternetwork {
    Otterchain* layers;
    int num_layers;
    int optimizer;
    char* error_function;
    float learning_rate;
    float* optimizer_params; // Parameters for the optimizer, if needed
    Otterchain* end;
    Otterchain* start;

    Otterchain** end_of_line;
    int num_end_of_line; // Number of layers with no forward connections
    Otterchain** start_of_line;
    int num_start_of_line; // Number of layers with no backward connections

    Otterchain** order;
    OtterTensor** input;
    OtterTensor** output;

    OtterTensor** errors;
} Otternetwork;



Otternetwork* ON_initialise_otternetwork();
void ON_add_layer(Otternetwork* network, Otterchain* new_layer);
void ON_compile_otternetwork(Otternetwork* network, char* optimizer, char* error_function, float learning_rate, float* optimizer_params);
OtterTensor** ON_feed_forward(Otternetwork* network, OtterTensor** input, int gradient_register);


OtterTensor** ON_predict(Otternetwork* network, OtterTensor** input);
OtterTensor* ON_Cost_derivative(OtterTensor* output, OtterTensor* labels, char* error_function);
float ON_cost(OtterTensor* output, OtterTensor* labels, char* error_function);

void ON_update_weights_and_biases(Otternetwork* network);

//void ON_display_weights(Dense_network* network);
void ON_fit(Otternetwork* network, OtterDataset* inputs, OtterDataset* labels, int epochs, int batch_size);
void derivative_activation_functions(char* activation_function, OtterTensor* zs);


Otterchain** calculate_distances_ordered(Otternetwork* net) ;

void free_otternetwork(Otternetwork* network);
void free_otterchain(Otterchain* chain);

void ON_reset_network(Otternetwork* network) ;

void ON_reset_layer(Otterchain* chain) ;
void ON_free_layer(Otterchain* layer);
#endif