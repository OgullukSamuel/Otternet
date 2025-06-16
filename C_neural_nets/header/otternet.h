#ifndef OTTERNET_H
#define OTTERNET_H

#include "ottertensors.h"
#include "ottertensors_utilities.h"
#include "ottertensors_operations.h"
#include "ottertensors_random.h"
#include "ottermath.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

typedef struct Dense_layer {
    OtterTensor weights;
    OtterTensor biases;
    int num_neurons;
    char* activation_function;

}Dense_layer;

typedef struct Dense_network {
    Dense_layer** layers;
    int num_layers;
    char* optimizer;
    char* error_function;
    float learning_rate;
}Dense_network;


Dense_network* ON_initialise_network(int* dense_layers,int num_layers,char** activation_functions);
void ON_display_network(Dense_network* network);
void free_net(Dense_network* network);
int get_full_size_of_DN(Dense_network* network);

OtterTensor* ON_feed_forward(Dense_network* network, OtterTensor* input,OtterTensor** zs,OtterTensor** activations);

OtterTensor* layer_calc(OtterTensor* input, Dense_layer* layer,OtterTensor* zs,OtterTensor* activation);

OtterTensor* ON_predict(Dense_network* network, OtterTensor* input);
OtterTensor* ON_Cout(OtterTensor* predictions, OtterTensor* labels, char* error_function);
OtterTensor** ON_copy_weights(Dense_network* network);
OtterTensor** ON_copy_biases(Dense_network* network);


void ON_fit(Dense_network* network, OtterDataset* inputs, OtterDataset* labels, int epochs,int batch_size);
void ON_SGD(Dense_network* network,OtterTensor* input, OtterTensor* labels);
void ON_compile_network(Dense_network* network, char* optimizer, char* error_function, float learning_rate);
void derivative_activation_functions(char* activation_function, OtterTensor* zs);



#endif