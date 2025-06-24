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
    OtterTensor* weights;
    OtterTensor* biases;
    int num_neurons;
    char* activation_function;
}Dense_layer;

typedef struct Conv1D_layer{
    int filter;
    int stride;
    int padding;
    OtterTensor* weights;
    OtterTensor* biases;
    int num_neurons;
    char* activation_function;
} Conv1D_layer;

typedef struct Dense_network {
    Dense_layer** layers;
    int num_layers;
    int optimizer;
    char* error_function;
    float learning_rate;
    float* optimizer_params; // Parameters for the optimizer, if needed
}Dense_network;

typedef struct Otterchain {
    Otterchain* next;
    void* layer;
    int type; // 0 for Dense, 1 for Conv1D
} Otterchain;

typedef struct Otternetwork {
    Otterchain* layers;
    int num_layers;
    int optimizer;
    char* error_function;
    float learning_rate;
    float* optimizer_params; // Parameters for the optimizer, if needed
} Otternetwork;


typedef struct Activation_function{
    const char* name;
    void (*activation)(OtterTensor*);
    void (*derivative)(OtterTensor*);
} Activation_function;


Dense_network* ON_initialise_network(int* dense_layers,int num_layers,char** activation_functions);
void ON_display_network(Dense_network* network);
void free_net(Dense_network* network);
int get_full_size_of_DN(Dense_network* network);

OtterTensor* ON_feed_forward(Dense_network* network, OtterTensor* input,OtterTensor** zs,OtterTensor** activations);

OtterTensor* layer_calc(OtterTensor* input, Dense_layer* layer,OtterTensor* zs,OtterTensor* activation);

OtterTensor* ON_predict(Dense_network* network, OtterTensor* input);
OtterTensor* ON_Cost_derivative(OtterTensor* output, OtterTensor* labels, char* error_function);
OtterTensor* ON_cost(OtterTensor* output, OtterTensor* labels, char* error_function);
OtterTensor** ON_copy_weights(Dense_network* network);
OtterTensor** ON_copy_biases(Dense_network* network);
OtterTensor*** ON_init_params(Dense_network* network);
void ON_update_weights_and_biases(Dense_network* network, OtterTensor** weights_gradients, OtterTensor** biases_gradients);
void ON_display_weights(Dense_network* network);
void ON_fit(Dense_network* network, OtterDataset* inputs, OtterDataset* labels, int epochs,int batch_size);
void ON_compile_network(Dense_network* network, char* optimizer, char* error_function, float learning_rate, float* optimizer_params);
void derivative_activation_functions(char* activation_function, OtterTensor* zs);

void free_params(OtterTensor*** params, int num_layers);
OtterTensor*** ON_init_grads(Dense_network* network);

#endif