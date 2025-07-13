#ifndef OTTERLAYERS_H
#define OTTERLAYERS_H
#include "otternet.h"
#include "ottertensors.h"
#include "ottertensors_utilities.h"
#include "ottertensors_operations.h"
#include "OtterActivation.h"
#include "ottertensors_random.h"
#include "ottermath.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>


struct Otternetwork; // Forward declaration of Otternetwork to use in Otterchain

typedef struct Otterchain {
    Otterchain* next;
    void* layer;
    int type; 
    Otterchain** connections_backward; 
    Otterchain** connections_forward; 
    int num_connections_backward; 
    int num_connections_forward; 

    OtterTensor** biases;
    OtterTensor** weights;

    OtterTensor** weights_gradients;
    OtterTensor** biases_gradients ;
    OtterTensor* local_errors;
    OtterTensor** input; 
    OtterTensor* pre_activation;               
    OtterTensor* post_activations;

    int weights_depth;
    int* input_dims; 
    int* output_dims; 

} Otterchain;

//////////////////////////////////////////////////////////////////////

typedef struct Dense_layer {
    int num_neurons;
    char* activation_function;
    OtterTensor* output;
    
}Dense_layer;

Otterchain* ON_Dense_layer(int neurons, char* activation_function);
void ON_compile_Dense_layer(Otterchain* layer);
OtterTensor* ON_Dense_layer_forward(Otterchain* chain,OtterTensor* input, int gradient_register);
OtterTensor* ON_Dense_layer_backward(Otternetwork* network, Otterchain* chain, OtterTensor* input, int layer_number);
void free_Dense_layer(Dense_layer* layer);


//////////////////////////////////////////////////////


typedef struct Conv1D_layer{
    int kernel_size; 
    int filter;
    int stride;
    int padding;
    int num_neurons;
    char* activation_function;
    int* input_dims[2]; // Dimensions of the input tensor
    int* output_dims[2]; // Dimensions of the output tensor
    OtterTensor* output; // Output tensor after convolution
    
} Conv1D_layer;
/*
Otterchain* ON_Conv1D_layer(int kernel_size, int filter, int stride, int padding, int neurons, char* activation_function);
void ON_compile_Conv1D_layer(Otterchain* layer, int input_dims);
OtterTensor* ON_Conv1D_layer_forward(Conv1D_layer* layer, OtterTensor* input, OtterTensor** zs, OtterTensor** activations);
OtterTensor* ON_Conv1D_layer_backward(Otternetwork* network, Otterchain* chain, OtterTensor* input, int layer_number);
void free_Conv1D_layer(Conv1D_layer* layer);
*/
//////////////////////////////////////////////////////:


typedef struct Flatten_layer {
    int output_size;
    int type;
} Flatten_layer;
/*
Flatten_layer* ON_Flatten_layer(int neurons, char* activation_function);
void ON_compile_Flatten_layer(Otterchain* layer);
OtterTensor* ON_Flatten_layer_forward(Dense_layer* layer, OtterTensor* input, OtterTensor** zs, OtterTensor** activations);
OtterTensor* ON_Flatten_layer_backward(Otternetwork* network, Otterchain* chain, OtterTensor* input, int layer_number);
void free_Flatten_layer(Flatten_layer* layer);
*/











#endif