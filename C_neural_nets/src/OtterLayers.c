#include "../header/OtterLayers.h"

Otterchain* ON_Dense_layer(int neurons, char* activation_function) {
    Otterchain* chain = malloc(sizeof(Otterchain));
    Dense_layer* layer = malloc(sizeof(Dense_layer));
    chain->weights = malloc(sizeof(OtterTensor*));
    chain->biases = malloc(sizeof(OtterTensor*));
    chain->weights_depth = 1;
    chain->layer = layer;
    layer->num_neurons = neurons;
    layer->activation_function = strdup(activation_function);
    chain->type = 0; 

    return chain;
}

void ON_compile_Dense_layer(Otterchain* layer) { // faut gérer les connections avant et arriere
    int input_dims = 0;
    if (layer->num_connections_backward != 1) {
        fprintf(stderr, "Dense layer compilation : Dense can only take one input connection, received %i\n", layer->num_connections_backward);
        exit(EXIT_FAILURE);
    }
    Otterchain* previous_layer = layer->connections_backward[0];
    if (previous_layer->type == 0) {
        input_dims = ((Dense_layer*)previous_layer)->num_neurons; // Dense layer

    } else if (previous_layer->type == 2) {
        input_dims = ((Flatten_layer*)previous_layer->layer)->output_size; // Flatten layer
    }
    else {
        fprintf(stderr, "Unknown layer type for Dense layer compilation.\n");
        exit(EXIT_FAILURE);
    }
    int dims_w[2] = {((Dense_layer*)layer->layer)->num_neurons, input_dims};
    int dims_b[2] = {((Dense_layer*)layer->layer)->num_neurons, 1};
    layer->weights[0] = OT_random_uniform(dims_w, 2, -1.0f, 1.0f);
    layer->biases[0] = OT_zeros(dims_b, 2);
    for(int i = 0; i < layer->num_connections_backward; i++) {
        layer->connections_backward[i]->connections_forward = realloc(layer->connections_backward[i]->connections_forward, sizeof(Otterchain*) * (layer->connections_backward[i]->num_connections_forward + 1));
        layer->connections_backward[i]->connections_forward[layer->connections_backward[i]->num_connections_forward] = layer;
        layer->connections_backward[i]->num_connections_forward++;
    }

    layer->input_dims = malloc(2 * sizeof(int));
    layer->input_dims[0] = input_dims;
    layer->input_dims[1] = 1;
    layer->output_dims = malloc(2 * sizeof(int));
    layer->output_dims[0] = ((Dense_layer*)layer->layer)->num_neurons;
    layer->output_dims[1] = 1;
    
    layer->weights_depth= 1;

}

OtterTensor* ON_Dense_layer_forward(Otterchain* chain,OtterTensor* input, int gradient_register) {
    if(gradient_register) {
        chain->input = malloc(sizeof(OtterTensor*));
        chain->input[0] = OT_copy(input);
    }
    OtterTensor* prod = OT_Matrix_multiply(chain->weights[0], input);
    OT_ref_tensors_sum(prod, chain->biases[0]);
    if (gradient_register) {
        chain->pre_activation = OT_copy(prod); 
    }
    Activation_functions(((Dense_layer*)chain->layer)->activation_function, prod);
    if (gradient_register) chain->post_activations = OT_copy(prod);
    return prod;
}

OtterTensor* ON_Dense_layer_backward(Otternetwork* network, Otterchain* chain, OtterTensor* input, int layer_number) {
    OtterTensor* error = OT_copy(chain->connections_forward[0]->local_errors);
    derivative_activation_functions(((Dense_layer*)chain->layer)->activation_function, chain->post_activations);
    
    OtterTensor* dZ = OT_dot(error, chain->post_activations);
    free_malloc_tensor(error);
    OtterTensor* W_next_T = OT_Transpose(chain->weights[0]);
    chain->local_errors = OT_Matrix_multiply(W_next_T, dZ);
    free_malloc_tensor(W_next_T);
    
    OtterTensor* X_T = OT_Transpose((chain->input)[0]);
    chain->weights_gradients[0] = OT_Matrix_multiply(dZ, X_T);
    free_malloc_tensor(X_T);
    
    chain->biases_gradients[0]= OT_column_sum(dZ);
    free_malloc_tensor(dZ);

    return chain->local_errors;
}



void free_Dense_layer(Dense_layer* layer) {
    if (layer == NULL) return;
    if (layer->activation_function) {
        free(layer->activation_function);
    }
    free(layer);
}




//////////////////////////////////////////////////





Otterchain* ON_Conv1D_layer(int kernet_size,int filter, int stride, int padding, int neurons, char* activation_function) {
    Otterchain* chain = malloc(sizeof(Otterchain));
    Conv1D_layer* layer = malloc(sizeof(Conv1D_layer));
    layer->filter = filter;
    layer->kernel_size = kernet_size;
    layer->stride = stride;
    layer->padding = padding;
    chain->weights = malloc(filter * sizeof(OtterTensor*));
    chain->biases = NULL;
    layer->num_neurons = neurons;
    layer->activation_function = strdup(activation_function);
    chain->type = 1; 
    chain->layer = layer;
    return chain;
}

void ON_compile_Conv1D_layer(Otterchain* chain, int input_length) {
    /*
    for(int i = 0; i < ((Conv1D_layer*)chain->layer)->filter; i++) {
        ((Conv1D_layer*)chain->layer)->weights[i] = OT_random_uniform((int[2]){1,((Conv1D_layer*)chain->layer)->kernel_size}, 2, -1.0f, 1.0f);
    }
    ((Conv1D_layer*)chain->layer)->biases = OT_zeros((int[2]){((Conv1D_layer*)chain->layer)->kernel_size,1}, 2);
    
    chain->weights_depth = ((Conv1D_layer*)chain->layer)->filter;
    chain->input_dims = malloc(2 * sizeof(int));
    chain->output_dims = malloc(2 * sizeof(int));
    chain->input_dims[0] = input_length;
    chain->input_dims[1] = 1;

    */
}

/*
OtterTensor* ON_Conv1D_layer_forward(Otterchain* chain,OtterTensor* input, int gradient_register) { //à travailler
    
    int N=0;
    OtterTensor* prod = OT_zeros((int[2]){}, 2);
    OtterTensor** slice=OT_slice_padding(input, layer->filter, layer->stride, layer->padding);
    
    if(layer->padding){
        N = input->dims[0];
    }else{
        N= input->dims[0] - layer->filter + 1;
    }

    
    return prod;
}

void free_Conv1D_layer(Conv1D_layer* layer) {
    if (layer == NULL) return;
    for (int i = 0; i < layer->filter; i++) {
        free_malloc_tensor(layer->weights[i]);
    }
    free(layer->weights);
    free_malloc_tensor(layer->biases);
    if (layer->activation_function) {
        free(layer->activation_function);
    }
    free(layer);
}


/////////////////////////////////////////////////



Flatten_layer* ON_Flatten_layer(int neurons, char* activation_function) {
    Flatten_layer* layer = malloc(sizeof(ON_Flatten_layer));
    layer->type = 2; 
    return layer;
}

void ON_compile_Flatten_layer(Otterchain* layer) {
    switch(layer->connections[0]->type) {
        case 0: // Dense layer
            ((Flatten_layer*)layer->layer)->output_size = ((Dense_layer*)layer->connections[0]->layer)->num_neurons;
            break;
        case 1 : 
            ((Flatten_layer*)layer->layer)->output_size = ((Conv1D_layer*)layer->connections[0]->layer)->output_dims[0]* ((Conv1D_layer*)layer->connections[0]->layer)->output_dims[1];
            break;
        case 2: // Flatten layer
            ((Flatten_layer*)layer->layer)->output_size = ((Flatten_layer*)layer->connections[0]->layer)->output_size;
            break;
        default:
            fprintf(stderr, "Unknown layer type for Flatten layer compilation.\n");
            exit(EXIT_FAILURE);
    }
    return;
}

OtterTensor* ON_Flatten_layer_forward(Dense_layer* layer, OtterTensor* input, OtterTensor** zs, OtterTensor** activations) {
    OtterTensor* prod = OT_Flatten(input);
    return prod;
}

*/