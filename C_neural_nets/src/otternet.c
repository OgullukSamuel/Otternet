#include "../header/otternet.h"
#include "../header/otternet_optimizers.h"




Activation_function activation_table[] = {
    {"relu", OM_tensor_relu, OM_tensor_heaviside},
    {"sigmoid", OM_tensor_sigmoid, OM_tensor_dsigmoid},
    {"tanh", OM_tensor_tanh, OM_tensor_dtanh},
    {"softmax", OM_ref_softmax, NULL}, // gestion spéciale
    {"linear", OM_tensor_linear, OM_tensor_ones},
    {NULL, NULL, NULL}
};


void Activation_functions(char* function_name, OtterTensor* x) {
    for (int i = 0; activation_table[i].name != NULL; i++) {
        if (strcmp(function_name, activation_table[i].name) == 0) {
            if (activation_table[i].activation) activation_table[i].activation(x);
            return;
        }
    }
    fprintf(stderr, "Unknown activation function: %s\n", function_name);
    exit(EXIT_FAILURE);
}

void derivative_activation_functions(char* function_name, OtterTensor* x) {
    for (int i = 0; activation_table[i].name != NULL; i++) {
        if (strcmp(function_name, activation_table[i].name) == 0) {
            if (activation_table[i].derivative) activation_table[i].derivative(x);
            else for (int i = 0; i < x->size; i++) x->data[i] = 1.0f; // linear case
            return;
        }
    }
    fprintf(stderr, "Unknown activation function: %s\n", function_name);
    exit(EXIT_FAILURE);
}

int get_full_size_of_DN(Dense_network* network){
    int size=0;
    for(int i =0; i<network->num_layers; i++){
        size += network->layers[i]->weights->size + network->layers[i]->biases->size;
    }
    return(size);
}

Dense_network* ON_initialise_network(int* dense_layers,int num_layers,char** activation_functions){
    Dense_network* network = malloc(sizeof(Dense_network));
    network->layers = malloc(num_layers * sizeof(Dense_layer*));
    for(int i = 0; i < num_layers; i++) {
        network->layers[i] = malloc(sizeof(Dense_layer));
        int in_dim = (i == 0) ? dense_layers[0] : dense_layers[i-1];
        int out_dim = dense_layers[i];
        int dims_w[2] = {out_dim, in_dim};
        int dims_b[2] = {out_dim, 1};


        network->layers[i]->weights = OT_random_uniform(dims_w, 2, -1.0f, 1.0f);
        network->layers[i]->biases = OT_zeros(dims_b, 2);

        network->layers[i]->num_neurons = out_dim;
        network->layers[i]->activation_function = strdup(activation_functions[i]);
    }
    network->num_layers = num_layers;
    return network;
}

void ON_compile_network(Dense_network* network, char* optimizer, char* error_function, float learning_rate, float* optimizer_params) {
    network->error_function = strdup(error_function);
    network->learning_rate = learning_rate;
    if (strcmp(optimizer, "SGD") == 0) {
        network->optimizer= 0;
        network->optimizer_params = NULL; 
    } else if (strcmp(optimizer, "SGDM") == 0) {
        network->optimizer = 1;
        network->optimizer_params = optimizer_params; 
    } else if (strcmp(optimizer, "Adam") == 0) {
        network->optimizer = 2;
        network->optimizer_params = optimizer_params;
    } else {
        fprintf(stderr, "Unknown optimizer: %s\n", optimizer);
        exit(EXIT_FAILURE);
    }
    return;
}

void ON_display_network(Dense_network* network){
    int full_param = get_full_size_of_DN(network);
    printf("Network with %i layers, for a total of %i parameters \n",network->num_layers,full_param);
    printf("The network structure is the following : \n");
    printf("============================================\n");
    printf("| Layer |  number of neurons | Parameters  |\n");
    printf("============================================\n");
    for(int j=0;j<network->num_layers;j++){
        printf("|  %i  |  %i   | %i\n", j, network->layers[j]->num_neurons,network->layers[j]->weights->size+network->layers[j]->biases->size);
    }
    return;
}

void ON_display_weights(Dense_network* network){
    printf("Weights of the network : \n");
    for(int i=0;i<network->num_layers;i++){
        printf("Layer %i : \n",i);
        printf("Weights : \n");
        print_tensor(network->layers[i]->weights,1);
        printf("Biases : \n");
        print_tensor(network->layers[i]->biases,1);
    }
}

OtterTensor* layer_calc(OtterTensor* input, Dense_layer* layer, OtterTensor* zs, OtterTensor* activation){
    OtterTensor* prod= OT_Matrix_multiply(layer->weights,input);
    OT_ref_tensors_sum(prod,layer->biases);
    if(zs!=NULL){zs=OT_copy(prod);}
    if (layer->activation_function != NULL) {
        Activation_functions(layer->activation_function, prod);
    }
    if(activation!=NULL){
        activation=OT_copy(prod);
    }
    return(prod);
}



OtterTensor* ON_feed_forward(Dense_network* network, OtterTensor* input, OtterTensor** zs, OtterTensor** activations) {
    OtterTensor* last_values = OT_copy(input);
    for (int i = 0; i < network->num_layers; i++) {
        OtterTensor* prod = OT_Matrix_multiply(network->layers[i]->weights, last_values);
        OT_ref_tensors_sum(prod, network->layers[i]->biases);
        if (zs) zs[i] = OT_copy(prod); 
        if (network->layers[i]->activation_function != NULL) {
            Activation_functions(network->layers[i]->activation_function, prod);
        }
        if (activations) activations[i] = OT_copy(prod);
        free_malloc_tensor(last_values);
        last_values = prod;
    }
    return last_values;
}


OtterTensor* ON_Cost_derivative(OtterTensor* output, OtterTensor* labels, char* error_function){
    if (strcmp(error_function, "MSE") == 0) {
        OtterTensor* diff = OT_tensors_substract(output, labels);
        OT_ref_scalar_multiply(diff, 2.0f);
        return diff;
    }
    return NULL;
}



OtterTensor* ON_cost(OtterTensor* output, OtterTensor* labels, char* error_function){
    if (strcmp(error_function, "MSE") == 0) {
        OtterTensor* diff = OT_tensors_substract(output, labels);
        OT_ref_square(diff);
        return diff;
    }
    return NULL;
}

OtterTensor** ON_copy_weights(Dense_network* network){
    OtterTensor** weights = malloc(network->num_layers * sizeof(OtterTensor*));
    for (int i = 0; i < network->num_layers; i++) {
        weights[i] = OT_copy(network->layers[i]->weights);
    }
    return weights;
}

OtterTensor** ON_copy_biases(Dense_network* network){
    OtterTensor** biases = malloc(network->num_layers * sizeof(OtterTensor*));
    for (int i = 0; i < network->num_layers; i++) {
        biases[i] = OT_copy(network->layers[i]->biases);
    }
    return biases;
}



OtterTensor** ON_local_error_def(Dense_network* network){
    OtterTensor** errors = malloc(network->num_layers * sizeof(OtterTensor*));
    for (int i = 0; i < network->num_layers; i++) {
        int dims[2] = {network->layers[i]->num_neurons, 1};
        errors[i] = OT_zeros(dims, 2);
    }
    return errors;
}

void ON_update_weights_and_biases(Dense_network* network, OtterTensor** weights_gradients, OtterTensor** biases_gradients) {
    for (int i = 0; i < network->num_layers; i++) {
        OT_ref_tensors_sum(network->layers[i]->weights, weights_gradients[i]);
        OT_ref_tensors_sum(network->layers[i]->biases, biases_gradients[i]);
    }
}

OtterTensor* ON_predict(Dense_network* network, OtterTensor* input) {
    OtterTensor* predictions = ON_feed_forward(network, input, NULL, NULL);
    return predictions;
}

OtterTensor*** ON_init_params(Dense_network* network) {
    OtterTensor*** learnable_params = malloc(2 * sizeof(OtterTensor**));
    learnable_params[0] = malloc(network->num_layers * sizeof(OtterTensor*));
    learnable_params[1] = malloc(network->num_layers * sizeof(OtterTensor*));
    for (int k = 0; k < network->num_layers; k++) {
        learnable_params[0][k] = OT_zeros(network->layers[k]->weights->dims, 2);
        learnable_params[1][k] = OT_zeros(network->layers[k]->biases->dims, 2);
    }
    return learnable_params;
}
OtterTensor*** ON_init_grads(Dense_network* network) {
    OtterTensor*** learnable_params = malloc(2 * sizeof(OtterTensor**));
    learnable_params[0] = calloc(network->num_layers, sizeof(OtterTensor*));
    learnable_params[1] = calloc(network->num_layers, sizeof(OtterTensor*));
    return learnable_params;
}


void ON_fit(Dense_network* network, OtterDataset* inputs, OtterDataset* labels, int epochs, int batch_size) {
    switch (network->optimizer) 
    {
    case 0:
        ON_SGD_fit(network, inputs, labels, epochs, batch_size);
        break;
    case 1:
        ON_SGDM_fit(network, inputs, labels, epochs, batch_size);
        break;
    case 2:
        ON_Adam_fit(network, inputs, labels, epochs, batch_size);
        break;
    default:
        fprintf(stderr, "Unknown optimizer: %d\n", network->optimizer);
        exit(EXIT_FAILURE);
        break;
    }
    return;
}

void free_params(OtterTensor*** params, int num_layers) {
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < num_layers; j++) {
            if (params[i][j]) free_malloc_tensor(params[i][j]);
        }
        free(params[i]);
    }
    free(params);
}


void free_net(Dense_network* network){
    if (!network) return;
    for (int i = 0; i < network->num_layers; i++) {
        free_malloc_tensor(network->layers[i]->weights);
        free_malloc_tensor(network->layers[i]->biases);
        if (network->layers[i]->activation_function){
            free(network->layers[i]->activation_function);
        }
        free(network->layers[i]);
    }
    if (network->error_function) free(network->error_function);
    free(network->layers);
    free(network);
}