#include "../header/otternet.h"
#include "../header/otternet_optimizers.h"




Otternetwork* ON_initialise_otternetwork(Otterchain* layers) { // a retravailler
    Otternetwork* network = malloc(sizeof(Otternetwork));
    network->layers = layers;
    network->num_layers = 0;
    return network;
}


void ON_add_layer(Otternetwork* network, Otterchain* new_layer) {
    new_layer->next = NULL;

    Otterchain* current = network->layers;
    while (current->next != NULL) {
        current = current->next;
    }
    current->next = new_layer;

    network->num_layers++;
}



void ON_compile_otternetwork(Otternetwork* network, char* optimizer, char* error_function, float learning_rate, float* optimizer_params) {
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
    Otterchain* current_layer = network->layers;
    network->start = current_layer;
    for(int i=0;i<network->num_layers;i++){
        switch(current_layer->type){
            case 0: // Dense layer
                ON_compile_Dense_layer(current_layer);
                break;
            case 1: // Conv1D layer
                ON_compile_Conv1D_layer(current_layer);
                break;
            case 2: // Flatten layer
                ON_compile_Flatten_layer(current_layer);
                break;
            default:
                fprintf(stderr, "Unknown layer type for compilation.\n");
                exit(EXIT_FAILURE);
        }
        current_layer = current_layer->next;
        if(i == network->num_layers - 1){
            network->end = current_layer;
        }
    }

    printf("Network compiled correctly with %s optimizer and %s error function.\n", optimizer, error_function);
    return;
}

/*
void ON_display_network(Otternetwork* network){
    int full_param = get_full_size_of_OTN(network);
    printf("Network with %i layers, for a total of %i parameters \n",network->num_layers,full_param);
    printf("The network structure is the following : \n");
    printf("============================================\n");
    printf("| Layer |  number of neurons | Parameters  |\n");
    printf("============================================\n");
    for(int j=0;j<network->num_layers;j++){
        printf("|  %i  |  %i   | %i\n", network->layers[j]->type, network->layers[j]->num_neurons,network->layers[j]->weights->size+network->layers[j]->biases->size);
    }
    return;
}
*/



Otterchain** calculate_distances_ordered(Otternetwork* net) {
    int* distances = malloc(net->num_layers * sizeof(int));
    int* unvisited_distance = malloc(net->num_layers * sizeof(int));
    Otterchain** unvisited = malloc(net->num_layers * sizeof(Otterchain*));
    Otterchain** result = malloc(net->num_layers * sizeof(Otterchain*));

    Otterchain* current = net->layers;
    for (int i = 0; i < net->num_layers; i++) {
        distances[i] = 9999999;
        unvisited_distance[i] = 9999999;
        unvisited[i] = current;
        current = current->next;
    }
    distances[0] = 0;
    unvisited_distance[0] = 0;

    for (int i = 0; i < net->num_layers; i++) {
        int u = argmin(unvisited_distance, net->num_layers);
        current = unvisited[u];
        unvisited_distance[u] = 9999999;

        int distance = distances[u] + 1;
        for (int j = 0; j < current->num_connections; j++) {
            Otterchain* neighbor = current->connections[j];
            int idx = find_index(unvisited, net->num_layers, neighbor);
            if (idx != -1 && distance < distances[idx]) {
                distances[idx] = distance;
                unvisited_distance[idx] = distance;
            }
        }
    }

    int* ranks = malloc(net->num_layers * sizeof(int));
    rankify(distances, ranks, net->num_layers);
    for (int i = 0; i < net->num_layers; i++) {
        result[ranks[i]] = unvisited[i];
    }

    free(unvisited);
    free(unvisited_distance);
    free(distances);
    free(ranks);

    return result;
}




OtterTensor* ON_feed_forward(Otternetwork* network, OtterTensor* input, int gradient_register) { // à retravailler
    OtterTensor* last_values = OT_copy(input);
    for (int i = 0; i < network->num_layers; i++) {
        switch (network->order[i]->type)
        {
        case 0:
            OtterTensor* prod= ON_Dense_layer_forward((Dense_layer*)network->order[i]->layer,network->order[i], last_values, gradient_register);
            free_malloc_tensor(last_values);
            last_values = prod;
            break;
        case 1:
            OtterTensor* prod_conv= ON_Conv1D_layer_forward((Conv1D_layer*)network->order[i]->layer,network->order[i], last_values, gradient_register);
            free_malloc_tensor(last_values);
            last_values = prod_conv;
            break;
        case 2:
            OtterTensor* prod_flat= ON_Flatten_layer_forward((Flatten_layer*)network->order[i]->layer,network->order[i], last_values, gradient_register);
            free_malloc_tensor(last_values);
            last_values = prod_flat;
            break;
        
        default:
            fprintf(stderr, "Unknown layer type for feed forward.\n");
            exit(EXIT_FAILURE);
            break;
        }
        
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
/* 
OtterTensor** ON_copy_weights(Otternetwork* network){
    OtterTensor** weights = malloc(network->num_layers * sizeof(OtterTensor*));
    for (int i = 0; i < network->num_layers; i++) {
        weights[i] = OT_copy(network->layers[i]->weights);
    }
    return weights;
}

OtterTensor** ON_copy_biases(Otternetwork* network){
    OtterTensor** biases = malloc(network->num_layers * sizeof(OtterTensor*));
    for (int i = 0; i < network->num_layers; i++) {
        biases[i] = OT_copy(network->layers[i]->biases);
    }
    return biases;
} */



/* OtterTensor** ON_local_error_def(Dense_network* network){
    OtterTensor** errors = malloc(network->num_layers * sizeof(OtterTensor*));
    for (int i = 0; i < network->num_layers; i++) {
        int dims[2] = {network->layers[i]->num_neurons, 1};
        errors[i] = OT_zeros(dims, 2);
    }
    return errors;
} */

void ON_update_weights_and_biases(Otternetwork* network) {
    for (int i = 0; i < network->num_layers; i++) {
        for(int j = 0; j < network->order[i]->weights_depth; j++) {
            OT_ref_tensors_sum(network->order[i]->weights[j], network->order[i]->weights_gradients[j]);
            free_malloc_tensor(network->order[i]->weights_gradients[j]);
            
            OT_ref_tensors_sum(network->order[i]->biases[j], network->order[i]->biases_gradients[j]);
            free_malloc_tensor(network->order[i]->biases_gradients[j]);
        }
        
    }
    
}

OtterTensor* ON_predict(Otternetwork* network, OtterTensor* input) {
    OtterTensor* predictions = ON_feed_forward(network, input, NULL, NULL);
    return predictions;
}
/* 
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
} */


void ON_fit(Otternetwork* network, OtterDataset* inputs, OtterDataset* labels, int epochs, int batch_size) {
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




void free_otternetwork(Otternetwork* network) {
    if (!network) return;
    Otterchain* current = network->layers;
    while (current != NULL) {
        Otterchain* next = current->next;
        get_layer_type(current->layer);
        if (current->type == 0) {
            free_Dense_layer((Dense_layer*)current->layer);
        } else if (current->type == 1) {
            //free_Conv1D_layer((Conv1D_layer*)current->layer);
        } else if (current->type == 2) {
            //free_Flatten_layer((Flatten_layer*)current->layer);
        } else {
            fprintf(stderr, "Unknown layer type for freeing.\n");
            exit(EXIT_FAILURE);
        }
        free(current->layer);
        free(current);
        current = next;
    }
    if (network->error_function) free(network->error_function);
    free(network);
}