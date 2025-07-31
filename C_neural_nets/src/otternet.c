#include "../header/otternet.h"
#include "../header/otternet_optimizers.h"




Otternetwork* ON_initialise_otternetwork() { 
    Otternetwork* network = malloc(sizeof(Otternetwork));
    network->num_layers = 0;
    network->layers = NULL;
    network->end= NULL;
    network->start = NULL;
    network->error_function = NULL;
    network->optimizer = -1; // -1 means no optimizer set
    network->learning_rate = 0.01; // Default learning rate
    network->optimizer_params = NULL; // No optimizer parameters by default
    network->order = NULL; // No order set initially
    network->end_of_line = NULL;
    network->num_end_of_line = 0; // No layers with no forward connections
    network->start_of_line = NULL;
    network->num_start_of_line = 0; // No layers with no backward connections
    network->output = NULL; // No output tensor set initially
    network->errors = NULL; // No errors tensor set initially
    return network;
}


void ON_add_layer(Otternetwork* network, Otterchain* new_layer) {
    if (network->layers == NULL) {
    network->layers = new_layer;
    network->start  = new_layer;
    } else {
        network->end->next = new_layer;
    }
    network->end = new_layer;
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
    printf("Compiling network with %s optimizer and %s error function...\n", optimizer, error_function);
    network->order = calculate_distances_ordered(network);
    printf("Network order calculated.\n");
    Otterchain* current_layer = network->layers;
    network->start = current_layer;
    for(int i=0;i<network->num_layers;i++){
        for(int i = 0; i < current_layer->num_connections_backward; i++) {
            current_layer->connections_backward[i]->connections_forward = realloc(current_layer->connections_backward[i]->connections_forward, sizeof(Otterchain*) * (current_layer->connections_backward[i]->num_connections_forward + 1));
            current_layer->connections_backward[i]->connections_forward[current_layer->connections_backward[i]->num_connections_forward] = current_layer;
            current_layer->connections_backward[i]->num_connections_forward++;
        }
        switch(current_layer->type){
            case 0: // Dense layer
                ON_compile_Dense_layer(current_layer);
                break;
            case 1: // Conv1D layer
                //ON_compile_Conv1D_layer(current_layer);
                break;
            case 2: // Flatten layer
                //ON_compile_Flatten_layer(current_layer);
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
    for(int i = 0; i < network->num_layers; i++) {
        Otterchain* node = network->order[i];
        node->network_rank = i; // Assigning rank to each layer
        if(node->num_connections_backward == 0){
            Otterchain** tmp = realloc(network->start_of_line, (network->num_start_of_line + 1) * sizeof(Otterchain*));
            if (!tmp) {
                fprintf(stderr, "Failed to realloc start_of_line\n");
                exit(EXIT_FAILURE);
            }
            network->start_of_line = tmp;
            network->start_of_line[network->num_start_of_line] = node;
            node->idx_input = network->num_start_of_line; // Assigning index to input layer
            network->num_start_of_line++;
        }
        if(node->num_connections_forward == 0){
            Otterchain** tmp = realloc(network->end_of_line, (network->num_end_of_line + 1) * sizeof(Otterchain*));
            if (!tmp) {
                fprintf(stderr, "Failed to realloc end_of_line\n");
                exit(EXIT_FAILURE);
            }
            network->end_of_line = tmp;
            network->end_of_line[network->num_end_of_line] = node;
            node->idx_output = network->num_end_of_line; // Assigning index to output layer
            network->num_end_of_line++;
        }
    }
    network->output = malloc(sizeof(OtterTensor*) * network->num_end_of_line);

    printf("Network compiled correctly with %s optimizer and %s error function.\n", optimizer, error_function);
    return;
}






Otterchain** calculate_distances_ordered(Otternetwork* net) {
    int n = net->num_layers;

    // 1) Récupérer tous les nœuds dans un tableau
    Otterchain** nodes    = calloc(n, sizeof *nodes);
    int*        in_degree = calloc(n, sizeof *in_degree);
    for (int i = 0; i < n; i++) {
        nodes[i] = (i == 0
            ? net->layers
            : nodes[i-1]->next
        );
    }

    // 2) Calculer in_degree (nombre d'arêtes entrantes) via connections_backward
    for (int i = 0; i < n; i++) {
        Otterchain* layer = nodes[i];
        for (int j = 0; j < layer->num_connections_backward; j++) {
            Otterchain* pred = layer->connections_backward[j];
            int idx = find_index(nodes, n, pred);
            if (idx >= 0) {
                // couche `pred` → couche `layer`
                in_degree[i]++;
            }
        }
    }

    // 3) Préparer la file (résultat servira aussi de file)
    Otterchain** result = malloc(n * sizeof *result);
    int head = 0, tail = 0;
    // Enfiler les couches sans prédécesseur
    for (int i = 0; i < n; i++) {
        if (in_degree[i] == 0) {
            result[tail++] = nodes[i];
        }
    }

    // 4) Parcourir la file
    while (head < tail) {
        Otterchain* u = result[head++];
        // Pour chaque voisin v : c'est tout node v dont u est dans connections_backward
        for (int k = 0; k < n; k++) {
            Otterchain* v = nodes[k];
            // chercher si u ∈ v->connections_backward
            for (int m = 0; m < v->num_connections_backward; m++) {
                if (v->connections_backward[m] == u) {
                    if (--in_degree[k] == 0) {
                        result[tail++] = v;
                    }
                    break;
                }
            }
        }
    }

    free(nodes);
    free(in_degree);

    for(int i = 0; i< net->num_layers; i++) {
        result[i]->network_rank = i; 
    }

    return result;
}


OtterTensor** ON_feed_forward(Otternetwork* network, OtterTensor** input, int gradient_register) { // à retravailler
    network->input = malloc(network->num_start_of_line* sizeof(OtterTensor*));
    for( int i = 0; i < network->num_start_of_line; i++) {
        network->input[i] = OT_copy(input[i]);
    }
    
    for (int i = 0; i < network->num_layers; i++) {

        switch (network->order[i]->type)
        {
        case 0:
            ON_Dense_layer_forward(network,network->order[i], gradient_register);
            
            break;
        case 1:
            //OtterTensor* prod_conv= ON_Conv1D_layer_forward((Conv1D_layer*)network->order[i]->layer,network->order[i], last_values, gradient_register);

            //last_values = prod_conv;
            break;
        case 2:
            //OtterTensor* prod_flat= ON_Flatten_layer_forward((Flatten_layer*)network->order[i]->layer,network->order[i], last_values, gradient_register);

            //last_values = prod_flat;
            break;
        
        default:
            fprintf(stderr, "Unknown layer type for feed forward.\n");
            exit(EXIT_FAILURE);
            break;
        }
        
    }

    for(int i = 0; i < network->num_end_of_line; i++) {
        network->output[i] = OT_copy(network->end_of_line[i]->post_activations);
    }

    return network->output;
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


void ON_update_weights_and_biases(Otternetwork* network) {
    for (int i = 0; i < network->num_layers; i++) {
        for(int j = 0; j < network->order[i]->weights_depth; j++) {
            OT_ref_tensors_sum(network->order[i]->weights[j], network->order[i]->weights_gradients[j]);
            free_malloc_tensor(network->order[i]->weights_gradients[j]);
            network->order[i]->weights_gradients[j] = NULL;
            
            OT_ref_tensors_sum(network->order[i]->biases[j], network->order[i]->biases_gradients[j]);
            free_malloc_tensor(network->order[i]->biases_gradients[j]);
            network->order[i]->biases_gradients[j] = NULL;
        }
        
    }
    return;
}

OtterTensor** ON_predict(Otternetwork* network, OtterTensor** input) {
    OtterTensor** predictions = ON_feed_forward(network, input, 0);
    return predictions;
}



void ON_fit(Otternetwork* network, OtterDataset* inputs, OtterDataset* labels, int epochs, int batch_size) {
    switch (network->optimizer) 
    {
    case 0:
        ON_SGD_fit(network, inputs, labels, epochs, batch_size);
        break;
    case 1:
        //ON_SGDM_fit(network, inputs, labels, epochs, batch_size);
        break;
    case 2:
        //ON_Adam_fit(network, inputs, labels, epochs, batch_size);
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
    return;
}




void free_otternetwork(Otternetwork* network) {
    if (!network) return;
    Otterchain* current = network->layers;
    while (current != NULL) {
        Otterchain* next = current->next;
        free_otterchain(current);
        current = next;
    }
    if(network->input){
        for (int i = 0; i < network->num_start_of_line; i++) {
            free_malloc_tensor(network->input[i]);
        }
        free(network->input);
    }
    free(network->order);
    if (network->optimizer_params) free(network->optimizer_params);
    if (network->error_function) free(network->error_function);
    if (network->start_of_line) free(network->start_of_line);
    if (network->end_of_line) free(network->end_of_line);
    if (network->output) {
        for (int i = 0; i < network->num_end_of_line; i++) {
            free_malloc_tensor(network->output[i]);
        }
        free(network->output);
    }
    if( network->errors) {
        for (int i = 0; i < network->num_end_of_line; i++) {
            free_malloc_tensor(network->errors[i]);
        }
        free(network->errors);
        network->errors = NULL;
    }
    free(network);
    return;
}


void free_otterchain(Otterchain* chain){
    if (!chain) return;
    if (chain->layer) {
        if (chain->type == 0) {
            free_Dense_layer(chain->layer);
        } else if (chain->type == 1) {
            //free_Conv1D_layer((Conv1D_layer*)chain->layer);
        } else if (chain->type == 2) {
            //free_Flatten_layer((Flatten_layer*)chain->layer);
        } else {
            fprintf(stderr, "Unknown layer type for freeing.\n");
            exit(EXIT_FAILURE);
        }
    }

    if (chain->input_dims) free(chain->input_dims);
    if (chain->output_dims) free(chain->output_dims);

    // Only free gradients if not already freed in ON_update_weights_and_biases
    if(chain->weights_gradients) {
        for (int i = 0; i < chain->weights_depth; i++) {
            if (chain->weights_gradients[i]) free_malloc_tensor(chain->weights_gradients[i]);
            if (chain->biases_gradients[i]) free_malloc_tensor(chain->biases_gradients[i]);
        }
        free(chain->weights_gradients);
        free(chain->biases_gradients);
    }

    if(chain->weights) {
        for (int i = 0; i < chain->weights_depth; i++) {
            if (chain->weights[i]) free_malloc_tensor(chain->weights[i]);
            if (chain->biases[i]) free_malloc_tensor(chain->biases[i]);
        }
        free(chain->weights);
        free(chain->biases);
    }
    if (chain->connections_backward) {
        free(chain->connections_backward);
    }
    if (chain->connections_forward) {
        free(chain->connections_forward);
    }
    if(chain->local_errors) {
        free_malloc_tensor(chain->local_errors);
    }
    if (chain->input){ 
        for (int i = 0; i < chain->num_connections_backward; i++) {
            if (chain->input[i]) free_malloc_tensor(chain->input[i]);
        }
        free(chain->input);
    }
    if (chain->pre_activation) {free_malloc_tensor(chain->pre_activation);}
    if (chain->post_activations){free_malloc_tensor(chain->post_activations);}
    free(chain);
    return;
}