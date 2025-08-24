/* #include "../header/optimized_dense_nets.h"

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


void ON_compile_Dense_network(Dense_network* network, char* optimizer, char* error_function, float learning_rate, float* optimizer_params) {
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

OtterTensor* ON_feed_forward(Dense_network* network, OtterTensor* input, OtterTensor** zs, OtterTensor** activations) {
    OtterTensor* last_values = OT_copy(input);
    for (int i = 0; i < network->num_layers; i++) {
        if(network->layers[i]->type==1){
            OtterTensor* prod= ON_Dense_layer_forward(network->layers[i], last_values, zs, activations);
            free_malloc_tensor(last_values);
            last_values = prod;
        }

        
    }
    return last_values;
}


int get_full_size_of_DN(Dense_network* network){
    int size=0;
    for(int i =0; i<network->num_layers; i++){
        size += network->layers[i]->weights->size + network->layers[i]->biases->size;
    }
    return(size);
}






#include "../header/otternet_optimizers.h"

void ON_first_moment_estimation(OtterTensor* momentum, OtterTensor* gradient, float beta1) {
    // m_t = beta1 * m_t-1 + (1 - beta1) * grad_t
    OT_ref_scalar_multiply(momentum, beta1);
    OT_ref_scalar_multiply(gradient, 1 - beta1);
    OT_ref_tensors_sum(momentum, gradient);
}

void ON_second_moment_estimation(OtterTensor* velocity, OtterTensor* gradient, float beta2) {
    // v_t = beta2 * v_t-1 + (1 - beta2) * grad_t^2
    OT_ref_scalar_multiply(velocity, beta2);
    OT_ref_square(gradient);
    OT_ref_scalar_multiply(gradient, 1 - beta2);
    OT_ref_tensors_sum(velocity, gradient);
}

void ON_SGD_fit(Dense_network* network, OtterDataset* inputs, OtterDataset* labels, int epochs, int batch_size){
    for (int epoch = 0; epoch < epochs; epoch++) {
        int* indices = OR_select_batch(inputs->size, batch_size);

        for (int i = 0; i < batch_size; i++) {
            int idx = indices[i];
            OtterTensor*** grads = ON_init_grads(network);
            ON_SGD(network, inputs->dataset[idx], labels->dataset[idx], grads);

            for (int l = 0; l < network->num_layers; l++) {
                OT_ref_scalar_multiply(grads[0][l], -network->learning_rate);
                OT_ref_scalar_multiply(grads[1][l], -network->learning_rate);
            }

            ON_update_weights_and_biases(network, grads[0], grads[1]);
            free_params(grads, network->num_layers);
        }
        free(indices);
        ON_verbose1(epoch, network, inputs, labels);
    }
    
}

void ON_verbose1(int epoch,Dense_network* network, OtterDataset* inputs, OtterDataset* labels) {
    if (epoch % 50 == 0) {
        OtterTensor* pred = ON_predict(network, inputs->dataset[0]);
        OtterTensor* loss = ON_cost(pred, labels->dataset[0], network->error_function);
        printf("Epoch %d - Prediction: %.3f, Error: %.3f\n", epoch, pred->data[0], loss->data[0]);
        free_malloc_tensor(pred);
        free_malloc_tensor(loss);
    }
}


void ON_SGDM_fit(Dense_network* network, OtterDataset* inputs, OtterDataset* labels, int epochs, int batch_size){
    OtterTensor*** momentum = ON_init_params(network);
    for (int epoch = 0; epoch < epochs; epoch++) {
        int* indices = OR_select_batch(inputs->size, batch_size);
        for (int i = 0; i < batch_size; i++) {
            int idx = indices[i];
            OtterTensor*** grads = ON_init_grads(network);
            ON_SGD(network, inputs->dataset[idx], labels->dataset[idx], grads);

            for (int l = 0; l < network->num_layers; l++) {
                ON_first_moment_estimation(momentum[0][l], grads[0][l], network->optimizer_params[0]);
                ON_first_moment_estimation(momentum[1][l], grads[1][l], network->optimizer_params[0]);
                if (grads[0][l]) free_malloc_tensor(grads[0][l]);
                if (grads[1][l]) free_malloc_tensor(grads[1][l]);
                grads[0][l] = OT_scalar_multiply(momentum[0][l], -network->learning_rate);
                grads[1][l] = OT_scalar_multiply(momentum[1][l], -network->learning_rate);

            }
            ON_update_weights_and_biases(network, grads[0], grads[1]);
            free_params(grads, network->num_layers);
        }
        free(indices);
        ON_verbose1(epoch, network, inputs, labels);
    }
    free_params(momentum, network->num_layers);
}


void ON_SGD(Dense_network* network, OtterTensor* input, OtterTensor* labels,OtterTensor*** params) {
    int L = network->num_layers;

    OtterTensor** weights_gradients = calloc(L, sizeof(OtterTensor*));
    OtterTensor** biases_gradients  = calloc(L, sizeof(OtterTensor*));
    OtterTensor** local_errors      = calloc(L, sizeof(OtterTensor*));
    OtterTensor** zs                = calloc(L, sizeof(OtterTensor*));
    OtterTensor** activations       = calloc(L, sizeof(OtterTensor*));

    OtterTensor* predictions = ON_feed_forward(network, input, zs, activations);
    OtterTensor* error = ON_Cost_derivative(predictions, labels, network->error_function);
    free_malloc_tensor(predictions);

    for (int layer = L - 1; layer >= 0; --layer) {
    OtterTensor* dz = OT_copy(zs[layer]);
    derivative_activation_functions(network->layers[layer]->activation_function, dz);
    if (layer == L - 1) {
        local_errors[layer] = OT_dot(error, dz);
        free_malloc_tensor(error);
    } else {
        OtterTensor* W_next_T = OT_Transpose(network->layers[layer + 1]->weights);
        OtterTensor* temp = OT_Matrix_multiply(W_next_T, local_errors[layer + 1]);
        free_malloc_tensor(W_next_T);
        local_errors[layer] = OT_dot(temp, dz);
        free_malloc_tensor(temp);
    }
    free_malloc_tensor(dz);
    biases_gradients[layer] = OT_copy(local_errors[layer]);
    OtterTensor* a_prev = (layer == 0) ? input : activations[layer - 1];
    OtterTensor* a_prev_T = OT_Transpose(a_prev);
    weights_gradients[layer] = OT_Matrix_multiply(local_errors[layer], a_prev_T);
    free_malloc_tensor(a_prev_T);
    }

    for (int i = 0; i < L; ++i) {
    params[0][i] = OT_copy(weights_gradients[i]); // plus de scaling ici
    params[1][i] = OT_copy(biases_gradients[i]);  // idem
    }

    for (int i = 0; i < L; ++i) {
        if (local_errors[i])      free_malloc_tensor(local_errors[i]);
        if (zs[i])                free_malloc_tensor(zs[i]);
        if (activations[i])       free_malloc_tensor(activations[i]);
        if (weights_gradients[i]) free_malloc_tensor(weights_gradients[i]);
        if (biases_gradients[i])  free_malloc_tensor(biases_gradients[i]);
    }
    free(local_errors);
    free(zs);
    free(activations);
    free(weights_gradients);
    free(biases_gradients);
}

OtterTensor*** ON_deepcopy(OtterTensor*** params, int num_layers) {
    OtterTensor*** new_params = malloc(2 * sizeof(OtterTensor**));
    new_params[0] = malloc(num_layers * sizeof(OtterTensor*));
    new_params[1] = malloc(num_layers * sizeof(OtterTensor*));
    for (int i = 0; i < num_layers; i++) {
        new_params[0][i] = OT_copy(params[0][i]);
        new_params[1][i] = OT_copy(params[1][i]);
    }
    return new_params;
}

void ON_Adam_fit(Dense_network* network, OtterDataset* inputs, OtterDataset* labels, int epochs, int batch_size){
    // Initialize Adam parameters
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;

    if (network->optimizer_params[0]) beta1 = network->optimizer_params[0];
    if (network->optimizer_params[1]) beta2 = network->optimizer_params[1];
    if (network->optimizer_params[2]) epsilon = network->optimizer_params[2];

    OtterTensor*** momentum = ON_init_params(network);
    OtterTensor*** momentum_bias = ON_init_grads(network);
    OtterTensor*** velocity = ON_init_params(network);
    OtterTensor*** velocity_bias = ON_init_grads(network);
    for (int epoch = 0; epoch < epochs; epoch++) {
        float beta1_power = 1. / (1. - OM_int_power(beta1, epoch + 1));
        float beta2_power = 1. / (1. - OM_int_power(beta2, epoch + 1));
        int* indices = OR_select_batch(inputs->size, batch_size);
        for (int i = 0; i < batch_size; i++) {
            int idx = indices[i];
            OtterTensor*** grads = ON_init_grads(network);
            ON_SGD(network, inputs->dataset[idx], labels->dataset[idx], grads);
            OtterTensor*** grads2= ON_deepcopy(grads, network->num_layers);
            for (int l = 0; l < network->num_layers; l++) {
                for(int j =0; j<2;j++){
                    ON_first_moment_estimation(momentum[j][l], grads[j][l], beta1); // 1rst moment estimate
                    ON_second_moment_estimation(velocity[j][l], grads2[j][l], beta2); // 2nd moment estimate
                    
                    if (momentum_bias[j][l]) free_malloc_tensor(momentum_bias[j][l]);
                    if (velocity_bias[j][l]) free_malloc_tensor(velocity_bias[j][l]);
                    
                    momentum_bias[j][l] = OT_scalar_multiply(momentum[j][l], beta1_power);

                    velocity_bias[j][l] = OT_scalar_multiply(velocity[j][l], beta2_power);
                    OM_ref_sqrt(velocity_bias[j][l]);
                    OT_ref_scalar_sum(velocity_bias[j][l],epsilon);
                    
                    OT_ref_dot_divide(momentum_bias[j][l], velocity_bias[j][l]);

                    if (grads[j][l]) free_malloc_tensor(grads[j][l]);
                    grads[j][l] = OT_scalar_multiply(momentum_bias[j][l], -network->learning_rate);
                }
            }
            ON_update_weights_and_biases(network, grads[0], grads[1]);
            free_params(grads, network->num_layers);
            free_params(grads2, network->num_layers);
        }
        free(indices);
        ON_verbose1(epoch, network, inputs, labels);
    }
    free_params(momentum, network->num_layers);
    free_params(momentum_bias, network->num_layers);
    free_params(velocity, network->num_layers);
    free_params(velocity_bias, network->num_layers);
}

void free_dense_net(Dense_network* network){
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


void ON_display_network(Dense_network* network){
    int full_param = get_full_size_of_DN(network);
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


OtterTensor* ON_predict(Otternetwork* network, OtterTensor* input) {
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
} */