#include "../header/otternet_optimizers.h"


void ON_first_moment_estimation(OtterTensor* momentum, OtterTensor* gradient, float beta1) {
    // m_t = beta1 * m_t-1 + (1 - beta1) * grad_t
    OT_ref_scalar_multiply(momentum, beta1);
    OT_ref_scalar_multiply(gradient, 1 - beta1);
    OT_ref_tensors_sum(momentum, gradient, "ON_first_moment_estimation");
}

void ON_second_moment_estimation(OtterTensor* velocity, OtterTensor* gradient, float beta2) {
    // v_t = beta2 * v_t-1 + (1 - beta2) * grad_t^2
    OT_ref_scalar_multiply(velocity, beta2);
    OT_ref_square(gradient);
    OT_ref_scalar_multiply(gradient, 1 - beta2);
    OT_ref_tensors_sum(velocity, gradient, "ON_second_moment_estimation");
}

void ON_SGD_fit(Otternetwork* network, OtterDataset* inputs, OtterDataset* labels, int epochs, int batch_size) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        int* indices = OR_select_batch(inputs->size[0], batch_size);

        for (int i = 0; i < batch_size; i++) {
            int idx = indices[i];
            OtterTensor** input_arr = calloc(network->num_start_of_line, sizeof(OtterTensor*));
            OtterTensor** label_arr = calloc(network->num_end_of_line, sizeof(OtterTensor*));
            for (int j = 0; j < network->num_start_of_line; j++) {
                input_arr[j] = inputs->dataset[idx][j];
            }
            for (int j = 0; j < network->num_end_of_line; j++) {
                label_arr[j] = labels->dataset[idx][j];
            }

            ON_SGD(network, input_arr, label_arr);

            for (int l = 0; l < network->num_layers; l++) {
                for (int k = 0; k < network->order[l]->weights_depth; k++) {
                    OT_ref_scalar_multiply(network->order[l]->weights_gradients[k], -network->learning_rate);
                    OT_ref_scalar_multiply(network->order[l]->biases_gradients[k], -network->learning_rate);
                    
                }
            }

            ON_update_weights_and_biases(network);

            free(input_arr);
            free(label_arr);
        }
            if (epoch % ((int)(epochs/10)+1) == 0) {ON_verbose1(epoch, network, inputs, labels, indices,batch_size);}
        free(indices);
    }
}



void ON_SGD(Otternetwork* network, OtterTensor** input, OtterTensor** labels) {
    int L = network->num_layers;

    OtterTensor** predictions = ON_feed_forward(network, input, 1);

    for (int i = 0; i < network->num_end_of_line; i++) {
        if (network->errors[i]) {
            free_malloc_tensor(&network->errors[i]);
        }
        network->errors[i] = ON_Cost_derivative(predictions[i], labels[i], network->error_function);
        if (!network->errors[i]) {
            fprintf(stderr, "Error: Null tensor encountered during cost derivative computation.\n");
            exit(EXIT_FAILURE);
        }
    }


    for (int i_layer = L - 1; i_layer >= 0; i_layer--) {
        switch (network->order[i_layer]->type) {
            case 0: // Dense layer
                ON_Dense_layer_backward(network, network->order[i_layer]);
                break;
            default:
                fprintf(stderr, "Unknown layer type for backward pass.\n");
                exit(EXIT_FAILURE);
        }
    }
    for(int i=0;i<network->num_end_of_line;i++){
        free_malloc_tensor(&predictions[i]);
    }
    free(predictions);
    predictions = NULL; 
    ON_reset_network(network);
}


void ON_verbose1(int epoch, Otternetwork* network, OtterDataset* inputs, OtterDataset* labels, int* indices,int batch_size) {
    printf("Epoch %d\n", epoch);
    for (int i = 0; i < network->num_end_of_line; i++) {
        printf("  Output %d:\n", i);
        float loss = 0.0f;
        for (int j = 0; j < batch_size; j++) {
            int idx = indices ? indices[i] : i;
            OtterTensor** pred = ON_predict(network, inputs->dataset[idx]);
            loss += ON_cost(pred[i], labels->dataset[idx][i], network->error_function);
            free_ottertensor_list(pred, network->num_end_of_line);
            
        }
        printf("losses: %.3f\n", loss/batch_size);
    }

}

/* 
void ON_SGDM_fit(Otternetwork* network, OtterDataset* inputs, OtterDataset* labels, int epochs, int batch_size){
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
} */

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
/* 
void ON_Adam_fit(Otternetwork* network, OtterDataset* inputs, OtterDataset* labels, int epochs, int batch_size){
    // Initialize Adam parameters
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;

    if (network->optimizer_params[0]) {beta1 = network->optimizer_params[0];  }
    if (network->optimizer_params[1]) {beta2 = network->optimizer_params[1];  }
    if (network->optimizer_params[2]) {epsilon = network->optimizer_params[2];}

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

 */