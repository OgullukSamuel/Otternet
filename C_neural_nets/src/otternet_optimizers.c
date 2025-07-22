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

void ON_SGD_fit(Otternetwork* network, OtterDataset* inputs, OtterDataset* labels, int epochs, int batch_size){
    for (int epoch = 0; epoch < epochs; epoch++) {
        int* indices = OR_select_batch(inputs->size, batch_size);

        for (int i = 0; i < batch_size; i++) {
            int idx = indices[i];
            ON_SGD(network, inputs->dataset[idx], labels->dataset[idx]);

            for (int l = 0; l < network->num_layers; l++) {
                for(int i= 0; i < network->order[l]->weights_depth; i++) {
                    OT_ref_scalar_multiply(network->order[l]->weights_gradients[i], -network->learning_rate);
                    OT_ref_scalar_multiply(network->order[l]->biases_gradients[i], -network->learning_rate);    
                }
            }

            ON_update_weights_and_biases(network);
            
        }
        free(indices);
        ON_verbose1(epoch, network, inputs, labels);
    }
}

void ON_verbose1(int epoch,Otternetwork* network, OtterDataset* inputs, OtterDataset* labels) {
    if (epoch % 50 == 0) {
        OtterTensor* pred = ON_predict(network, inputs->dataset[0]);
        OtterTensor* loss = ON_cost(pred, labels->dataset[0], network->error_function);
        printf("Epoch %d - Prediction: %.3f, Error: %.3f\n", epoch, pred->data[0], loss->data[0]);
        free_malloc_tensor(pred);
        free_malloc_tensor(loss);
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


void ON_SGD(Otternetwork* network, OtterTensor* input, OtterTensor* labels) {
    int L = network->num_layers;
    for(int i=0; i<L;i++){
        network->order[i]->weights_gradients = malloc(network->order[i]->weights_depth * sizeof(OtterTensor*));
        network->order[i]->biases_gradients  = malloc(network->order[i]->weights_depth * sizeof(OtterTensor*));
        network->order[i]->local_errors      = OT_zeros(network->order[i]->output_dims, 2);
        network->order[i]->pre_activation    = OT_zeros(network->order[i]->output_dims, 2);
        network->order[i]->post_activations  = OT_zeros(network->order[i]->output_dims, 2);    
    }
    OtterTensor* predictions = ON_feed_forward(network, input, 1);
    OtterTensor* error = ON_Cost_derivative(predictions, labels, network->error_function);
    free_malloc_tensor(predictions);
    free_malloc_tensor(error);
    for(int i_layer= network->num_layers-1; i_layer >=0 ; i_layer--){
        switch (network->order[i_layer]->type) {
            case 0: // Dense layer

                ON_Dense_layer_backward( network->order[i_layer]);
                break;
            case 1: // Conv1D layer
                //ON_Conv1D_layer_backward(network, network->order[i_layer], input, i_layer);
                break;
            case 2: // Flatten layer
                //ON_Flatten_layer_backward(network, network->order[i_layer], input, i_layer);
                break;
            default:
                fprintf(stderr, "Unknown layer type for backward pass.\n");
                exit(EXIT_FAILURE);
        }
    }

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