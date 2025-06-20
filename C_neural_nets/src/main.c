#include "../header/main.h"

int main() {
    int layers[3] = {2, 3, 1};
    char* activations[3] = {"tanh", "tanh", "linear"};
    float optimizer_params[3] = {0.9f}; // Momentum parameter for SGDM
    Dense_network* net = ON_initialise_network(layers, 3, activations);
    ON_compile_network(net, "Adam", "MSE", 0.1f, optimizer_params);
    ON_display_network(net);

    int input_dims[2] = {2, 1};
    OtterTensor* input = OT_ones(input_dims, 2);

    int label_dims[2] = {1, 1};
    OtterTensor* label = OT_ones(label_dims, 2);

    OtterDataset* inputs = malloc(sizeof(OtterDataset));
    inputs->size = 1;
    inputs->dataset = malloc(sizeof(OtterTensor*));
    inputs->dataset[0] = input;

    OtterDataset* labels = malloc(sizeof(OtterDataset));
    labels->size = 1;
    labels->dataset = malloc(sizeof(OtterTensor*));
    labels->dataset[0] = label;
    
    printf("Prediction before training:\n");
    OtterTensor* pred = ON_predict(net, input);
    print_tensor(pred, 3);
    free_malloc_tensor(pred);

    printf("\n entrainement \n");

    ON_fit(net, inputs, labels, 500, 1);

    printf("Prediction after training:\n");
    pred = ON_predict(net, input);
    print_tensor(pred, 3);
    free_malloc_tensor(pred);
    
    free_net(net);
    free_malloc_tensor(input);
    free_malloc_tensor(label);
    free(inputs->dataset);
    free(labels->dataset);
    free(inputs);
    free(labels);
    
    // Free the network
    return 0;
}