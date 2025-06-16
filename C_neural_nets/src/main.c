#include "../header/main.h"

int main() {
    // Define a simple network: 2 inputs, 3 hidden, 1 output
    int layers[3] = {2, 3, 1};
    char* activations[3] = {"relu", "sigmoid", "linear"};
    Dense_network* net = ON_initialise_network(layers, 3, activations);
    ON_compile_network(net, "SGD", "MSE", 0.01f);

    // Create a dummy input tensor (2x1)
    int input_dims[2] = {2, 1};
    OtterTensor* input = OT_ones(input_dims, 2);

    // Create a dummy label tensor (1x1)
    int label_dims[2] = {1, 1};
    OtterTensor* label = OT_ones(label_dims, 2);

    // Wrap input and label in datasets
    OtterDataset* inputs = malloc(sizeof(OtterDataset));
    inputs->size = 1;
    inputs->dataset = malloc(sizeof(OtterTensor*));
    inputs->dataset[0] = input;

    OtterDataset* labels = malloc(sizeof(OtterDataset));
    labels->size = 1;
    labels->dataset = malloc(sizeof(OtterTensor*));
    labels->dataset[0] = label;
    printf("Network initialized with %d layers.\n", net->layers[1]->weights.rank);
    // Predict before training
    printf("Prediction before training:\n");
    OtterTensor* pred = ON_predict(net, input);
    print_tensor(pred, 3);
    free_malloc_tensor(pred);
    printf("\n entrainement \n");

    // Fit the model (train for 5 epochs, batch size 1)
    ON_fit(net, inputs, labels, 5, 1);

    // Predict after training
    printf("Prediction after training:\n");
    pred = ON_predict(net, input);
    print_tensor(pred, 3);
    free_malloc_tensor(pred);

    // Cleanup
    free_net(net);
    free_malloc_tensor(input);
    free_malloc_tensor(label);
    free(inputs->dataset);
    free(labels->dataset);
    free(inputs);
    free(labels);

    return 0;
}