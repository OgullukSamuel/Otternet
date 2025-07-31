#include "../header/main.h"




void display_order(Otternetwork* net) {
    printf("Network order:\n");
    for (int i = 0; i < net->num_layers; i++) {
        printf("layer : %p",(net->order[i]));
    }
    printf("\n");
}

void print_tensor_list(OtterTensor** tensors, int num_tensors, int significant_digits) {
    for (int i = 0; i < num_tensors; i++) {
        printf("Tensor %d:\n", i);
        print_tensor(tensors[i], significant_digits);
    }
}




int main() {

    // Construire le réseau
    Otterchain* dense_1 = ON_Dense_layer(4, "relu", NULL, 0);
    Otterchain* dense_2 = ON_Dense_layer(1, "linear", dense_1, 1);
    Otterchain* dense_3 = ON_Dense_layer(1, "linear", dense_2, 1);
    Otterchain* dense_4 = ON_Dense_layer(2, "linear", dense_2, 1);
    Otterchain* dense_5 = ON_Dense_layer(2, "linear", dense_4, 1);
    

    Otternetwork* net = ON_initialise_otternetwork();

    ON_add_layer(net, dense_1);
    ON_add_layer(net, dense_2);
    ON_add_layer(net, dense_3);
    ON_add_layer(net, dense_4);
    ON_add_layer(net, dense_5);

    ON_compile_otternetwork(net, "SGD", "MSE", 0.01f, NULL);

    ON_display_network(net);

    // === FAUSSES DONNÉES DE TEST ===
    int dims[2] = {4, 1}; // 4 features, 1 colonne
    OtterTensor* fake_input = OT_zeros(dims, 2);
    for (int i = 0; i < fake_input->size; i++) {
        fake_input->data[i] = (float)(i + 1); // [1, 2, 3, 4]
    }
    printf("Input tensor avant feed forward :\n");
    print_tensor(fake_input, 3);

    // Correction ici : créer un tableau de OtterTensor* pour feed_forward
    OtterTensor** input_array = malloc(sizeof(OtterTensor*));
    input_array[0] = fake_input;

    // === FEED FORWARD ===
    OtterTensor** output = ON_feed_forward(net, input_array, 0);
    printf("Output tensor après feed forward :\n");
    print_tensor_list(output, net->num_end_of_line, 2);

    free(input_array);

    // === FAUSSES DONNÉES POUR FITTING ===
    int batch_size = 10;
    int input_dims[2] = {4, 1};
    int target_dims0[2] = {1, 1}; // For output 0
    int target_dims1[2] = {2, 1}; // For output 1

    OtterTensor** fake_inputs_list = malloc(batch_size * 5 * sizeof(OtterTensor*));
    OtterTensor** fake_targets_list0 = malloc(batch_size * 5 * sizeof(OtterTensor*)); // For output 0
    OtterTensor** fake_targets_list1 = malloc(batch_size * 5 * sizeof(OtterTensor*)); // For output 1

    for (int i = 0; i < batch_size*5; i++) {
        fake_inputs_list[i] = OT_zeros(input_dims, 2);

        fake_targets_list0[i] = OT_zeros(target_dims0, 2);
        fake_targets_list1[i] = OT_zeros(target_dims1, 2);

        for (int j = 0; j < fake_inputs_list[i]->size; j++) {
            fake_inputs_list[i]->data[j] = (float)(i * 10 + j);
        }
        for (int j = 0; j < fake_targets_list0[i]->size; j++) {
            fake_targets_list0[i]->data[j] = (float)(i % 2); // alternance 0/1 for output 0
        }
        for (int j = 0; j < fake_targets_list1[i]->size; j++) {
            fake_targets_list1[i]->data[j] = (float)(i % 3); // alternance 0/1/2 for output 1
        }
    }

    OtterTensor*** labels = malloc(batch_size * 5 * sizeof(OtterTensor**));
    for (int i = 0; i < batch_size*5; i++) {
        labels[i] = malloc(2 * sizeof(OtterTensor*));
        labels[i][0] = fake_targets_list0[i];
        labels[i][1] = fake_targets_list1[i];
    }

    OtterTensor*** inputs = OT_tensor_list_to_tensor3d(fake_inputs_list, batch_size*5);

    OtterDataset* fake_inputs = malloc(sizeof(OtterDataset));
    fake_inputs->dataset = inputs;
    fake_inputs->size = batch_size * 5;
    OtterDataset* fake_targets = malloc(sizeof(OtterDataset));
    fake_targets->dataset = labels;
    fake_targets->size = batch_size * 5;

    printf("Fitting sur fausses données...\n");

    /* printf("Fake inputs:\n");
    print_tensor_list(fake_inputs_list, batch_size * 5, 2);
    printf("Fake targets output 0:\n");
    print_tensor_list(fake_targets_list0, batch_size * 5, 2);
    printf("Fake targets output 1:\n");
    print_tensor_list(fake_targets_list1, batch_size * 5, 2);
     */
    ON_fit(net, fake_inputs, fake_targets, 10, batch_size); // 10 epochs, batch_size



    free_malloc_tensor(fake_input);

    // === FREE FAKE DATASETS ===
    for (int i = 0; i < batch_size * 5; i++) {
        free_malloc_tensor(fake_inputs_list[i]);
        free_malloc_tensor(fake_targets_list0[i]);
        free_malloc_tensor(fake_targets_list1[i]);
        free(labels[i]); // free sub-arrays of labels
    }
    free(fake_inputs_list);
    free(fake_targets_list0);
    free(fake_targets_list1);
    free(labels);

    // Free tensor3d array returned by OT_tensor_list_to_tensor3d
    for (int i = 0; i < batch_size * 5; i++) {
        free(inputs[i]);
    }
    free(inputs);

    free(fake_inputs);
    free(fake_targets);

    free_otternetwork(net);

    printf("Network freed\n");
    return 0;
}