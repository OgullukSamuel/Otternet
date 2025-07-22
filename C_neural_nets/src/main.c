#include "../header/main.h"




void display_order(Otternetwork* net) {
    printf("Network order:\n");
    for (int i = 0; i < net->num_layers; i++) {
        printf("layer : %p",(net->order[i]));
    }
    printf("\n");
}


// ...existing code...

int main() {

    // Construire le réseau
    Otterchain* dense_1 = ON_Dense_layer(4, "relu", NULL, 0);
    Otterchain* dense_2 = ON_Dense_layer(1, "linear", dense_1, 1);
    Otterchain* dense_3 = ON_Dense_layer(1, "linear", dense_2, 1);
    Otterchain* dense_4 = ON_Dense_layer(2, "linear", dense_2, 1);
    Otterchain* dense_5 = ON_Dense_layer(2, "linear", dense_4, 1);
    
    printf("layers pointers : %p, %p,%p,%p\n", dense_1, dense_2,dense_3, dense_4);

    Otternetwork* net = ON_initialise_otternetwork();
    printf("Network initialised\n");
    ON_add_layer(net, dense_1);
    ON_add_layer(net, dense_2);
    ON_add_layer(net, dense_3);
    ON_add_layer(net, dense_4);
    ON_add_layer(net, dense_5);
    printf("Layers added to the network\n");
    ON_compile_otternetwork(net, "SGD", "MSE", 0.01f, NULL);
    printf("Network compiled\n");
    printf("%i\n",net->num_layers);
    display_order(net);
    ON_display_network(net);

    // === FAUSSES DONNÉES DE TEST ===
    int dims[2] = {4, 1}; // 4 features, 1 colonne
    OtterTensor* fake_input = OT_zeros(dims, 2);
    for (int i = 0; i < fake_input->size; i++) {
        fake_input->data[i] = (float)(i + 1); // [1, 2, 3, 4]
    }
    printf("Input tensor avant feed forward :\n");
    print_tensor(fake_input, 3);

    // === FEED FORWARD ===
    OtterTensor* output = ON_feed_forward(net, fake_input, 0);
    printf("Output tensor après feed forward :\n");
    print_tensor(output, 3);

    free_malloc_tensor(fake_input);
    free_malloc_tensor(output);

    free_otternetwork(net);

    printf("Network freed\n");
    return 0;
}
