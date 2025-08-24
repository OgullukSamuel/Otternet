#include "../header/OtterDisplay.h"

void ON_display_network(Otternetwork* network){
    printf("\n============================================\n");
    printf("Network with %i layers, %i input layers, %i output layers \n",network->num_layers,network->num_start_of_line, network->num_end_of_line);
    printf("The network structure is the following : \n");
    printf("============================================\n");
    printf("| Layer |  number of parameters | connection  |\n");
    printf("============================================\n");
    for(int j=0;j<network->num_layers;j++){
        if(network->order[j]->connections_backward!= NULL){
            printf("|  %s  |   %i  | \n", LAYER_TYPE[network->order[j]->type],network->order[j]->connections_backward[0]->network_rank);
        } else {
            printf("|  %s  |   %i  | \n", LAYER_TYPE[network->order[j]->type],-1);
        }
    }
    printf("============================================\n");

    return;
}

void ON_display_network_connections(Otternetwork* network) {
    printf("\n============================================\n");
    printf("Network with %i layers, %i input layers, %i output layers\n",
           network->num_layers, network->num_start_of_line, network->num_end_of_line);
    printf("The network structure is the following:\n");
    printf("============================================\n");
    printf("| Layer | Parameters | Backward connections | Forward connections |\n");
    printf("============================================\n");
    
    /*
            Otterchain* layer = network->order[j];
        int total_params = 0;

        for (int k = 0; k < layer->weights_depth; k++) {
            if (layer->weights[k]) {
                total_params += layer->weights[k]->size;
            }
            if (layer->biases[k]) {
                total_params += layer->biases[k]->size;
            }
        }*/


    for (int j = 0; j < network->num_layers; j++) {
        Otterchain* layer = network->order[j];

        // Nombre de paramètres : sommaire simple (poids + biais)

        // Connexions backward
        if (layer->connections_backward != NULL && layer->num_connections_backward > 0) {
            printf("| %s | %d | ", LAYER_TYPE[layer->type], 1);
            for (int k = 0; k < layer->num_connections_backward; k++) {
                printf("%d ", layer->connections_backward[k]->network_rank);
            }
        } else {
            printf("| %s | %d | -1 ", LAYER_TYPE[layer->type], 2);
        }

        // Connexions forward
        if (layer->connections_forward != NULL && layer->num_connections_forward > 0) {
            printf("| ");
            for (int k = 0; k < layer->num_connections_forward; k++) {
                printf("%d ", layer->connections_forward[k]->network_rank);
            }
        } else {
            printf("| -1 ");
        }

        printf("|\n");
    }
    printf("============================================\n");
}


void print_parameters(Otternetwork* network) {
    printf("\n============================================\n");
    printf("Network Parameters:\n");
    printf("============================================\n");

    for (int j = 0; j < network->num_layers; j++) {
        printf("Layer %d (%s):\n", j, LAYER_TYPE[network->order[j]->type]);
        Otterchain* layer = network->order[j];
        for(int i=0;i<layer->weights_depth;i++){
            if (layer->weights[i]) {
                printf("  Weights %d:\n", i);
                print_tensor(layer->weights[i], 2);
            }
            if (layer->biases[i]) {
                printf("  Biases %d:\n", i);
                print_tensor(layer->biases[i], 2);
            }
            printf("---------\n");
        }

        
    }

    printf("============================================\n");
}