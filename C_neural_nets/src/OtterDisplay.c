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