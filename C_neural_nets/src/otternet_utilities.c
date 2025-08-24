#include "../header/otternet_utilities.h"


/* int get_layer_type(void* layer) {
    if (layer == NULL) {
        fprintf(stderr, "Layer is NULL.\n");
        exit(EXIT_FAILURE);
    }
    if (((Dense_layer*)layer)->type == 0) {
        return 0; // Dense layer
    } else if (((Conv1D_layer*)layer)->type == 1) {
        return 1; // Conv1D layer
    } else {
        fprintf(stderr, "Unknown layer type.\n");
        exit(EXIT_FAILURE);
    }
}
 */

/*
int get_full_size_of_OTN(Otternetwork* network) {
    if (network == NULL) {
        fprintf(stderr, "Network is NULL.\n");
        exit(EXIT_FAILURE);
    }
    int size = 0;
    for (int i = 0; i < network->num_layers; i++) {
        size += network->layers[i]->weights->size + network->layers[i]->biases->size;
    }
    return size;
}

*/
int argmin(int* distances, int size) {
    int min_index = -1;
    int min_value = 9999999;
    for (int i = 0; i < size; i++) {
        if (distances[i] < min_value) {
            min_value = distances[i];
            min_index = i;
        }
    }
    return min_index;
}

void rankify(int* input, int* output, int size) {
    for (int i = 0; i < size; i++) {
        int rank = 0;
        for (int j = 0; j < size; j++) {
            if (input[j] < input[i]) rank++;
        }
        output[i] = rank;
    }
}
int find_index(Otterchain** list, int size, Otterchain* target) {
    for (int i = 0; i < size; i++) {
        if (list[i] == target) return i;
    }
    return -1;
}


void CHECK_NAN_TENSOR(const OtterTensor* t, const char* where){
    if(!t || !t->data) return;
    size_t n = 1;
    for(int i=0;i<t->rank;i++) n *= t->dims[i];
    for(size_t i=0;i<n;i++){
        float v = t->data[i];
        if(isnan(v) || isinf(v)){
            fprintf(stderr,"NaN/Inf in %s at %s\n", where, __FILE__);
            /* dump minimal context */
            fprintf(stderr,"tensor ptr=%p idx=%zu val=%f\n", (void*)t, i, (double)v);
            abort();
        }
    }
}