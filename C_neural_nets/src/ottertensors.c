#include "../header/ottertensors.h"




int index_tensor(OtterTensor *t, int* idx) {
    int index = 0;
    for (int i=0;i<t->rank;i++) {
        index += idx[i] * t->strides[i];
    }
    return index;
}

float get(OtterTensor* t, int* idx) {
    return t->data[index_tensor(t, idx)];
}


void compute_strides(OtterTensor *t) {
    t->strides[t->rank - 1] = 1;
    for (int i = t->rank - 2; i >= 0; i--) {
        t->strides[i] = t->strides[i + 1] * t->dims[i + 1];
    }
}

void set_dims(OtterTensor* t, int* dimensions, int rank) {
    t->rank = rank;
    t->dims = malloc(rank * sizeof(int));
    t->strides = malloc(rank * sizeof(int));
    t->size = 1;
    for (int i = 0; i < t->rank; i++) {
        t->dims[i] = dimensions[i];
        t->size *= dimensions[i];
    }
    compute_strides(t);
}

void set(OtterTensor *t, int* index, float value) {
    t->data[index_tensor(t, index)] = value;
}



void free_tensor(OtterTensor* tensor) {
    if (!tensor) return;
    if (tensor->data) { free(tensor->data); tensor->data = NULL; }
    if (tensor->dims) { free(tensor->dims); tensor->dims = NULL; }
    if (tensor->strides) { free(tensor->strides); tensor->strides = NULL; }
}

void free_malloc_tensor(OtterTensor** tensor) {
    if (!tensor || !*tensor) return;
    free_tensor(*tensor);
    free(*tensor);
    *tensor = NULL;
}




void free_ottertensor_list(OtterTensor** tensors, int count) {
    if (!tensors) return;
    
    for (int i = 0; i < count; i++) {
        if (tensors[i]) {
            free_malloc_tensor(&tensors[i]);
        }
    }
    
    free(tensors);
}






OtterDataset* Init_dataset(OtterTensor*** data,int num_input, int size_dataset){
    OtterDataset* dataset= malloc(sizeof(OtterDataset));
    dataset->size[0] = size_dataset;
    dataset->size[1] = num_input;
    
    dataset->dataset = data;    
    return(dataset);
}

void OD_free_dataset(OtterDataset* dataset){
    if(!dataset) return;
    
    if(dataset->dataset){
        for(int i=0;i<dataset->size[0];i++){
            if(dataset->dataset[i]){
                for(int j=0;j<dataset->size[1];j++){
                    free_malloc_tensor(&dataset->dataset[i][j]);
                }
            }
            free(dataset->dataset[i]);
            dataset->dataset[i] = NULL;
        }
        free(dataset->dataset);
        dataset->dataset = NULL;
    }
    free(dataset);
    dataset = NULL;
    return;
}
