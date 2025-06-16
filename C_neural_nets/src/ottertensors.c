#include "../header/ottertensors.h"


void compute_strides(OtterTensor *t) {
    t->strides[t->rank-1] = 1;
    for (int i = t->rank-2; i >= 0; --i) {
        t->strides[i] = t->strides[i + 1] * t->dims[i + 1];
    }
}

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

void set_dims(OtterTensor* t, int* dimensions, int rank) {
    if (t->dims) free(t->dims);
    if (t->strides) free(t->strides);
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
    if (tensor->data) free(tensor->data);
    if (tensor->dims) free(tensor->dims);
    if (tensor->strides) free(tensor->strides);
    return;
}

void free_malloc_tensor(OtterTensor* tensor) {
    if (!tensor) return;
    if (tensor->data) free(tensor->data);
    if (tensor->dims) free(tensor->dims);
    if (tensor->strides) free(tensor->strides);
    free(tensor);
    return;
}


void Init_dataset(int size){
    OtterDataset* dataset= malloc(sizeof(OtterDataset));
    dataset->size=size;
    
}