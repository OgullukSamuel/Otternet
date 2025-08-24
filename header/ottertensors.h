#ifndef OTTER_TENSORS_H
#define OTTER_TENSORS_H
#include <stdio.h>
#include <stdlib.h>


typedef struct {
    float* data;
    int rank;
    int* dims;     // Sizes of each dimension
    int* strides;  // Strides for indexing
    int size;
    int freed;
} OtterTensor;

typedef struct {
    OtterTensor*** dataset;
    int size[2];
} OtterDataset;

void compute_strides(OtterTensor *t);
void set_dims(OtterTensor* t, int* dimensions, int rank);
void set(OtterTensor *t, int* index, float value);
float get(OtterTensor* t, int* idx);
void free_tensor(OtterTensor* tensor);
void free_malloc_tensor(OtterTensor** tensor);
int index_tensor(OtterTensor *t, int* idx);
OtterDataset* Init_dataset(OtterTensor*** data,int num_input, int size_dataset);
void free_ottertensor_list(OtterTensor** tensors, int count);
void OD_free_dataset(OtterDataset* dataset);

#endif