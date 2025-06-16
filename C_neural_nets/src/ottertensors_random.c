#include "../header/ottertensors_random.h"


OtterTensor* OT_random_uniform(int* dims, int rank, float min, float max) {
    OtterTensor* tensor = malloc(sizeof(OtterTensor));
    tensor->dims = NULL;
    tensor->strides = NULL;
    tensor->data = NULL;
    set_dims(tensor, dims, rank);
    tensor->data = malloc(tensor->size * sizeof(float));
    srand((unsigned int)time(NULL));
    for (int i = 0; i < tensor->size; i++) {
        tensor->data[i] = min + (float)rand() / (float)(RAND_MAX / (max - min));
    }
    return tensor;
}

int* OR_select_batch(int total_size, int batch_size) {
    if (batch_size > total_size){
        printf("batch_size must be smaller than the total size.\n");
        return NULL;
    }
    int *arr = malloc(total_size * sizeof(int));

    for (int i = 0; i < total_size; i++){arr[i] = i;}

    for (int i = 0; i < batch_size; i++) {
        int j = i + rand() % (total_size - i);
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
    int *batch = malloc(batch_size * sizeof(int));
    for (int i = 0; i < batch_size; i++) {batch[i] = arr[i];}
    free(arr);
    return batch;
}