#include "../header/ottertensors_utilities.h"

void print_tensor_recursive(OtterTensor* t, int level, int ndims,int idx,int significant_digits) {
    if (level == ndims - 1) {
        printf("[");
        for (int i = 0; i < t->dims[level]; i++) {
            printf("%.*f",significant_digits,t->data[idx+i]);
            if (i != t->dims[level] - 1) {
                printf(",");
            }
        }
        printf("]");
    } else {
        // Dimensions intermédiaires : on ouvre une liste, on appelle récursivement
        printf("[");
        for (int i = 0; i < t->dims[level]; i++) {
            print_tensor_recursive(t, level + 1, ndims, idx + i * t->strides[level], significant_digits);
            if (i != t->dims[level] - 1) {
                printf(",");
            }
        }
        printf("]");
    }
}


void print_tensor(OtterTensor* t,int significant_digits) {
    printf("Ottertensor with shape: (");
    for (int i = 0; i < t->rank; i++) {
        printf("%d", t->dims[i]);
        if (i < t->rank - 1) {
            printf(", ");
        }
    }
    printf(")\n");
    printf("Ottertensor(");
    if (t->rank ==0) {
        printf("%.*f",significant_digits,t->data[0]);
    } else {
        
        //printf("[");

        print_tensor_recursive(t, 0, t->rank, 0, significant_digits);
        //printf("]");
    }
    printf(")\n");
}


OtterTensor* OT_copy(OtterTensor* a){
    OtterTensor* result = OT_zeros(a->dims, a->rank);
    
    for (int i = 0; i < a->size; i++) {
        result->data[i] = a->data[i];
    }
    
    return result;
}

void OT_initialize_copy(OtterTensor* a, OtterTensor* copy){
    copy->dims = NULL;
    copy->strides = NULL;
    copy->data = NULL;
    set_dims(copy, a->dims, a->rank);
    copy->data = malloc(copy->size * sizeof(float));
    for (int i = 0; i < a->size; i++) {
        copy->data[i] = a->data[i];
    }
    return;
}


OtterTensor* OT_Flatten(OtterTensor* t) {
    OtterTensor* flat_tensor = OT_copy(t);
    for (int i = 0; i < t->size; i++) {
        flat_tensor->data[i] = t->data[i];
    }
    
    return flat_tensor;
}




OtterTensor* OT_zeros(int* dims, int rank){
    OtterTensor* tensor = malloc(sizeof(OtterTensor));
    tensor->dims = NULL;
    tensor->strides = NULL;
    tensor->data = NULL;
    set_dims(tensor, dims, rank);
    tensor->data = malloc(tensor->size * sizeof(float));
    for(int i = 0; i < tensor->size; i++) {
        tensor->data[i] = 0.0f;
    }
    return tensor;
}

OtterTensor* OT_ones(int* dims, int rank){
    OtterTensor* tensor = malloc(sizeof(OtterTensor));
    tensor->dims = NULL;
    tensor->strides = NULL;
    tensor->data = NULL;
    set_dims(tensor, dims, rank);
    tensor->data = malloc(tensor->size * sizeof(float));
    for(int i = 0; i < tensor->size; i++) {
        tensor->data[i] = 1.0f;
    }
    return tensor;
}