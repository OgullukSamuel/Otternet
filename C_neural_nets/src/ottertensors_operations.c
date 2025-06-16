#include "../header/ottertensors_operations.h"

void OT_ref_tensors_sum(OtterTensor* a, OtterTensor* b) {

    if (a->rank != b->rank ) {
        fprintf(stderr, "Tensors must have the same rank for addition.\n found rank %i and %i\n",a->rank,b->rank);
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < a->size; i++) {
        a->data[i] = a->data[i] + b->data[i];
    }
    return;
}

void OT_ref_tensors_substract(OtterTensor* a, OtterTensor* b) {
    if (a->rank != b->rank ) {
        fprintf(stderr, "Tensors must have the same rank for subtraction.\n");
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < a->size; i++) {
        a->data[i] = a->data[i] - b->data[i];
    }
    return;
}




OtterTensor* OT_tensors_sum(OtterTensor* a, OtterTensor* b) {
    OtterTensor* result=OT_zeros(a->dims, a->rank);

    if (a->rank != b->rank || a->rank != result->rank) {
        fprintf(stderr, "Tensors must have the same rank for addition.\n found rank %i and %i \n",a->rank,b->rank);
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < a->size; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }
    return result;
}

OtterTensor* OT_tensors_substract(OtterTensor* a, OtterTensor* b) {
    OtterTensor* result=OT_zeros(a->dims, a->rank);
    if (a->rank != b->rank || a->rank != result->rank) {
        fprintf(stderr, "Tensors must have the same rank for subtraction.\n");
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < a->size; i++) {
        result->data[i] = a->data[i] - b->data[i];
    }
    return result;
}

OtterTensor* OT_Matrix_multiply(OtterTensor* a, OtterTensor* b) {
    if(a->rank>2 || b->rank>2 ){
        printf("matrix multiplcation is only possible for rank 2 and 1 matrices");
        exit(EXIT_FAILURE);
    }
    else if(a->rank==0){
        return(OT_scalar_multiply(b,a->data[0]));
    } else if (b->rank==0){
        return(OT_scalar_multiply(a,b->data[0]));
    }else if (a->dims[1] != b->dims[0]) {
        printf( "Inner dimensions must match for matrix multiplication.\n");
        exit(EXIT_FAILURE);
    }else {
        OtterTensor* result=OT_zeros((int[2]){a->dims[0],b->dims[1]},2);
        for(int i =0;i<a->dims[0];i++){
            for(int j = 0 ; j<b->dims[1];j++){
                for(int k = 0 ; k<a->dims[1];k++){
                    result->data[index_tensor(result,(int[2]){i,j})] += a->data[index_tensor(a,(int[2]){i,k})] * b->data[index_tensor(b,(int[2]){k,j})];
                }
            }
        }
        return result;
    }
}


OtterTensor* OT_dot(OtterTensor* a, OtterTensor* b) {
    OtterTensor* result = OT_zeros(a->dims, a->rank);
    if (a->rank != b->rank || a->rank != result->rank) {
        fprintf(stderr, "Tensors must have the same rank for multiplication.\n");
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < a->size; i++) {
        result->data[i] = a->data[i] * b->data[i];
    }
    return result;
}



OtterTensor* OT_scalar_multiply(OtterTensor* main, float lambda) {
    OtterTensor* result=OT_zeros(main->dims,main->rank);
    for (int i = 0; i < main->size; i++) {
        result->data[i] = lambda * main->data[i];
    }
    return result;
}

OtterTensor* OT_scalar_add(OtterTensor* main, float lambda) {
    OtterTensor* result=OT_zeros(main->dims, main->rank);
    for (int i = 0; i < main->size; i++) {
        result->data[i] = main->data[i] + lambda;
    }
    return result;
}

OtterTensor* OT_scalar_subtract(OtterTensor* main, float lambda) {
    OtterTensor* result=OT_zeros(main->dims,main->rank);
    for (int i = 0; i < main->size; i++) {
        result->data[i] = main->data[i] - lambda;
    }
    return result;
}


OtterTensor* OT_Transpose(OtterTensor* t) {
    if (t->rank != 2) {
        fprintf(stderr, "Transpose is only defined for 2D tensors.\n");
        exit(EXIT_FAILURE);
    }
    OtterTensor* transposed = OT_copy(t);
    int temp[2]={t->dims[1],t->dims[0]};
    set_dims(transposed, temp, 2);
    return transposed;
}