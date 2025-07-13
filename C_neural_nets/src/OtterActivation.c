#include "../header/OtterActivation.h"

Activation_function activation_table[] = {
    {"relu", OM_tensor_relu, OM_tensor_heaviside},
    {"sigmoid", OM_tensor_sigmoid, OM_tensor_dsigmoid},
    {"tanh", OM_tensor_tanh, OM_tensor_dtanh},
    {"softmax", OM_ref_softmax, NULL}, // gestion spéciale
    {"linear", OM_tensor_linear, OM_tensor_ones},
    {NULL, NULL, NULL}
};


void Activation_functions(char* function_name, OtterTensor* x) {
    for (int i = 0; activation_table[i].name != NULL; i++) {
        if (strcmp(function_name, activation_table[i].name) == 0) {
            if (activation_table[i].activation) activation_table[i].activation(x);
            return;
        }
    }
    fprintf(stderr, "Unknown activation function: %s\n", function_name);
    exit(EXIT_FAILURE);
}

void derivative_activation_functions(char* function_name, OtterTensor* x) {
    for (int i = 0; activation_table[i].name != NULL; i++) {
        if (strcmp(function_name, activation_table[i].name) == 0) {
            if (activation_table[i].derivative) activation_table[i].derivative(x);
            else for (int i = 0; i < x->size; i++) x->data[i] = 1.0f; // linear case
            return;
        }
    }
    fprintf(stderr, "Unknown activation function: %s\n", function_name);
    exit(EXIT_FAILURE);
}