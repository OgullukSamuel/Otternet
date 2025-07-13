#ifndef OTTERACTIVATION_H
#define OTTERACTIVATION_H

#include "ottertensors.h"
#include "ottertensors_utilities.h"
#include "ottertensors_operations.h"
#include "ottermath.h"

#include <stdio.h>



typedef struct Activation_function{
    const char* name;
    void (*activation)(OtterTensor*);
    void (*derivative)(OtterTensor*);
} Activation_function;

void derivative_activation_functions(char* function_name, OtterTensor* x);
void Activation_functions(char* function_name, OtterTensor* x);








#endif