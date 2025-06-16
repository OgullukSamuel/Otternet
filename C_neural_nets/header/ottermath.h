#ifndef OTTERMATH_H
#define OTTERMATH_H
#include "ottertensors.h"
#include "ottertensors_utilities.h"
#include "ottertensors_operations.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <math.h>


#define LN2 0.693147180559945f
#define LOG10 0.30102999566f
float OM_exp(float x);
float OM_ldexp(float x, int n);
float OM_sigmoid(float x);
float OM_log2(float x);
float OM_log10(float x);
float OM_ln(float x);
float OM_tanh(float x);
float OM_relu(float x);
float OM_prelu(float x, float alpha);
float OM_leaky_relu(float x, float alpha);
float OM_elu(float x, float alpha);
float OM_heaviside(float x);
float OM_dsigmoid(float x);
float OM_dtanh(float x) ;

OtterTensor* OM_softmax(OtterTensor* input);

void OM_ref_softmax(OtterTensor* input) ;
OtterTensor* OM_softmax_with_temperature(OtterTensor* input, float temperature);

float OM_cross_entropy(OtterTensor* predictions, OtterTensor* truth);


OtterTensor* Vectorize(float x, float (*func)(float));

void OM_ref_Vectorize(OtterTensor* x, float (*func)(float));



#endif