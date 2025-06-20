#ifndef OTTERNET_OPTIMIZERS_H
#define OTTERNET_OPTIMIZERS_H
#include "../header/ottertensors.h"
#include "../header/ottertensors_utilities.h"
#include "../header/ottertensors_operations.h"
#include "../header/ottermath.h"
#include "../header/otternet.h"





void ON_SGD(Dense_network* network, OtterTensor* input, OtterTensor* labels,OtterTensor*** params);
void ON_SGD_fit(Dense_network* network, OtterDataset* inputs, OtterDataset* labels, int epochs, int batch_size);
void ON_SGDM_fit(Dense_network* network, OtterDataset* inputs, OtterDataset* labels, int epochs, int batch_size);
void ON_verbose1(int epoch,Dense_network* network, OtterDataset* inputs, OtterDataset* labels) ;
void ON_first_moment_estimation(OtterTensor* momentum, OtterTensor* gradient, float beta1);
void ON_Adam_fit(Dense_network* network, OtterDataset* inputs, OtterDataset* labels, int epochs, int batch_size);
void ON_second_moment_estimation(OtterTensor* velocity, OtterTensor* gradient, float beta2);
OtterTensor*** ON_deepcopy(OtterTensor*** params, int num_layers);

#endif