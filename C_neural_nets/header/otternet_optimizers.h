#ifndef OTTERNET_OPTIMIZERS_H
#define OTTERNET_OPTIMIZERS_H
#include "../header/ottertensors.h"
#include "../header/ottertensors_utilities.h"
#include "../header/ottertensors_operations.h"
#include "../header/ottermath.h"
#include "../header/OtterLayers.h"
#include "../header/otternet.h"



typedef struct Otternetwork Otternetwork;

typedef struct Otterchain Otterchain;

void ON_SGD(Otternetwork* network, OtterTensor** input, OtterTensor** labels);
void ON_SGD_fit(Otternetwork* network, OtterDataset* inputs, OtterDataset* labels, int epochs, int batch_size);
//void ON_SGDM_fit(Otternetwork* network, OtterDataset* inputs, OtterDataset* labels, int epochs, int batch_size);
//void ON_verbose1(int epoch,Otternetwork* network, OtterDataset* inputs, OtterDataset* labels) ;
void ON_first_moment_estimation(OtterTensor* momentum, OtterTensor* gradient, float beta1);
//void ON_Adam_fit(Otternetwork* network, OtterDataset* inputs, OtterDataset* labels, int epochs, int batch_size);
void ON_second_moment_estimation(OtterTensor* velocity, OtterTensor* gradient, float beta2);
//OtterTensor*** ON_deepcopy(OtterTensor*** params, int num_layers);

#endif