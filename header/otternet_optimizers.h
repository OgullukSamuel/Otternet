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


typedef struct Momentums{
    OtterTensor** first_moment_weights;
    OtterTensor** first_moment_biases; 
    OtterTensor** second_moment_weights;
    OtterTensor** second_moment_biases;
    int t;
} Momentums;

void ON_SGD(Otternetwork* network, OtterTensor** input, OtterTensor** labels);
void ON_SGD_fit(Otternetwork* network, OtterDataset* inputs, OtterDataset* labels, int epochs, int batch_size);
void ON_SGDM_fit(Otternetwork* network, OtterDataset* inputs, OtterDataset* labels, int epochs, int batch_size) ;
void ON_verbose1(int epoch, Otternetwork* network, OtterDataset* inputs, OtterDataset* labels, int* indices,int batch_size) ;

void ON_Adam_fit(Otternetwork* network, OtterDataset* inputs, OtterDataset* labels, int epochs, int batch_size);

Momentums* init_first_Momentums(Otternetwork* network);
void init_second_Momentums(Otternetwork* network,Momentums* momentums) ;

void ON_moments_estimation(Otternetwork* net, Momentums* momentums,float beta1, float beta2,float epsilon,int t);
void ON_update_gradients(Otternetwork* net, OtterTensor* first_moment, OtterTensor* second_moment, float epsilon,OtterTensor** weights_grad,float norm1, float norm2);
void ON_update_first_moment(OtterTensor* first_moment, OtterTensor* gradient, float beta1);
void ON_update_second_moment(OtterTensor* second_moment, OtterTensor* gradient, float beta2);


void free_first_momentums(Otternetwork* network,Momentums* momentums) ;
void free_all_momentums(Otternetwork* network,Momentums* momentums) ;


#endif