#include "../header/otternet.h"






void Activation_functions(char* function_name, OtterTensor* x) {                                           //fonction à travailler
    if (strcmp(function_name, "linear") == 0 || strlen(function_name) == 0) {
        return;
    } else if (strcmp(function_name, "relu") == 0) {
        OM_ref_Vectorize(x, &OM_relu);
        return;
    } else if (strcmp(function_name, "sigmoid") == 0) {
        OM_ref_Vectorize(x, &OM_sigmoid);
        return;
    } else if (strcmp(function_name, "tanh") == 0) {
        OM_ref_Vectorize(x, &OM_tanh);
        return;
    } else if (strcmp(function_name, "softmax") == 0) {
        OM_ref_softmax(x);
        return;
    } else {
        fprintf(stderr, "Unknown activation function: %s\n", function_name);
        exit(EXIT_FAILURE);
    }
}

void derivative_activation_functions(char* function_name, OtterTensor* x) {                                           //fonction à travailler
    if (!function_name || strlen(function_name) == 0 || strcmp(function_name, "linear") == 0) {
        for(int i = 0; i < x->size; i++) {
            x->data[i] = 0;
        }
        return;
    }
    
    if (strcmp(function_name, "linear") == 0 || strlen(function_name) == 0) {
        for(int i =0;i<x->size;i++){
            x->data[i]=0;
        }

    } else if (strcmp(function_name, "relu") == 0) {
        OM_ref_Vectorize(x,&OM_heaviside);
        return;
    } else if (strcmp(function_name, "sigmoid") == 0) {
        OM_ref_Vectorize(x, &OM_dsigmoid);
        return;
    } else if (strcmp(function_name, "tanh") == 0) {
        OM_ref_Vectorize(x, &OM_dtanh);
        return;
    } else if (strcmp(function_name, "softmax") == 0) {
        OM_ref_softmax(x);
        return;
    } else {
        fprintf(stderr, "Unknown activation function: %s\n", function_name);
        exit(EXIT_FAILURE);
    }
}




int get_full_size_of_DN(Dense_network* network){
    int size=0;
    for(int i =0; i<network->num_layers; i++){
        size += network->layers[i]->weights.size + network->layers[i]->biases.size;
    }
    return(size);
}

Dense_network* ON_initialise_network(int* dense_layers,int num_layers,char** activation_functions){
    Dense_network* network = malloc(sizeof(Dense_network));
    network->layers = malloc(num_layers * sizeof(Dense_layer*));
    for(int i = 0; i < num_layers; i++) {
        Dense_layer* layer = malloc(sizeof(Dense_layer));
        if (i >0) {
            OtterTensor* weights = OT_random_uniform((int[2]){dense_layers[i], dense_layers[i-1]}, 2, -1.0f, 1.0f);
            OtterTensor* biases = OT_zeros((int[2]){dense_layers[i],1}, 2);
            OT_initialize_copy(weights, &layer->weights);
            OT_initialize_copy(biases, &layer->biases);
            free_malloc_tensor(weights);
            free_malloc_tensor(biases);   
            layer->activation_function = activation_functions[i];
        } else {
            int zero = 0;
            OtterTensor* w = OT_zeros(&zero, 1);
            OtterTensor* b = OT_zeros((int[2]){dense_layers[i],1}, 2);
            OT_initialize_copy(w, &layer->weights);
            OT_initialize_copy(b, &layer->biases);
            free_malloc_tensor(w);
            free_malloc_tensor(b);         
            layer->activation_function = NULL;
        }
        layer->num_neurons = dense_layers[i];
        network->layers[i] = layer;
    }
    network->num_layers = num_layers;
    return network;
}

void ON_display_network(Dense_network* network){
    int full_param = get_full_size_of_DN(network);
    printf("Network with %i layers, for a total of %i parameters \n",network->num_layers,full_param);
    printf("The network structure is the following : \n");
    printf("============================================\n");
    printf("| Layer |  number of neurons | Parameters  |\n");
    printf("============================================\n");
    for(int j=0;j<network->num_layers;j++){
        printf("|  %i  |  %i   | %i\n ", j, network->layers[j]->num_neurons,network->layers[j]->weights.size+network->layers[j]->biases.size);
    }
    return;
}

OtterTensor* layer_calc(OtterTensor* input, Dense_layer* layer, OtterTensor* zs, OtterTensor* activation){
    OtterTensor* prod= OT_Matrix_multiply(&layer->weights,input);
    OT_ref_tensors_sum(prod,&layer->biases);
    if(zs!=NULL){zs=OT_copy(prod);}
    if (layer->activation_function != NULL) {
        Activation_functions(layer->activation_function, prod);
    }
    if(activation!=NULL){activation=OT_copy(prod);}
    return(prod);
}



OtterTensor* ON_feed_forward(Dense_network* network, OtterTensor* input, OtterTensor** zs, OtterTensor** activations) {
    OtterTensor* last_values = OT_copy(input);
    if (activations) activations[0] = OT_copy(input);
    printf("1%i,%i\n",last_values->rank,network->layers[0]->biases.rank);
    OT_ref_tensors_sum(last_values, &network->layers[0]->biases);
    if (zs) zs[0] = OT_copy(last_values);

    for (int i = 1; i < network->num_layers; i++) {
        OtterTensor* prod = OT_Matrix_multiply(&network->layers[i]->weights, last_values);
        printf("2%i,%i\n",prod->rank,network->layers[i]->biases.rank);
        OT_ref_tensors_sum(prod, &network->layers[i]->biases);
        if (zs) zs[i] = OT_copy(prod);
        if (network->layers[i]->activation_function != NULL) {
            Activation_functions(network->layers[i]->activation_function, prod);
        }
        if (activations) activations[i] = OT_copy(prod);
        free_tensor(last_values);
        free(last_values);
        last_values = prod;
    }
    return last_values;
}

void ON_compile_network(Dense_network* network, char* optimizer, char* error_function, float learning_rate) {
    network->optimizer = optimizer;
    network->error_function = error_function;
    network->learning_rate = learning_rate;
    if (strcmp(optimizer, "SGD") != 0) {
        fprintf(stderr, "Unknown optimizer: %s\n", optimizer);
        exit(EXIT_FAILURE);
    }
    return;
}

OtterTensor* ON_Cout(OtterTensor* output, OtterTensor* labels,char* error_function){
    // Ici on prend deux vecteurs de rang 2
    if(strcmp(error_function,"MSE")==0){
        OtterTensor* cout = OT_tensors_substract(output,labels);
        OtterTensor* temp = OT_Transpose(cout);
        OtterTensor* final = OT_Matrix_multiply(cout,temp);
        free_malloc_tensor(temp);
        free_malloc_tensor(cout);
        return(final);

    }
    else{
        fprintf(stderr, "Unknown error function: %s\n", error_function);
        exit(EXIT_FAILURE);
    }
}

OtterTensor** ON_copy_weights(Dense_network* network){
    OtterTensor** tensor = malloc(network->num_layers* sizeof(OtterTensor*));
    for(int i=0; i<network->num_layers;i++){
        tensor[i]=OT_zeros(network->layers[i]->weights.dims,network->layers[i]->weights.rank);
    }
    return tensor;
}

OtterTensor** ON_copy_biases(Dense_network* network){
    OtterTensor** tensor = malloc(network->num_layers* sizeof(OtterTensor*));
    for(int i=0; i<network->num_layers;i++){
        tensor[i]=OT_zeros(network->layers[i]->biases.dims,network->layers[i]->biases.rank);
    }
    return tensor;
}

OtterTensor** ON_local_error_def(Dense_network* network){
    OtterTensor** tensor = malloc(network->num_layers* sizeof(OtterTensor*));
    for(int i=0; i<network->num_layers;i++){
        tensor[i]=OT_zeros(&network->layers[i]->num_neurons,1);
    }
    return tensor;
}

void ON_SGD(Dense_network* network,OtterTensor* input, OtterTensor* labels){
    int L= network->num_layers;
    OtterTensor** weights_gradients = ON_copy_weights(network);
    OtterTensor** biases_gradients = ON_copy_biases(network);
    OtterTensor** local_errors = ON_local_error_def(network);
    OtterTensor** zs = malloc(L*sizeof(OtterTensor*));
    OtterTensor** activations = malloc(L*sizeof(OtterTensor*));
    
    OtterTensor* predictions=ON_feed_forward(network,input,zs,activations);
    
    OtterTensor* cout = ON_Cout(predictions, labels,network->error_function);
    
    free_malloc_tensor(predictions);
    derivative_activation_functions(network->layers[L-1]->activation_function,zs[L-1]);
    local_errors[L-1]= OT_dot(cout,zs[L-1] );
    free_malloc_tensor(cout);
    biases_gradients[L-1]=OT_copy(local_errors[L-1]);
    weights_gradients[L-1]=OT_dot(local_errors[L-1],activations[L-1]);

    for(int layer=L-2;layer>0;layer--){
        OtterTensor* w_T=OT_Transpose(&network->layers[layer+1]->weights);
        OtterTensor* A = OT_Matrix_multiply(w_T, local_errors[layer+1]);
        derivative_activation_functions(network->layers[layer]->activation_function,zs[layer]);
        local_errors[layer]=OT_dot(A,zs[layer]);
        biases_gradients[layer]=OT_copy(local_errors[layer]);
        weights_gradients[layer]=OT_dot(local_errors[layer],activations[layer]);
        free_malloc_tensor(w_T);
        free_malloc_tensor(A);
    
    }

    for(int i=1; i<L; i++){
    printf("3 : %i,%i",network->layers[i]->weights.rank,weights_gradients[i]->rank);
    printf("4 : %i,%i",network->layers[i]->biases.rank,biases_gradients[i]->rank);
    OtterTensor* gradlr = OT_scalar_multiply(weights_gradients[i],-network->learning_rate);
    OtterTensor* gradlr_b = OT_scalar_multiply(biases_gradients[i],-network->learning_rate);
    
    // ADD THESE CHECKS:
    if (network->layers[i]->weights.size != gradlr->size || network->layers[i]->weights.rank != gradlr->rank) {
        fprintf(stderr, "Shape mismatch in weights update at layer %d\n", i);
        exit(EXIT_FAILURE);
    }
    if (network->layers[i]->biases.size != gradlr_b->size || network->layers[i]->biases.rank != gradlr_b->rank) {
        fprintf(stderr, "Shape mismatch in biases update at layer %d\n", i);
        exit(EXIT_FAILURE);
    }
    
    OT_ref_tensors_sum(&network->layers[i]->weights,gradlr);
    OT_ref_tensors_sum(&network->layers[i]->biases,gradlr_b);
    free_malloc_tensor(gradlr);
    free_malloc_tensor(gradlr_b);
    }
    for(int i=0;i<L;i++){
        free_malloc_tensor(weights_gradients[i]);
        free_malloc_tensor(biases_gradients[i]);
        free_malloc_tensor(local_errors[i]);
        free_malloc_tensor(zs[i]);
        free_malloc_tensor(activations[i]);
    }

    
    free(weights_gradients);
    free(biases_gradients);
    free(local_errors);
    free(zs);
    free(activations);

    return;
}

OtterTensor* ON_predict(Dense_network* network, OtterTensor* input) {
    OtterTensor* predictions = ON_feed_forward(network, input, NULL, NULL);
    return predictions;
}

void ON_fit(Dense_network* network, OtterDataset* inputs, OtterDataset* labels, int epochs,int batch_size){
    for(int epoch=0;epoch<epochs;epoch++){
        int* indices= OR_select_batch(inputs->size,batch_size);
        for(int i=0; i<batch_size;i++){
            ON_SGD(network, inputs->dataset[indices[i]] , labels->dataset[indices[i]]);
            free(indices);
        }

    }

}


void free_net(Dense_network* network){
    for(int i=0;i<network->num_layers;i++){
        free_tensor(&(network->layers[i]->weights));
        free_tensor(&(network->layers[i]->biases));
        // Do not free activation_function if not heap-allocated!
        free(network->layers[i]);
    }
    free(network->layers);
    free(network);
}