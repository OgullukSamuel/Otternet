#include "../header/ottermath.h"

float OM_ldexp(float x, int n) {
    if (x == 0.0f) return 0.0f;

    union { float f; uint32_t u; } u = { x };
    int exp = ((u.u >> 23) & 0xFF) + n;

    if (exp <= 0) return 0.0f; // sous-flux simple
    if (exp >= 0xFF) return x > 0 ? INFINITY : -INFINITY;

    u.u = (u.u & ~(0xFFU << 23)) | ((uint32_t)exp << 23);
    return u.f;
}

float OM_exp(float x) {
    // Calculer n tel que x = n*LN2 + dev avec dev dans [-LN2/2, LN2/2]
    int n = (int)(x / LN2 + (x >= 0 ? 0.5f : -0.5f));
    float dev = x - n * LN2;

    float exp_r = 1.0f;   // somme initiale (i=0)
    float power = 1.0f;   // dev^i
    float factorial = 1.0f;

    for (int i = 1; i <= 9; i++) {
        power *= dev;         // dev^i
        factorial *= i;       // i!
        exp_r += power / factorial;
    }

    return OM_ldexp(exp_r, n);
}


float OM_log2(float x) {
    // Gestion des cas invalides
    if (x <= 0.0f) return -INFINITY;

    // Représentation binaire du float
    uint32_t bits;
    memcpy(&bits, &x, sizeof(bits));

    // Extraction de l'exposant : bits 23 à 30
    int exponent = ((bits >> 23) & 0xFF) - 127;

    // Mise à zéro de l'exposant pour obtenir une mantisse entre [1.0, 2.0)
    bits = (bits & 0x7FFFFF) | (127 << 23);  // force l'exposant à 127

    float mantissa;
    memcpy(&mantissa, &bits, sizeof(mantissa));

    // ln(x) = ln(mantissa) + exponent * ln(2)
    // => approximer ln(mantissa) où mantissa ∈ [1, 2)
    float y = mantissa - 1.0f;       // y ∈ [0, 1)
    float y2 = y * y;
    float y3 = y2 * y;
    float y4 = y2 * y2;

    // Approximation de ln(1 + y) par un polynôme d'ordre 5 (minimax)
    float ln_mantissa = y * (
        1.0f
        - 0.5f * y
        + 0.3333333f * y2
        - 0.25f * y3
        + 0.2f * y4
    );

    // Résultat final
    return ln_mantissa/LN2 + exponent;
}

float OM_log10(float x){
    return(OM_log2(x)*LOG10);
}

float OM_ln(float x){
    return OM_log2(x)*LN2;
}



void OM_tensor_linear(OtterTensor* input) {
    return;
}

void OM_tensor_zeros(OtterTensor* input) {
    for (int i = 0; i < input->size; i++) {
        input->data[i] = 0.0f;
    }
    return;
}

void OM_tensor_ones(OtterTensor* input) {
    for (int i = 0; i < input->size; i++) {
        input->data[i] = 1.0f;
    }
    return;
}

float OM_heaviside(float x){
    return x > 0.0f;
}

void OM_tensor_heaviside(OtterTensor* input) {
    for (int i = 0; i < input->size; i++) {
        input->data[i] = OM_heaviside(input->data[i]);
    }
    return;
}


// tanh

float OM_tanh(float x) {
    float exp2x = OM_exp(2 * x);
    return (exp2x - 1) / (exp2x + 1);
}

float OM_dtanh(float x) {
    float exp2x = OM_exp(2 * x);
    float tanh = (exp2x - 1) / (exp2x + 1);
    return 1-tanh*tanh;
}

void OM_tensor_tanh(OtterTensor* input) {
    for (int i = 0; i < input->size; i++) {
        input->data[i] = OM_tanh(input->data[i]);
    }
    return;
}

void OM_tensor_dtanh(OtterTensor* input) {
    for (int i = 0; i < input->size; i++) {
        input->data[i] = OM_dtanh(input->data[i]);
    }
    return;
}

// sigmoid

float OM_sigmoid(float x){
    return(1.0f / (1.0f + OM_exp(-x)));
}

float OM_dsigmoid(float x){
    float sig = OM_sigmoid(x);
    return sig*(1-sig);
}


void OM_tensor_sigmoid(OtterTensor* input) {
    for (int i = 0; i < input->size; i++) {
        input->data[i] = OM_sigmoid(input->data[i]);
    }
    return;
}

void OM_tensor_dsigmoid(OtterTensor* input) {
    for (int i = 0; i < input->size; i++) {
        input->data[i] = OM_dsigmoid(input->data[i]);
    }
    return;
}

// relu

float OM_relu(float x) {
    return (x < 0) ? 0 : x;
}

void OM_tensor_relu(OtterTensor* input) {
    for (int i = 0; i < input->size; i++) {
        input->data[i] = OM_relu(input->data[i]);
    }
    return;
}

// prelu

float OM_prelu(float x, float alpha) {
    return (x < 0) ? alpha * x : x;
}

void OM_tensor_prelu(OtterTensor* input, float alpha) {
    for (int i = 0; i < input->size; i++) {
        input->data[i] = OM_prelu(input->data[i], alpha);
    }
    return;
}

// leaky relu

float OM_leaky_relu(float x, float alpha) {
    return (x < 0) ? alpha * x : x;
}

void OM_tensor_leaky_relu(OtterTensor* input, float alpha) {
    for (int i = 0; i < input->size; i++) {
        input->data[i] = OM_leaky_relu(input->data[i], alpha);
    }
    return;
}

// elu

float OM_elu(float x, float alpha) {
    return (x < 0) ? alpha * (OM_exp(x) - 1) : x;
}

void OM_tensor_elu(OtterTensor* input, float alpha) {
    for (int i = 0; i < input->size; i++) {
        input->data[i] = OM_elu(input->data[i], alpha);
    }
    return;
}

//

OtterTensor* OM_softmax(OtterTensor* input) {
    OtterTensor* values = OT_zeros(input->dims, input->rank);    
    float sum = 0.0f;
    for(int i = 0; i < values->size; i++) {
        values->data[i] = OM_exp(input->data[i]);
        sum += values->data[i];
    }
    for(int i = 0; i < values->size; i++) {
        values->data[i] /= sum;
    }
    return values;
}

void OM_ref_softmax(OtterTensor* input) {
    float sum = 0.0f;
    for(int i = 0; i < input->size; i++) {
        input->data[i] = OM_exp(input->data[i]);
        sum += input->data[i];
    }
    for(int i = 0; i < input->size; i++) {
        input->data[i] /= sum;
    }
    return;
}

OtterTensor* OM_softmax_with_temperature(OtterTensor* input, float temperature){
    OtterTensor* values = OT_zeros(input->dims, input->rank);    
    float sum = 0.0f;
    for(int i = 0; i < values->size; i++) {
        values->data[i] = OM_exp(input->data[i]/temperature);
        sum += values->data[i];
    }
    for(int i = 0; i < values->size; i++) {
        values->data[i] /= sum;
    }
    return values;
}


float OM_cross_entropy(OtterTensor* predictions, OtterTensor* truth) {
    float sum=0.0f;
    if(predictions->size != truth->size) {
        fprintf(stderr, "Error: Predictions and truth tensors must have the same size.\n");
        exit(EXIT_FAILURE);
    }
    for(int i=0; i<predictions->size;i++){
        sum+=truth->data[i]*OM_ln(predictions->data[i])+(1-truth->data[i])*OM_ln(1-predictions->data[i]);
    }
    return -sum/truth->size;
}



OtterTensor* Vectorize(float x, float (*func)(float)){
    OtterTensor* result = OT_zeros((int[]){1, 1}, 2);
    for(int i = 0; i < result->size; i++) {
        result->data[i] = func(x);
    }
    return result;
}

void OM_ref_Vectorize(OtterTensor* x, float (*func)(float)){
    for(int i = 0; i < x->size; i++) {
        x->data[i] = func(x->data[i]);
    }
    return;
}

float OM_int_power(float base, int exponent) { // à travailler
    float result = 1.0f;
    for (int i = 0; i < exponent; i++) {
        result *= base;
    }
    return result;
}

float OM_sqrt(float x) {
    if (x < 0.0f) {
        fprintf(stderr, "Error: Cannot compute square root of a negative number.\n");
        exit(EXIT_FAILURE);
    }
    float result = 1.0f;
    for (int i = 0; i < 10; i++) { // Newton's method
        result = 0.5f * (result + x / result);
    }
    return result;
}

void OM_ref_sqrt(OtterTensor* t) {
    for (int i = 0; i < t->size; i++) {
        t->data[i] = OM_sqrt(t->data[i]);
    }
    return;
}