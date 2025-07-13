#include "../header/main.h"

int main() {
    // Création d'un tenseur 1D de taille 8 rempli de 1
    int dims[1] = {8};
    OtterTensor* t = OT_ones(dims, 1);

    // Paramètres pour le slicing
    int filter_size = 3;
    int stride = 2;
    int padding = 1;

    // Appel de la fonction de slicing
    OtterTensor** slices = OT_slice_padding(t, filter_size, stride, padding);

    // Calcul du nombre de slices
    int number_slides = t->dims[0] / stride;

    // Affichage des slices
    printf("Slices:\n");
    for (int i = 0; i < number_slides; i++) {
        print_tensor(slices[i], 3);
        free_malloc_tensor(slices[i]);
    }

    free(slices);
    free_malloc_tensor(t);

    return 0;
}