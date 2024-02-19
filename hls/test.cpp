#include <stdio.h>
#include <stdlib.h>

void matmul(
    int x_height,
    int x_width,
    int y_height, 
    int y_width,
    double *x,
    double *y,
    double *res
) {

    // naive implementation
    double mult = 0;
    for (int i = 0; i < x_height; i++) {
        for (int j = 0; j < y_width; j++) {
            res[j + i*y_width] = 0;

            for (int k = 0; k < y_height; k++) {
                mult = x[k + i*y_height] * y[j + k*y_width];
                res[j + i*y_width] += mult;
            }
        }
    }
}

int 
main(
    int argc,
    char *argv[]
) {

    int x_height = 2;
    int x_width = 4;
    int y_height = 4;
    int y_width = 2;

    double x[x_height * x_width] = {0, 1, 2, 3, 4, 5, 6, 7};
    double y[y_height * y_width] = {0, 1, 2, 3, 4, 5, 6, 7};
    double res[x_height * y_width] = {0, 0, 0, 0};
    
    matmul(x_height, x_width, y_height, y_width, x, y, res);

    for (int i = 0; i < x_height * y_width; ++i) {
        printf("%f\n", res[i]);
    }

};