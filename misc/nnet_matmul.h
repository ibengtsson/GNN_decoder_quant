#ifndef NNET_MATMUL_H_
#define NNET_MATMUL_H_

#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_helpers.h"
#include "nnet_mult.h"
#include <math.h>

namespace nnet {

struct matmul_config {
    static const unsigned x_height = 10;
    static const unsigned x_width = 10;
    static const unsigned y_height = 10;
    static const unsigned y_width = 10;

    // product function to use
    template <class x_T, class y_T> using product = nnet::product::mult<x_T, y_T>;
};

template<class data_T, typename CONFIG_T>
void matmul(
    data_T x[CONFIG_T::x_height * CONFIG_T::x_width],
    data_T y[CONFIG_T::y_height * CONFIG_T::y_width],
    data_T res[CONFIG_T::x_height * CONFIG_T::y_width]
) {

    // naive implementation
    data_T mult = 0;
    for (int i = 0; i < CONFIG_T::x_height; i++) {
        for (int j = 0; j < CONFIG_T::y_width; j++) {
            res[j + i*CONFIG_T::y_width] = 0;

            for (int k = 0; k < CONFIG_T::y_height; k++) {
                mult = CONFIG_T::template product <data_T, data_T>::product(x[k + i*CONFIG_T::y_height], y[j + k*CONFIG_T::y_width]);
                res[j + i*CONFIG_T::y_width] += mult;
            }
        }
    }
}




}
#endif