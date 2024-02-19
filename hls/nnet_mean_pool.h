#ifndef NNET_MATMUL_H_
#define NNET_MATMUL_H_

#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_helpers.h"
#include "nnet_mult.h"
#include <math.h>

namespace nnet {

    struct mean_pool_config {
        static const unsigned in_height = 10;
        static const unsigned in_width = 5;

        // product function to use
        template <class x_T, class y_T> using product = nnet::product::mult<x_T, y_T>;
    };

    template<class data_T, typename CONFIG_T>
    void mean_pool(
        data_T in_data[CONFIG_T::in_height * CONFIG_T::in_width],
        data_T out_data[CONFIG_T::in_width]
    ) {

        // initialise out_data
        for (int ix = 0; ix < CONFIG_T::in_width; ix++) {
            out_data[ix] = 0;
        }

        // aggregate over input matrix
        int i = 0;
        while (
            in_data[in_width - 1 + i*CONFIG_T::in_width] != 0 || 
            in_data[in_width - 2 + i*CONFIG_T::in_width] != 0
            ) {
                for (int jx = 0; jx < CONFIG_T::in_width; jx++) {
                    out_data[jx] += in_data[jx + i*CONFIG_T::in_width];
                }
                i++;
            }
        
        // divide result to get mean
        for (int ix = 0; ix < CONFIG_T::in_width; ix++) {
            out_data[ix] /= i;
        }

    }
}


#endif

