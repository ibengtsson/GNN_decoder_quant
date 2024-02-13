#ifndef QEC_FPGA_H_
#define QEC_FPGA_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"

// Prototype of top level function for C-synthesis
void qec_fpga(
    input_t x[N_INPUT_1_1*N_INPUT_2_1], input2_t adj[N_INPUT_1_2*N_INPUT_2_2], input3_t batch[N_INPUT_1_3*N_INPUT_2_3],
    result_t layer19_out[N_LAYER_1_19*N_LAYER_2_19]
);

#endif
