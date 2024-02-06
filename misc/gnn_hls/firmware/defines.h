#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 64
#define N_INPUT_2_1 5
#define N_INPUT_1_2 64
#define N_INPUT_2_2 64
#define N_INPUT_1_3 1
#define N_INPUT_2_3 64
#define N_OUTPUTS_20 64
#define N_FILT_20 32
#define N_INPUT_1_2 64
#define N_INPUT_2_1 5
#define N_OUTPUTS_21 64
#define N_FILT_21 32
#define N_LAYER_1_4 64
#define N_LAYER_2_4 32
#define N_LAYER_1_4 64
#define N_LAYER_2_4 32
#define N_OUTPUTS_22 64
#define N_FILT_22 64
#define N_INPUT_1_2 64
#define N_LAYER_2_4 32
#define N_OUTPUTS_23 64
#define N_FILT_23 64
#define N_LAYER_1_9 64
#define N_LAYER_2_9 64
#define N_LAYER_1_9 64
#define N_LAYER_2_9 64
#define N_INPUT_1_3 1
#define N_LAYER_2_9 64
#define N_LAYER_1_15 1
#define N_LAYER_2_15 64
#define N_LAYER_1_15 1
#define N_LAYER_2_15 64
#define N_LAYER_1_17 1
#define N_LAYER_2_17 32
#define N_LAYER_1_17 1
#define N_LAYER_2_17 32
#define N_LAYER_1_19 1
#define N_LAYER_2_19 1

// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<512,256> input_t;
typedef ap_fixed<512,256> input2_t;
typedef ap_fixed<512,256> input3_t;
typedef ap_fixed<512,256> model_default_t;
typedef ap_fixed<512,256> layer20_t;
typedef ap_uint<1> bias20_t;
typedef ap_fixed<512,256> layer5_t;
typedef ap_fixed<512,256> layer21_t;
typedef ap_fixed<512,256> layer7_t;
typedef ap_fixed<512,256> layer8_t;
typedef ap_fixed<18,8> activation_table_t;
typedef ap_fixed<512,256> layer22_t;
typedef ap_uint<1> bias22_t;
typedef ap_fixed<512,256> layer10_t;
typedef ap_fixed<512,256> layer23_t;
typedef ap_fixed<512,256> layer12_t;
typedef ap_fixed<512,256> layer13_t;
typedef ap_fixed<18,8> activation_1_table_t;
typedef ap_fixed<512,256> layer14_t;
typedef ap_fixed<512,256> layer15_t;
typedef ap_uint<1> layer15_index;
typedef ap_fixed<512,256> layer16_t;
typedef ap_fixed<18,8> activation_2_table_t;
typedef ap_fixed<512,256> layer17_t;
typedef ap_uint<1> layer17_index;
typedef ap_fixed<512,256> layer18_t;
typedef ap_fixed<18,8> activation_3_table_t;
typedef ap_fixed<512,256> result_t;
typedef ap_uint<1> layer19_index;

#endif
