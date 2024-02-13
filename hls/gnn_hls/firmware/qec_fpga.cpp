#include <iostream>

#include "qec_fpga.h"
#include "parameters.h"

void qec_fpga(
    input_t x[N_INPUT_1_1*N_INPUT_2_1], input2_t adj[N_INPUT_1_2*N_INPUT_2_2], input3_t batch[N_INPUT_1_3*N_INPUT_2_3],
    result_t layer19_out[N_LAYER_1_19*N_LAYER_2_19]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=x complete dim=0
    #pragma HLS ARRAY_RESHAPE variable=adj complete dim=0
    #pragma HLS ARRAY_RESHAPE variable=batch complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer19_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=x,adj,batch,layer19_out 
    #pragma HLS DATAFLOW 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<model_default_t, 160>(w20, "w20.txt");
        nnet::load_weights_from_txt<bias20_t, 32>(b20, "b20.txt");
        nnet::load_weights_from_txt<model_default_t, 160>(w21, "w21.txt");
        nnet::load_weights_from_txt<model_default_t, 32>(b21, "b21.txt");
        nnet::load_weights_from_txt<model_default_t, 2048>(w22, "w22.txt");
        nnet::load_weights_from_txt<bias22_t, 64>(b22, "b22.txt");
        nnet::load_weights_from_txt<model_default_t, 2048>(w23, "w23.txt");
        nnet::load_weights_from_txt<model_default_t, 64>(b23, "b23.txt");
        nnet::load_weights_from_txt<model_default_t, 4096>(w15, "w15.txt");
        nnet::load_weights_from_txt<model_default_t, 64>(b15, "b15.txt");
        nnet::load_weights_from_txt<model_default_t, 2048>(w17, "w17.txt");
        nnet::load_weights_from_txt<model_default_t, 32>(b17, "b17.txt");
        nnet::load_weights_from_txt<model_default_t, 32>(w19, "w19.txt");
        nnet::load_weights_from_txt<model_default_t, 1>(b19, "b19.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    layer20_t layer20_out[N_OUTPUTS_20*N_FILT_20];
    #pragma HLS ARRAY_PARTITION variable=layer20_out complete dim=0
    nnet::pointwise_conv_1d_cl<input_t, layer20_t, config20>(x, layer20_out, w20, b20); // graph_layers_0_lin_root

    layer5_t layer5_out[N_INPUT_1_2*N_INPUT_2_1];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
    nnet::matmul<input2_t, config5>(adj, x, layer5_out); // pmat_mul

    layer21_t layer21_out[N_OUTPUTS_21*N_FILT_21];
    #pragma HLS ARRAY_PARTITION variable=layer21_out complete dim=0
    nnet::pointwise_conv_1d_cl<layer5_t, layer21_t, config21>(layer5_out, layer21_out, w21, b21); // graph_layers_0_lin_rel

    layer7_t layer7_out[N_LAYER_1_4*N_LAYER_2_4];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0
    nnet::add<layer20_t, layer21_t, layer7_t, config7>(layer20_out, layer21_out, layer7_out); // add

    layer8_t layer8_out[N_LAYER_1_4*N_LAYER_2_4];
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0
    nnet::relu<layer7_t, layer8_t, ReLU_config8>(layer7_out, layer8_out); // activation

    layer22_t layer22_out[N_OUTPUTS_22*N_FILT_22];
    #pragma HLS ARRAY_PARTITION variable=layer22_out complete dim=0
    nnet::pointwise_conv_1d_cl<layer8_t, layer22_t, config22>(layer8_out, layer22_out, w22, b22); // graph_layers_1_lin_root

    layer10_t layer10_out[N_INPUT_1_2*N_LAYER_2_4];
    #pragma HLS ARRAY_PARTITION variable=layer10_out complete dim=0
    nnet::matmul<input2_t, config10>(adj, layer8_out, layer10_out); // pmat_mul_1

    layer23_t layer23_out[N_OUTPUTS_23*N_FILT_23];
    #pragma HLS ARRAY_PARTITION variable=layer23_out complete dim=0
    nnet::pointwise_conv_1d_cl<layer10_t, layer23_t, config23>(layer10_out, layer23_out, w23, b23); // graph_layers_1_lin_rel

    layer12_t layer12_out[N_LAYER_1_9*N_LAYER_2_9];
    #pragma HLS ARRAY_PARTITION variable=layer12_out complete dim=0
    nnet::add<layer22_t, layer23_t, layer12_t, config12>(layer22_out, layer23_out, layer12_out); // add_1

    layer13_t layer13_out[N_LAYER_1_9*N_LAYER_2_9];
    #pragma HLS ARRAY_PARTITION variable=layer13_out complete dim=0
    nnet::relu<layer12_t, layer13_t, ReLU_config13>(layer12_out, layer13_out); // activation_1

    layer14_t layer14_out[N_INPUT_1_3*N_LAYER_2_9];
    #pragma HLS ARRAY_PARTITION variable=layer14_out complete dim=0
    nnet::matmul<input3_t, config14>(batch, layer13_out, layer14_out); // pmat_mul_2

    layer15_t layer15_out[N_LAYER_1_15*N_LAYER_2_15];
    #pragma HLS ARRAY_PARTITION variable=layer15_out complete dim=0
    nnet::dense<layer14_t, layer15_t, config15>(layer14_out, layer15_out, w15, b15); // dense_layers_0

    layer16_t layer16_out[N_LAYER_1_15*N_LAYER_2_15];
    #pragma HLS ARRAY_PARTITION variable=layer16_out complete dim=0
    nnet::relu<layer15_t, layer16_t, ReLU_config16>(layer15_out, layer16_out); // activation_2

    layer17_t layer17_out[N_LAYER_1_17*N_LAYER_2_17];
    #pragma HLS ARRAY_PARTITION variable=layer17_out complete dim=0
    nnet::dense<layer16_t, layer17_t, config17>(layer16_out, layer17_out, w17, b17); // dense_layers_1

    layer18_t layer18_out[N_LAYER_1_17*N_LAYER_2_17];
    #pragma HLS ARRAY_PARTITION variable=layer18_out complete dim=0
    nnet::relu<layer17_t, layer18_t, ReLU_config18>(layer17_out, layer18_out); // activation_3

    nnet::dense<layer18_t, result_t, config19>(layer18_out, layer19_out, w19, b19); // output_layer

}
