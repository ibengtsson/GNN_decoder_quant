#ifndef QEC_FPGA_BRIDGE_H_
#define QEC_FPGA_BRIDGE_H_

#include "firmware/qec_fpga.h"
#include "firmware/nnet_utils/nnet_helpers.h"
#include <algorithm>
#include <map>

// hls-fpga-machine-learning insert bram

namespace nnet {
bool trace_enabled = false;
std::map<std::string, void *> *trace_outputs = NULL;
size_t trace_type_size = sizeof(double);
} // namespace nnet

extern "C" {

struct trace_data {
    const char *name;
    void *data;
};

void allocate_trace_storage(size_t element_size) {
    nnet::trace_enabled = true;
    nnet::trace_outputs = new std::map<std::string, void *>;
    nnet::trace_type_size = element_size;
}

void free_trace_storage() {
    for (std::map<std::string, void *>::iterator i = nnet::trace_outputs->begin(); i != nnet::trace_outputs->end(); i++) {
        void *ptr = i->second;
        free(ptr);
    }
    nnet::trace_outputs->clear();
    delete nnet::trace_outputs;
    nnet::trace_outputs = NULL;
    nnet::trace_enabled = false;
}

void collect_trace_output(struct trace_data *c_trace_outputs) {
    int ii = 0;
    for (std::map<std::string, void *>::iterator i = nnet::trace_outputs->begin(); i != nnet::trace_outputs->end(); i++) {
        c_trace_outputs[ii].name = i->first.c_str();
        c_trace_outputs[ii].data = i->second;
        ii++;
    }
}

// Wrapper of top level function for Python bridge
void qec_fpga_float(
    float x[N_INPUT_1_1*N_INPUT_2_1], float adj[N_INPUT_1_2*N_INPUT_2_2], float batch[N_INPUT_1_3*N_INPUT_2_3],
    float layer19_out[N_LAYER_1_19*N_LAYER_2_19]
) {

    input_t x_ap[N_INPUT_1_1*N_INPUT_2_1];
    nnet::convert_data<float, input_t, N_INPUT_1_1*N_INPUT_2_1>(x, x_ap);
    input2_t adj_ap[N_INPUT_1_2*N_INPUT_2_2];
    nnet::convert_data<float, input2_t, N_INPUT_1_2*N_INPUT_2_2>(adj, adj_ap);
    input3_t batch_ap[N_INPUT_1_3*N_INPUT_2_3];
    nnet::convert_data<float, input3_t, N_INPUT_1_3*N_INPUT_2_3>(batch, batch_ap);

    result_t layer19_out_ap[N_LAYER_1_19*N_LAYER_2_19];

    qec_fpga(x_ap,adj_ap,batch_ap,layer19_out_ap);

    nnet::convert_data<result_t, float, N_LAYER_1_19*N_LAYER_2_19>(layer19_out_ap, layer19_out);
}

void qec_fpga_double(
    double x[N_INPUT_1_1*N_INPUT_2_1], double adj[N_INPUT_1_2*N_INPUT_2_2], double batch[N_INPUT_1_3*N_INPUT_2_3],
    double layer19_out[N_LAYER_1_19*N_LAYER_2_19]
) {
    input_t x_ap[N_INPUT_1_1*N_INPUT_2_1];
    nnet::convert_data<double, input_t, N_INPUT_1_1*N_INPUT_2_1>(x, x_ap);
    input2_t adj_ap[N_INPUT_1_2*N_INPUT_2_2];
    nnet::convert_data<double, input2_t, N_INPUT_1_2*N_INPUT_2_2>(adj, adj_ap);
    input3_t batch_ap[N_INPUT_1_3*N_INPUT_2_3];
    nnet::convert_data<double, input3_t, N_INPUT_1_3*N_INPUT_2_3>(batch, batch_ap);

    result_t layer19_out_ap[N_LAYER_1_19*N_LAYER_2_19];

    qec_fpga(x_ap,adj_ap,batch_ap,layer19_out_ap);

    nnet::convert_data<result_t, double, N_LAYER_1_19*N_LAYER_2_19>(layer19_out_ap, layer19_out);
}
}

#endif
