#ifndef LAYERS_H
#define LAYERS_H

#include "Mat.h"
#include "Node.h"
#include "Sample.h"

#include <memory>
#include <cmath>
#include <vector>

extern "C" {
#include <cblas.h>
}

class Layer;

using namespace std;
using LayerPtr = std::shared_ptr<Layer>;

float standard_sample() {
    float x;
    do {
        x = (float)rand() / RAND_MAX;
    } while (x == 0.0);
    float y = (float)rand() / RAND_MAX;
    float z = sqrt(-2 * log(x)) * cos(2 * M_PI * y);
    return z;
}

float normal_sample(float mean, float std) {
    return mean + std * standard_sample();
}

float normal_sample_applier(float val, void* args) {
    float* params = static_cast<float*>(args);
    float mean = params[0];
    float std = params[1];
    return mean + std * standard_sample();
}

class Layer {
   public:
    virtual void initialize() {}

    virtual void construct_forward(LayerPtr prev_layer, bool requires_grad) = 0;

    virtual NodePtr get_output() {
        return output;
    }

    virtual void update(float lr, float mini_batch_size) = 0;

    void set_enabled(bool enabled) {
        output->set_enabled(enabled);
    }

    virtual void print() {
        cout << "Generic layer";
    }

    void set_is_inference(bool is_inference) {
        this->is_inference = is_inference;
    }

    bool get_samples_along_cols() {
        return samples_along_cols;
    }

    void freeze_params() {
        params_freezed = true;
    }

    void unfreeze_params() {
        params_freezed = false;
    }

   protected:
    NodePtr output;
    bool is_inference;
    bool samples_along_cols;
    bool params_freezed = false;
};

class InputLayer : public Layer {
    public:

    InputLayer(int size, int mini_batch_size, bool separated = false, bool requires_grad = false) {
        inputs = make_shared<Node>(mini_batch_size, size, requires_grad);
        output = inputs;
        samples_along_cols = false;
        this->separated = separated;
        this->size = size;
    }

    void construct_forward(LayerPtr prev_layer, bool requires_grad) override {}

    void load_train_samples(Sample* samples[], int mini_batch_size) {

        this->mini_batch_size = mini_batch_size;

        if (mini_batch_size != output->getData().getRows()) {
            throw std::invalid_argument("Error: Loss::load_train_samples - mini_batch_size != output->getData().getRows()");
            return;
        }

        // float* data = output->getData().getData();
        float* data = inputs->getData().getData();

        if (!separated) {
            for (int i = 0, j = 0; j < mini_batch_size; i += size, j++)
                std::memcpy(data + i, samples[j]->getData(), size * sizeof(float));
        } else {
            separated_inputs.clear();
            for (int i = 0; i < mini_batch_size; i++) {
                shared_ptr<Mat> sample_mat = make_shared<Mat>(1, size, false);
                sample_mat->view(samples[i]->getData());
                separated_inputs.push_back(sample_mat);
            }
        }
    }

    void load_raw(float* data, int data_size) {
        // float* data = output->getData().getData();
        float* data_ = inputs->getData().getData();
        std::memcpy(data_, data, data_size * sizeof(float));
    }

    void update(float lr, float mini_batch_size) override {}

    void print() {
        cout << "Input layer\n";
    }

    vector<shared_ptr<Mat>>& get_separated_inputs() {
        return separated_inputs;
    }

    int get_mini_batch_size() {
        return mini_batch_size;
    }

    bool is_separated() {
        return separated;
    }

    void process_inputs_separated(bool separated) {
        this->separated = separated;
    }

    private:

    NodePtr inputs;
    vector<shared_ptr<Mat>> separated_inputs;
    int size;
    int mini_batch_size;
    bool separated;
};

class Linear : public Layer {
   public:
    Linear(int input_size, int output_size) {
        weights = make_shared<Node>(output_size, input_size, false);
        bias = make_shared<Node>(output_size, 1, true);
    }

    void construct_forward(LayerPtr prev_layer, bool requires_grad) override {

        NodePtr inputs = prev_layer->get_output();

        if (prev_layer->get_samples_along_cols() == false) {
            inputs = Node::transpose(inputs, requires_grad);
        }

        Wx = Node::matmul(weights, inputs, requires_grad);
        output = Node::mat_plus_vec(Wx, bias, requires_grad); // optimize this (probably the compute method of the node)?
        x = inputs;
        // ones = std::make_unique<Mat>(inputs->getData().getCols(), 1);
        // ones->fill(1);

        samples_along_cols = true;
    }

    void update(float lr, float mini_batch_size) override {

        if (params_freezed) return;

        // WEIGHTS UPDATE (W = W - lr/mini_batch_size * dL/dz * x^T)
        x->getData().transpose();
        Mat::matmul_mm(weights->getData(), Wx->getGrad(), x->getData(), -lr/mini_batch_size, 1);
        x->getData().transpose();

        // BIAS UPDATE

        // This bias update optimization seems to not be worth it
        // Mat::matmul_mv(bias->getData(), output->getGrad(), *ones, -lr/mini_batch_size, 1);

        // Could be optimized adding a Mat method to subtract and scale at the same time, but probably not worth it
        // (bias is just a vector that probably isn't huge)
        bias->getData() -= Mat::scale(bias->getGrad(), lr / mini_batch_size);
    }

    void print() {
        cout << "Linear layer - Weights:\n";
        weights->getData().print();
        cout << "Linear layer - Bias:\n";
        bias->getData().print();
    }

    void initialize() {
        if (params_freezed) return;
        initialize_xavier();
    }

    void initialize_xavier() {
        float mean = 0;

        // sqrt(2 / n_in)
        float std = sqrt(1.0 / weights->getData().getCols());

        float args[2] = {mean, std};
        Mat::apply(weights->getData(), weights->getData(), normal_sample_applier, static_cast<void*>(args));
        bias->getData().fill(0);
    }

    NodePtr get_weights() { return weights; }
    NodePtr get_bias() { return bias; }

   private:
    NodePtr weights;
    NodePtr bias;

    // std::unique_ptr<Mat> ones;

    NodePtr x; // inputs to the layer
    NodePtr Wx; // weights * inputs
    
    // NodePtr output; would be z = Wx + b (output is already defined in Layer)
};

class Conv2d_im2row : public Layer {
    public:

    Conv2d_im2row(int n, int h, int w, int c_i, int c_o, int k_h, int k_w, int s_h, int s_w, int p_h, int p_w)
    : n(n), h(h), w(w), c_i(c_i), c_o(c_o), k_h(k_h), k_w(k_w), s_h(s_h), s_w(s_w), p_h(p_h), p_w(p_w) {
        out_h = (h + 2 * p_h - k_h) / s_h + 1;
        out_w = (w + 2 * p_w - k_w) / s_w + 1;

        kernels = make_shared<Node>(c_i * k_h * k_w, c_o, false);
        bias = make_shared<Node>(1, out_h * out_w * c_o, true);

        samples_along_cols = false;
    }

    Conv2d_im2row(int n, int h, int w, int c_i, int c_o, int kernel_size, int stride, int padding)
    : Conv2d_im2row(n, h, w, c_i, c_o, kernel_size, kernel_size, stride, stride, padding, padding) {}

    void construct_forward(LayerPtr prev_layer, bool requires_grad) override {

        if (prev_layer->get_samples_along_cols() == true) {
            throw std::invalid_argument("Error: inputs must be along rows, not columns");
            return;
        }

        NodePtr inputs = prev_layer->get_output();

        n = inputs->getData().getRows();

        inputs_reshaped = Node::reshape(inputs, n * h, w * c_i, requires_grad);
        // inputs->getData().reshape(n * h, w * c_i);

        im2row_lowered = Node::im2row(inputs_reshaped, n, h, w, c_i, k_h, k_w, s_h, s_w, p_h, p_w);
        conv = Node::matmul(im2row_lowered, kernels, requires_grad);

        // conv_reshaped = Node::reshape(conv, bias_rows, conv->getData().getRows() / bias_rows);
        conv_reshaped = Node::reshape(conv, n, -1, requires_grad);

        mat_plus_row_vec = Node::mat_plus_row_vec(conv_reshaped, bias, requires_grad);
        output = mat_plus_row_vec;
        // output = mat_plus_row_vec;
    }

    void update (float lr, float mini_batch_size) override {
        if (params_freezed) return;
        im2row_lowered->getData().transpose();
        Mat::matmul_mm(kernels->getData(), im2row_lowered->getData(), conv->getGrad(), -lr/mini_batch_size, 1);
        im2row_lowered->getData().transpose();

        // BIAS UPDATE
        bias->getData() -= Mat::scale(bias->getGrad(), lr / mini_batch_size);
    }

    void initialize() {
        if (params_freezed) return;
        initialize_xavier();
    }

    void initialize_xavier() {
        float mean = 0;

        // sqrt(2 / n_in)
        // float std = sqrt(1.0 / kernels->getData().getCols());
        float std = 0.01;

        float args[2] = {mean, std};
        Mat::apply(kernels->getData(), kernels->getData(), normal_sample_applier, static_cast<void*>(args));
        // kernels->getData().fill(0);
        bias->getData().fill(0);
    }

    private:

    int n, h, w, c_i; // c_i = input channels
    int c_o; // output channels
    int k_h, k_w;
    int s_h, s_w;
    int p_h, p_w;
    int out_h, out_w;

    NodePtr inputs_reshaped;
    NodePtr im2row_lowered;
    NodePtr kernels;
    NodePtr conv;
    NodePtr conv_reshaped;
    NodePtr bias;
    NodePtr mat_plus_row_vec;
};

class Conv2d_mec : public Layer {
    public: 

    Conv2d_mec(int n, int h, int w, int c_i, int c_o, int k_h, int k_w, int s_h, int s_w, int p_h, int p_w)
    : n(n), h(h), w(w), c_i(c_i), c_o(c_o), k_h(k_h), k_w(k_w), s_h(s_h), s_w(s_w), p_h(p_h), p_w(p_w) {
        out_h = (h + 2 * p_h - k_h) / s_h + 1;
        out_w = (w + 2 * p_w - k_w) / s_w + 1;

        kernels = make_shared<Node>(c_i * k_h * k_w, c_o, false);
        bias = make_shared<Node>(1, c_o, true);

        im_padded_h = h + 2 * p_h;
        im_padded_w = w + 2 * p_w;

        samples_along_cols = false;
    }

    Conv2d_mec(int n, int h, int w, int c_i, int c_o, int kernel_size, int stride, int padding)
    : Conv2d_mec(n, h, w, c_i, c_o, kernel_size, kernel_size, stride, stride, padding, padding) {}

    void construct_forward(LayerPtr prev_layer, bool requires_grad) override {

        if (prev_layer->get_samples_along_cols() == true) {
            throw std::invalid_argument("Error: inputs must be along rows, not columns");
            return;
        }

        // bool lowered_requires_grad = true;

        // if (auto input_layer = dynamic_cast<InputLayer*>(prev_layer.get())) {
        //     // If prev_layer is an instance of InputLayer, it's now cast to InputLayer
        //     // You can use input_layer pointer here if needed
        //     lowered_requires_grad = false;
        //     if (input_layer->is_separated()) {
        //         vector<shared_ptr<Mat>> separated_inputs = input_layer->get_separated_inputs();
        //         n = separated_inputs.size();
        //         mec_lowered = Node::mec_lower_separated(separated_inputs, n, h, w, c_i, k_h, k_w, s_h, s_w);
        //     } else {

        //     }
        // }

        // NodePtr inputs = prev_layer->get_output();

        // n = inputs->getData().getRows();

        // inputs_reshaped = Node::reshape(inputs, n * h, w * c_i);

        // mec_lowered = Node::mec_lower(inputs_reshaped, n, h, w, c_i, k_h, k_w, s_h, s_w, p_h, p_w, lowered_requires_grad);

        mec_lowered = handle_inputs_to_mec_lower(prev_layer, requires_grad);

        conv_hnwc = Node::mec_conv_mm(mec_lowered, kernels, n, im_padded_h, im_padded_w, c_i, k_h, k_w, s_h, s_w, requires_grad);

        conv_reshaped_for_bias = Node::reshape(conv_hnwc, -1, c_o, requires_grad);

        conv_plus_bias = Node::mat_plus_row_vec(conv_reshaped_for_bias, bias, requires_grad);
        
        output = Node::reshape(conv_plus_bias, out_h, n * out_w * c_o, requires_grad);
    }

    NodePtr handle_inputs_to_mec_lower(LayerPtr prev_layer, bool requires_grad) {
        bool lowered_requires_grad = requires_grad;
        NodePtr inputs = prev_layer->get_output();
        n = inputs->getData().getRows();

        if (auto input_layer = dynamic_cast<InputLayer*>(prev_layer.get())) {
            lowered_requires_grad = false;
            if (input_layer->is_separated()) {
                vector<shared_ptr<Mat>>& separated_inputs = input_layer->get_separated_inputs();
                return Node::mec_lower_separated(separated_inputs, n, h, w, c_i, k_h, k_w, s_h, s_w, false);
            }
        }

        inputs_reshaped = Node::reshape(inputs, n * h, w * c_i);
        return Node::mec_lower(inputs_reshaped, n, h, w, c_i, k_h, k_w, s_h, s_w, p_h, p_w, lowered_requires_grad);
    }

    void update (float lr, float mini_batch_size) override {
        if (params_freezed) return;
        update_kernels(lr, mini_batch_size);
        bias->getData() -= Mat::scale(bias->getGrad(), lr / mini_batch_size);
    }

    void update_kernels(float lr, float mini_batch_size) {

        int next_grad_rows = out_w * n;
        int next_grad_cols = c_o;

        float* other_start = mec_lowered->getData().getData();
        float* next_grad_start = conv_hnwc->getGrad().getData();

        for (int hh = 0; hh < out_h; hh++) {
            float* cur_other_start = other_start + hh * s_h * c_i * k_w;
            float* cur_next_grad_start = next_grad_start + hh * out_w * c_o * n;

            cblas_sgemm(CblasRowMajor,
                    CblasTrans, CblasNoTrans,
                    kernels->getRows(), kernels->getCols(), next_grad_rows,
                    -lr / mini_batch_size, cur_other_start, mec_lowered->getData().getCols(), cur_next_grad_start, next_grad_cols,
                    1.0, kernels->getData().getData(), c_o);            
        }

    }

    void initialize() {
        if (params_freezed) return;
        initialize_xavier();
    }

    void initialize_xavier() {
        float mean = 0;

        // sqrt(2 / n_in)
        // float std = sqrt(1.0 / kernels->getData().getCols());
        float std = 0.01;

        float args[2] = {mean, std};
        Mat::apply(kernels->getData(), kernels->getData(), normal_sample_applier, static_cast<void*>(args));
        // kernels->getData().fill(0);
        bias->getData().fill(0);
    }

    void print() {
        cout << "Conv2d_mec layer - Outputs:\n";
        output->getData().print();
        cout << "Conv2d_mec layer - Kernels:\n";
        kernels->getData().print();
        cout << "Conv2d_mec layer - Bias:\n";
        bias->getData().print();
    }

    void load_kernels_raw(float* data, int data_size) {
        if (data_size != kernels->getData().getSize()) {
            throw std::invalid_argument("Error: Conv2d_mec::load_kernels_raw - data_size != kernels->getData().getSize()");
            return;
        }
        std::memcpy(kernels->getData().getData(), data, data_size * sizeof(float));
    }

    void load_biases_raw(float* data, int data_size) {
        if (data_size != bias->getData().getSize()) {
            throw std::invalid_argument("Error: Conv2d_mec::load_kernels_raw - data_size != kernels->getData().getSize()");
            return;
        }
        std::memcpy(bias->getData().getData(), data, data_size * sizeof(float));
    }

    NodePtr get_mec_lowered() {
        return mec_lowered;
    }

    NodePtr get_conv_hnwc() {
        return conv_hnwc;
    }

    private:

    int n, h, w, c_i; // c_i = input channels
    int c_o; // output channels
    int k_h, k_w;
    int s_h, s_w;
    int p_h, p_w;
    int im_padded_h, im_padded_w;
    int out_h, out_w;

    NodePtr inputs_reshaped;
    NodePtr mec_lowered;
    NodePtr kernels;
    NodePtr conv_hnwc;
    NodePtr conv_reshaped_for_bias;
    NodePtr bias;
    NodePtr conv_plus_bias;
};

class Maxpool_hnwc_to_nhwc : public Layer {
    public:

    Maxpool_hnwc_to_nhwc(int n, int h, int w, int c, int k_h, int k_w, int s_h, int s_w) :
    n(n), h(h), w(w), c(c), k_h(k_h), k_w(k_w), s_h(s_h), s_w(s_w) {
        out_h = (h - k_h) / s_h + 1;
        out_w = (w - k_w) / s_w + 1;

        samples_along_cols = false;
    }

    void construct_forward(LayerPtr prev_layer, bool requires_grad) override {

        if (prev_layer->get_samples_along_cols() == true) {
            throw std::invalid_argument("Error: inputs must be along rows, not columns");
            return;
        }

        NodePtr inputs = prev_layer->get_output();

        n = inputs->getData().getCols() / (w * c);

        maxpool = Node::maxpool_hnwc_to_nhwc(inputs, n, h, w, c, k_h, k_w, s_h, s_w, requires_grad);

        output = Node::reshape(maxpool, n, -1, requires_grad);
    }

    void update(float lr, float mini_batch_size) override {}

    void print() {
        cout << "Maxpool_hnwc_to_nhwc layer - Maxpool:\n";
        maxpool->getData().print();
        cout << "Maxpool_hnwc_to_nhwc layer - Outputs:\n";
        output->getData().print();
    }

    private:

    int n, h, w, c;
    int k_h, k_w;
    int s_h, s_w;
    int out_h, out_w;

    NodePtr maxpool;
};

// Inverse dropout (neuron scaling is applied during training, not during inference)
class Dropout : public Layer {
    public:
    
    Dropout(float p) : p(p) {}

    void construct_forward(LayerPtr prev_layer, bool requires_grad) override {

        NodePtr inputs = prev_layer->get_output();
        samples_along_cols = prev_layer->get_samples_along_cols();

        if (is_inference) {
            output = inputs;
            return;
        }

        NodePtr dropout_node = Node::dropout(inputs->getData().getRows(), inputs->getData().getCols(), p, false);
        output = Node::hadamard_product(inputs, dropout_node, requires_grad);

    }

    void update(float lr, float mini_batch_size) override {}

    private:

    float p;
     
};

class Sigmoid : public Layer {
   public:

    void construct_forward(LayerPtr prev_layer, bool requires_grad) override {
        NodePtr inputs = prev_layer->get_output();
        output = Node::sigmoid(inputs, requires_grad);
        samples_along_cols = prev_layer->get_samples_along_cols();
    }

    void update(float lr, float mini_batch_size) override {}

    void print() {
        cout << "Sigmoid layer\n";
    }
};

class Tanh : public Layer {
   public:

    void construct_forward(LayerPtr prev_layer, bool requires_grad) override {
        NodePtr inputs = prev_layer->get_output();
        output = Node::tanh(inputs, requires_grad);
        samples_along_cols = prev_layer->get_samples_along_cols();
    }

    void update(float lr, float mini_batch_size) override {}

    void print() {
        cout << "Tanh layer\n";
    }
};

class ReLU : public Layer {
   public:
   
    void construct_forward(LayerPtr prev_layer, bool requires_grad) override {
        NodePtr inputs = prev_layer->get_output();
        output = Node::relu(inputs, requires_grad);
        samples_along_cols = prev_layer->get_samples_along_cols();
    }

    void update(float lr, float mini_batch_size) override {}

    void print() {
        cout << "ReLU layer\n";
    }
};

class Summation : public Layer {
   public:
   
    void construct_forward(LayerPtr prev_layer, bool requires_grad) override {
        NodePtr inputs = prev_layer->get_output();
        output = inputs->sum(requires_grad);
        samples_along_cols = prev_layer->get_samples_along_cols();
    }

    void update(float lr, float mini_batch_size) override {}

    void print() {
        cout << "Summation layer\n";
    }
};

#endif