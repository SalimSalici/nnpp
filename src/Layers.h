#ifndef LAYERS_H
#define LAYERS_H

#include "Mat.h"
#include "Node.h"
#include "Sample.h"

#include <memory>
#include <cmath>

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

    virtual void construct_forward(NodePtr inputs) = 0;

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

   protected:
    NodePtr output;
    bool is_inference;
};

class InputLayer : public Layer {
    public:

    InputLayer(int size, int mini_batch_size) {
        // requires_grad = false for input layer -----------v
        output = make_shared<Node>(mini_batch_size, size, false);
        output->getData().transpose();
        output->getGrad().transpose();
        this->size = size;
    }

    void construct_forward(NodePtr inputs) override {}

    void load_train_samples(Sample* samples[], int mini_batch_size) {
        if (mini_batch_size != output->getData().getCols()) {
            throw std::invalid_argument("Error: Loss::load_train_samples - mini_batch_size != output->getData().getCols()");
            return;
        }

        float* data = output->getData().getData();

        for (int i = 0, j = 0; j < mini_batch_size; i += size, j++)
            std::memcpy(data + i, samples[j]->getData(), size * sizeof(float));
    }

    void update(float lr, float mini_batch_size) override {}

    void print() {
        cout << "Input layer\n";
    }

    private:

    int size;
    int mini_batch_size;
};

class Linear : public Layer {
   public:
    Linear(int input_size, int output_size) {
        weights = make_shared<Node>(output_size, input_size, false);
        bias = make_shared<Node>(output_size, 1, true);
    }

    void construct_forward(NodePtr inputs) override {
        Wx = Node::matmul(weights, inputs);
        output = Node::mat_plus_vec(Wx, bias); // optimize this (probably the compute method of the node)?
        x = inputs;
        ones = std::make_unique<Mat>(inputs->getData().getCols(), 1);
        ones->fill(1);
    }

    void update(float lr, float mini_batch_size) override {

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

    std::unique_ptr<Mat> ones;

    NodePtr x; // inputs to the layer
    NodePtr Wx; // weights * inputs
    
    // NodePtr output; would be z = Wx + b (output is already defined in Layer)
};

// Inverse dropout (neuron scaling is applied during training, not during inference)
class Dropout : public Layer {
    public:
    
    Dropout(float p) : p(p) {}

    void construct_forward(NodePtr inputs) override {

        if (is_inference) {
            output = inputs;
            return;
        }

        NodePtr dropout_node = Node::dropout(inputs->getData().getRows(), inputs->getData().getCols(), p, false);
        output = Node::hadamard_product(inputs, dropout_node);
    }

    void update(float lr, float mini_batch_size) override {}

    private:

    float p;
     
};

class Sigmoid : public Layer {
   public:

    void construct_forward(NodePtr inputs) override {
        output = Node::sigmoid(inputs);
    }

    void update(float lr, float mini_batch_size) override {}

    void print() {
        cout << "Sigmoid layer\n";
    }
};

class ReLU : public Layer {
   public:
   
    void construct_forward(NodePtr inputs) override {
        output = Node::activation(inputs, activation_functions::relu, activation_functions::relu_derivative);
    }

    void update(float lr, float mini_batch_size) override {}

    void print() {
        cout << "ReLU layer\n";
    }
};

class Summation : public Layer {
   public:
   
    void construct_forward(NodePtr inputs) override {
        output = inputs->sum();
    }

    void update(float lr, float mini_batch_size) override {}

    void print() {
        cout << "Summation layer\n";
    }
};

#endif