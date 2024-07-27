#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "Node.h"
#include "Layers.h"
#include "Sample.h"
#include "utils.h"
#include "Loss.h"

#include <memory>
#include <vector>

class NeuralNetwork {
   public:
    NeuralNetwork(int input_size) {
        this->input_size = input_size;
    }

    void add_layer(LayerPtr layer) {
        layers.push_back(layer);
    }

    void setup_mini_batch_size(int mini_batch_size) {
        input_layer = make_shared<InputLayer>(input_size, mini_batch_size);
        setup_comp_graph();
        loss = make_shared<Loss>(layers.back()->get_output()->getData().getRows(), mini_batch_size);
        loss->construct_forward(layers.back()->get_output());
        loss->get_loss()->topo_sort(sorted_nodes);
    }

    void setup_comp_graph() {
        layers[0]->construct_forward(input_layer->get_output());
        for (auto it = layers.begin() + 1; it != layers.end(); ++it) {
            auto& current = *it;
            auto& previous = *(it - 1);
            current->construct_forward(previous->get_output());
        }
    }

    void initialize_layers() {
        for (auto& layer : layers)
            layer->initialize();
    }

    void feed_forward(Sample samples[], int mini_batch_size) {
        input_layer->load_train_samples(samples, mini_batch_size);
        Node::forwardPass(sorted_nodes);
    }
    
    void feed_forward(Sample sample) {
        return feed_forward(&sample, 1);
    }

    void compute_loss(Sample samples[], int mini_batch_size) {
        feed_forward(samples, mini_batch_size);
    }

    void backprop() {
        loss->backprop();
        // Node::backwardPass(sorted_nodes);
    }

    void update(float lr, float mini_batch_size) {
        for (auto& layer : layers) {
            layer->update(lr, mini_batch_size);
        }
    }

    void sgd(shared_ptr<Sample> samples[], int samples_count, float lr, int epochs, int mini_batch_size) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            shuffleArray(samples, samples_count);
            int mini_batch_tracker = 0;
            while (mini_batch_tracker + mini_batch_size <= samples_count) {
                for (int i = 0; i < mini_batch_size; i++) {
                    
                }
                update(lr, mini_batch_size);
                mini_batch_tracker += mini_batch_size;
            }
        }
    }

    Mat& get_output_data() {
        return layers.back()->get_output()->getData();
    }

    shared_ptr<Loss> get_loss() {
        return loss;
    }

    void print() {
        for (auto& layer : layers) {
            layer->print();
        }
    }

    int get_input_size() {
        return input_size;
    }

   private:
    int input_size;
    shared_ptr<Loss> loss;
    shared_ptr<InputLayer> input_layer;
    vector<LayerPtr> layers; // input layer is not included
    deque<NodePtr> sorted_nodes;
};

#endif