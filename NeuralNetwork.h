#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "Node.h"
#include "Layers.h"

#include <memory>
#include <vector>

class NeuralNetwork {
   public:
    void add_layer(LayerPtr layer) {
        layers.push_back(layer);
    }

    void initialize() {
        layers.front()->construct_forward(nullptr);
        for (auto it = layers.begin() + 1; it != layers.end(); ++it) {
            auto& current = *it;
            auto& previous = *(it - 1);
            current->construct_forward(previous->get_output());
        }
    }

    void feed_forward() {
        layers.back()->get_output()->topo_sort(sorted_nodes);
        Node::forwardPass(sorted_nodes);
    }

    void backprop() {
        layers.back()->get_output()->backprop(sorted_nodes);
        // Node::backwardPass(sorted_nodes);
    }

    void update(float lr, float batch_size) {
        for (auto& layer : layers) {
            layer->update(lr, batch_size);
        }
    }

    Mat& get_output_data() {
        return layers.back()->get_output()->getData();
    }

    void print() {
        for (auto& layer : layers) {
            layer->print();
        }
    }

   private:
    vector<LayerPtr> layers;
    deque<NodePtr> sorted_nodes;
};

#endif