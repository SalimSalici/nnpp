#ifndef RES_BLOCK_H
#define RES_BLOCK_H

#include "Layers.h"
#include "Node.h"

#include <vector>

using namespace std;

class ResBlock : public Layer {
    public:

    void add_layer(LayerPtr layer) {
        layers.push_back(layer);
    }

    void initialize() override {
        for (auto& layer : layers) {
            layer->initialize();
        }
    }

    void construct_forward(LayerPtr previous, bool requires_grad) override {
        LayerPtr input_layer = previous;
        NodePtr input_node = previous->get_output();
        
        for (auto& layer : layers) {
            layer->construct_forward(previous, requires_grad);
            previous = layer;
        }

        if (input_node->getRows() != previous->get_output()->getRows()
        || input_node->getCols() != previous->get_output()->getCols()
        || input_layer->get_samples_along_cols() != previous->get_samples_along_cols()) {
            throw std::invalid_argument("ResBlock: input and output dimensions do not match");
        }

        output = Node::plus(input_node, previous->get_output(), requires_grad);
    }

    void update(float lr, float mini_batch_size) override {
        if (params_freezed) return;
        for (auto& layer : layers) {
            layer->update(lr, mini_batch_size);
        }
    }

    void print() override {
        cout << "ResBlock layer" << endl;
        for (auto& layer : layers) {
            cout << "   -> ";
            layer->print();
        }
    }

    void set_is_inference(bool is_inference) override {
        for (auto& layer : layers) {
            layer->set_is_inference(is_inference);
        }
    }

    bool get_samples_along_cols() override {
        return layers.back()->get_samples_along_cols();
    }

    void freeze_params() override {
        for (auto& layer : layers) {
            layer->freeze_params();
        }
    }

    void unfreeze_params() override {
        for (auto& layer : layers) {
            layer->unfreeze_params();
        }
    }

    private:

    vector<LayerPtr> layers;
};

#endif
