#ifndef LOSS_H
#define LOSS_H

#include "Node.h"

class Loss {
    public:

    Loss(int size, int mini_batch_size) {
        labels = make_shared<Node>(mini_batch_size, size);
        labels->getData().transpose();
        labels->getGrad().transpose();
        this->size = size;
    }

    void construct_forward(NodePtr inputs) {
        // MSE loss
        loss = Node::pow(*inputs - labels, 2)->sum();
    }

    void load_labels(Sample* samples[], int mini_batch_size) {
        if (mini_batch_size != labels->getData().getCols()) {
            throw std::invalid_argument("Error: Loss::load_labels - mini_batch_size != output->getData().getCols()");
            return;
        }

        if (samples[0]->get_label_size() != size) {
            throw std::invalid_argument("Error: Loss::load_labels - samples[0].get_data_size() != size");
            return;
        }

        float* data = labels->getData().getData();

        for (int i = 0, j = 0; j < mini_batch_size; i += size, j++)
            std::memcpy(data + i, samples[j]->getLabel(), size * sizeof(float));

        // std::cout << "Labels: ";
        // labels->getData().print();
    }

    NodePtr get_loss() {
        return loss;
    }

    void backprop() {
        loss->backprop();
    }

    void print() {
        std::cout << "Loss: ";
        loss->getData().print();
    }

    private:

    NodePtr labels;
    NodePtr loss;

    int size;
    int mini_batch_size;
};

#endif