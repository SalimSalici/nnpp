#ifndef LOSS_H
#define LOSS_H

#include "Node.h"

class Loss {
    public:

    Loss(int size, int mini_batch_size) {
        labels = make_shared<Node>(mini_batch_size, size, false);
        labels->getData().transpose();
        labels->getGrad().transpose();
        this->size = size;
    }

    void construct_forward(NodePtr predictions) {
        // MSE loss
        // loss = Node::pow(*predictions - labels, 2)->sum();
        loss = make_shared<RSSNode>(predictions, labels, true);
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
    }

    void set_compute_flag(bool flag) {
        loss->set_compute_flag(flag);
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
    // NodePtr loss;
    shared_ptr<RSSNode> loss;

    int size;
    int mini_batch_size;
};

#endif