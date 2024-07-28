#ifndef LOSS_H
#define LOSS_H

#include "Node.h"

enum LossType {
    MSE, // Mean Squared Error
    BCE, // Binary Cross Entropy - Applies sigmoid to the last layer and then calculates the loss
    // CCE, // Categorical Cross Entropy - Applies softmax to the last layer and then calculates the loss
};

class Loss {
    public:

    Loss(int size, int mini_batch_size) {
        labels = make_shared<Node>(mini_batch_size, size, false);
        labels->getData().transpose();
        labels->getGrad().transpose();
        this->size = size;
    }

    virtual void construct_forward(NodePtr predictions) = 0;

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

    void set_compute_loss_flag(bool flag) {
        loss->set_enabled(flag);
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

    protected:

    NodePtr labels;
    shared_ptr<Node> loss;

    int size;
    int mini_batch_size;
};

class MSELoss : public Loss {
    public:

    MSELoss(int size, int mini_batch_size) : Loss(size, mini_batch_size) {}

    void construct_forward(NodePtr predictions) override {
        // Using RSS (Residual Sum of Squares) and not MSE because the params update function will divide by mini_batch_size
        loss = make_shared<RSSNode>(labels, predictions, true);
    }
};

class BCELoss : public Loss {
    public:

    BCELoss(int size, int mini_batch_size) : Loss(size, mini_batch_size) {}

    void construct_forward(NodePtr logits) override {
        loss = make_shared<BCENode>(labels, logits, true);
    }
};

shared_ptr<Loss> make_loss(LossType loss_type, int size, int mini_batch_size) {
    switch (loss_type) {
        case MSE:
            return make_shared<MSELoss>(size, mini_batch_size);
        case BCE:
            return make_shared<BCELoss>(size, mini_batch_size);
        default:
            throw std::invalid_argument("Error: make_loss - Invalid loss type");
    }
}

#endif