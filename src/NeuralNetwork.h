#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "Node.h"
#include "Layers.h"
#include "Sample.h"
#include "utils.h"
#include "Loss.h"

#include <memory>
#include <vector>

typedef struct Evaluation {
    int correct;
    int total;
    float accuracy;
    float loss;
} Evaluation;

class NeuralNetwork {
   public:
    NeuralNetwork(int input_size) {
        this->input_size = input_size;
    }

    void add_layer(LayerPtr layer) {
        layers.push_back(layer);
    }

    void setup_mini_batch_size(int mini_batch_size) {

        Node::reset_markers(sorted_nodes);
        sorted_nodes.clear();

        input_layer = make_shared<InputLayer>(input_size, mini_batch_size);
        setup_computatioal_graph();
        loss = make_shared<Loss>(layers.back()->get_output()->getData().getRows(), mini_batch_size);
        loss->construct_forward(layers.back()->get_output());
        loss->get_loss()->topo_sort(sorted_nodes);
    }

    void setup_computatioal_graph() {
        layers[0]->construct_forward(input_layer->get_output());
        for (auto it = layers.begin() + 1; it != layers.end(); ++it) {
            auto& current = *it;
            auto& previous = *(it - 1);
            current->construct_forward(previous->get_output());
        }
    }

    // Initalizes params
    void initialize_layers() {
        for (auto& layer : layers)
            layer->initialize();
    }

    void load_labels(Sample* samples[], int mini_batch_size) {
        loss->load_labels(samples, mini_batch_size);
    }

    void feedforward(Sample* samples[], int mini_batch_size) {
        input_layer->load_train_samples(samples, mini_batch_size);
        Node::forwardPass(sorted_nodes);
    }
    
    void feedforward(Sample* sample) {
        return feedforward(&sample, 1);
    }

    void backprop() {
        Node::backwardPass(sorted_nodes);
    }

    void update(float lr, float mini_batch_size) {
        for (auto& layer : layers) {
            layer->update(lr, mini_batch_size);
        }
    }

    void sgd(Sample* samples[], int samples_count, float lr, int epochs, int mini_batch_size,
                Sample* test_samples[], int test_samples_count) {

        for (int epoch = 0; epoch < epochs; epoch++) {

            auto start = std::chrono::high_resolution_clock::now();

            shuffle_pointers((void**)samples, samples_count);
            setup_mini_batch_size(mini_batch_size);
            int mini_batch_tracker = 0;
            while (mini_batch_tracker + mini_batch_size <= samples_count) {
                Node::zero_grad(sorted_nodes);
                load_labels(samples + mini_batch_tracker, mini_batch_size);
                feedforward(samples + mini_batch_tracker, mini_batch_size);
                backprop();
                update(lr, mini_batch_size);
                mini_batch_tracker += mini_batch_size;
            }

            auto end = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double, std::milli> duration = end - start;

            // Evaluation train_eval = evaluate(samples, samples_count);
            Evaluation test_eval = evaluate(test_samples, test_samples_count);

            std::cout << "Epoch " << epoch << " completed. Took "<< duration.count() << "ms. Loss: " << test_eval.loss
                << " - Accuracy: " << test_eval.accuracy << std::endl;
        }
    }

    // For classification tasks
    Evaluation evaluate(Sample* samples[], int samples_count) {
        Evaluation eval = {0, samples_count, 0, 0};
        setup_mini_batch_size(samples_count);
        load_labels(samples, samples_count);
        feedforward(samples, samples_count);

        eval.loss = loss->get_loss()->getData().getData()[0] / samples_count;

        auto& output = layers.back()->get_output()->getData();
        for (int i = 0; i < samples_count; i++) {
            
            int max_index = 0;
            float max_value = output.getElement(0, i);


            for (int j = 1; j < output.getRows(); j++) {
                float value = output.getElement(j, i);
                if (value > max_value) {
                    max_value = value;
                    max_index = j;
                }
            }

            if (samples[i]->getLabel()[max_index] == 1)
                eval.correct++;
        }

        eval.correct = eval.correct;
        eval.accuracy = (float)eval.correct / samples_count;

        return eval;
    }

    Mat& get_output_data() {
        return layers.back()->get_output()->getData();
    }

    NodePtr get_output() {
        return layers.back()->get_output();
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