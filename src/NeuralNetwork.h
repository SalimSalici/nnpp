#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "Node.h"
#include "Layers.h"
#include "Sample.h"
#include "utils.h"
#include "Loss.h"

#include <memory>
#include <vector>
#include <iomanip>

typedef struct Evaluation {
    int correct;
    int total;
    float accuracy;
    float loss;
} Evaluation;

class NeuralNetwork {
   public:
    NeuralNetwork(int input_size, bool inputs_separated = false) {
        this->input_size = input_size;
        this->inputs_separated = inputs_separated;
    }

    void add_layer(LayerPtr layer) {
        layers.push_back(layer);
    }

    void setup_mini_batch_size(int mini_batch_size) {

        Node::reset_markers(sorted_nodes);
        sorted_nodes.clear();

        input_layer = make_shared<InputLayer>(input_size, mini_batch_size, inputs_separated);
        setup_computatioal_graph();
        loss = make_loss(loss_type, layers.back()->get_output()->getData().getRows(), mini_batch_size);
        loss->construct_forward(layers.back()->get_output());
        loss->get_loss()->topo_sort(sorted_nodes);
    }

    void setup_computatioal_graph() {
        layers[0]->construct_forward(input_layer, requires_grad);
        for (auto it = layers.begin() + 1; it != layers.end(); ++it) {
            auto& current = *it;
            auto& previous = *(it - 1);
            current->construct_forward(previous, requires_grad);
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

    void process_inputs_separated(bool separated) {
        this->inputs_separated = separated;
    }

    void sgd(Sample* samples[], int samples_count, float lr, int epochs, int mini_batch_size,
                Sample* test_samples[], int test_samples_count) {

        std::cout << "Starting SGD." << std::endl;

        // Evaluation initial_eval = evaluate(test_samples, test_samples_count);
        Evaluation initial_eval = split_evaluate(test_samples, test_samples_count, 4);

        std::cout << "Initial accuracy: " << initial_eval.accuracy * 100 << "%" << std::endl;
        std::cout << "Initial loss: " << initial_eval.loss << std::endl;

        for (int epoch = 0; epoch < epochs; epoch++) {

            set_is_inferece(false);

            auto start = std::chrono::high_resolution_clock::now();

            shuffle_pointers((void**)samples, samples_count);
            setup_mini_batch_size(mini_batch_size);
            loss->set_compute_loss_flag(false);
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
            std::chrono::duration<double, std::milli> train_time = end - start;

            start = std::chrono::high_resolution_clock::now();

            // Evaluation train_eval = evaluate(samples, samples_count);
            Evaluation test_eval = split_evaluate(test_samples, test_samples_count, 8);

            end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> test_time = end - start;

            double total_time = train_time.count() + test_time.count();

            std::cout << std::fixed << std::setprecision(2) << "Epoch " << epoch << " completed - Train_time: "<< train_time.count() / 1000
                << "s - Test time: " << test_time.count() / 1000
                << "s - Total time: " << total_time / 1000 << "s";

            std::cout.unsetf(std::ios::fixed); // Remove the fixed format flag
            std::cout.precision(6); // Reset precision to the default value (commonly 6)
            
            std::cout << " - Loss: " << test_eval.loss << " - Accuracy: " << test_eval.accuracy * 100 << "%" << std::endl;
            
        }
    }

    // For classification tasks
    Evaluation evaluate(Sample* samples[], int samples_count, bool disable_last_layer = false) {

        set_is_inferece(true);

        Evaluation eval = {0, samples_count, 0, 0};
        setup_mini_batch_size(samples_count);
        loss->set_compute_loss_flag(true);

        // For classification tasks we don't need the last layer,
        // just taking the highest logit is enough
        if (disable_last_layer) {
            set_enabled_last_layer(false);
            loss->set_compute_loss_flag(false); // Can't compute loss without last layer
        }

        load_labels(samples, samples_count);
        feedforward(samples, samples_count);        
        auto& output = layers.back()->get_output()->getData();

        if (disable_last_layer)
            set_enabled_last_layer(true);

        eval.loss = loss->get_loss()->getData().getData()[0] / samples_count;

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

    Evaluation split_evaluate(Sample* samples[], int samples_count, int splits, bool disable_last_layer = false) {
        Evaluation total_eval = {0, samples_count, 0, 0};
        int samples_per_split = samples_count / splits;

        for (int i = 0; i < splits; ++i) {
            int start_idx = i * samples_per_split;
            int end_idx = (i == splits - 1) ? samples_count : (i + 1) * samples_per_split;
            int current_split_size = end_idx - start_idx;

            Evaluation split_eval = evaluate(samples + start_idx, current_split_size, disable_last_layer);

            total_eval.correct += split_eval.correct;
            total_eval.loss += split_eval.loss * current_split_size;  // Weighted sum of losses
        }

        total_eval.loss /= samples_count;  // Calculate average loss
        total_eval.accuracy = static_cast<float>(total_eval.correct) / samples_count;

        return total_eval;
    }

    void set_enabled_last_layer(bool enabled) {
        layers.back()->set_enabled(enabled);
    }

    void set_is_inferece(bool is_inference) {
        for (auto& layer : layers) {
            layer->set_is_inference(is_inference);
        }
        requires_grad = !is_inference;
    }

    LayerPtr get_last_layer() {
        return layers.back();
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

    void set_loss_type(LossType loss_type) {
        this->loss_type = loss_type;
    }

    void freeze_params() {
        for (auto& layer : layers) {
            layer->freeze_params();
        }
    }

    shared_ptr<InputLayer> get_input_layer() {
        return input_layer;
    }

   private:
    int input_size;
    bool inputs_separated;
    shared_ptr<Loss> loss;
    shared_ptr<InputLayer> input_layer;
    vector<LayerPtr> layers; // input layer is not included
    deque<NodePtr> sorted_nodes;
    LossType loss_type = LossType::MSE;
    bool requires_grad = true;
};

#endif