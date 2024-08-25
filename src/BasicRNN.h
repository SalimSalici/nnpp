#ifndef BASIC_RNN_H
#define BASIC_RNN_H

#include "SampleRNN.h"
#include "Node.h"
#include "utils.h"
#include "activation_functions.h"

#include <utility>
#include <memory>
#include <chrono>
#include <iomanip>

using namespace std;

class BasicRNN {

public:

    BasicRNN(int input_size, int state_size, int output_size) {
        Wx = make_shared<Node>(state_size, input_size, true);
        Wh = make_shared<Node>(state_size, state_size, true);
        bs = make_shared<Node>(state_size, 1, true);
        Wy = make_shared<Node>(output_size, state_size, true);
        by = make_shared<Node>(output_size, 1, true);

        Wx->zero_grad();
        Wh->zero_grad();
        bs->zero_grad();
        Wy->zero_grad();
        by->zero_grad();
    }

    // returns <logits, hidden state>
    std::pair<NodePtr, NodePtr> construct_forward_to_logits_and_state(NodePtr inputs, NodePtr state, int mini_batch_size) {
        NodePtr state_and_inputs = Node::plus(Node::matmul(Wh, state, true), Node::matmul(Wx, inputs, true), true);
        NodePtr z = Node::mat_plus_vec(state_and_inputs, bs, true);
        NodePtr h = Node::tanh(z, true);
        // NodePtr h = Node::tanh(Node::plus(state, z, true), true);
        // NodePtr h = Node::plus(state, Node::tanh(z, true), true);
        NodePtr logits = Node::mat_plus_vec(Node::matmul(Wy, h, true), by, true);

        return std::make_pair(logits, h);
    }

    NodePtr construct_forward_to_total_loss(SampleRNN* samples[], int sequence_length, int mini_batch_size) {
        NodePtr state = make_shared<Node>(Wh->getData().getCols(), mini_batch_size, false);
        state->getData().fill(0);

        NodePtr total_loss = make_shared<Node>(1, 1, false);
        total_loss->getData().fill(0);

        for (int i = 0; i < sequence_length; i++) {
            std::pair<NodePtr, NodePtr> inputs_and_outputs = load_mini_batch_inputs_and_outputs(samples, i, mini_batch_size);
            NodePtr inputs = inputs_and_outputs.first;
            NodePtr outputs = inputs_and_outputs.second;
            inputs = Node::transpose(inputs, false);
            outputs = Node::transpose(outputs, false);

            std::pair<NodePtr, NodePtr> logits_and_state = construct_forward_to_logits_and_state(inputs, state, mini_batch_size);
            NodePtr logits = logits_and_state.first;
            state = logits_and_state.second;

            NodePtr loss = make_shared<CCENode>(outputs, logits, true);
            total_loss = Node::plus(total_loss, loss, true);
        }

        return total_loss;
    }

    // returns <input batch, output batch>
    std::pair<NodePtr, NodePtr> load_mini_batch_inputs_and_outputs(SampleRNN* samples[], int seq_idx, int mini_batch_size) {
        int vocab_size = samples[0]->get_input_seq_one_hots().getCols();
        NodePtr inputs = make_shared<Node>(mini_batch_size, vocab_size, false);
        NodePtr outputs = make_shared<Node>(mini_batch_size, vocab_size, false);


        float* cur_inputs_data = inputs->getData().getData();
        float* cur_outputs_data = outputs->getData().getData();
        int inputs_down = inputs->getData().getDown();
        int outputs_down = outputs->getData().getDown();
        for (int i = 0; i < mini_batch_size; i++) {
            // cout << "Loading minibatch inputs: " << SampleRNN::one_hots_to_string(samples[i]->get_input_seq_one_hots()) << endl;
            // cout << "Loading minibatch outputs: " << SampleRNN::one_hots_to_string(samples[i]->get_output_seq_one_hots()) << endl;

            Mat& inputs_one_hots = samples[i]->get_input_seq_one_hots();
            memcpy(cur_inputs_data, inputs_one_hots.getData() + seq_idx * inputs_one_hots.getDown(), vocab_size * sizeof(float));
            cur_inputs_data += inputs_down;

            Mat& outputs_one_hots = samples[i]->get_output_seq_one_hots();
            memcpy(cur_outputs_data, outputs_one_hots.getData() + seq_idx * outputs_one_hots.getDown(), vocab_size * sizeof(float));
            cur_outputs_data += outputs_down;
        }

        // cout << "Inputs: " << SampleRNN::one_hots_to_string(inputs->getData()) << endl;
        // cout << "Outputs: " << SampleRNN::one_hots_to_string(outputs->getData()) << endl;
        
        return std::make_pair(inputs, outputs);
    }

    void initialize_params() {
        float mean = 0;

        // sqrt(2 / n_in)
        float std_x = sqrt(1.0 / Wx->getData().getCols());
        float args_x[2] = {mean, std_x};
        Mat::apply(Wx->getData(), Wx->getData(), normal_sample_applier, static_cast<void*>(args_x));
        bs->getData().fill(0);

        float std_h = sqrt(1.0 / Wh->getData().getCols());
        float args_h[2] = {mean, std_h};
        Mat::apply(Wh->getData(), Wh->getData(), normal_sample_applier, static_cast<void*>(args_h));

        float std_y = sqrt(1.0 / Wy->getData().getCols());
        float args_y[2] = {mean, std_y};
        Mat::apply(Wy->getData(), Wy->getData(), normal_sample_applier, static_cast<void*>(args_y));
        by->getData().fill(0);
    }

    float feedforward(NodePtr loss) {
        Node::reset_markers(sorted_nodes);
        sorted_nodes.clear();
        loss->topo_sort(sorted_nodes);
        Node::forwardPass(sorted_nodes);
        return loss->getData().getData()[0];
    }

    void backprop() {
        Node::backwardPass(sorted_nodes);
    }

    void update(float lr, int mini_batch_size) {
        Wx->getData() -= Mat::scale(Wx->getGrad(), lr / mini_batch_size);
        Wh->getData() -= Mat::scale(Wh->getGrad(), lr / mini_batch_size);
        bs->getData() -= Mat::scale(bs->getGrad(), lr / mini_batch_size);
        Wy->getData() -= Mat::scale(Wy->getGrad(), lr / mini_batch_size);
        by->getData() -= Mat::scale(by->getGrad(), lr / mini_batch_size);
    }

    void sgd(SampleRNN* samples[], int samples_count, float lr, int epochs, int mini_batch_size) {
        std::cout << "Starting BasicRNN SGD." << std::endl;

        int seq_length = samples[0]->get_input_seq_one_hots().getRows();

        // // Evaluation initial_eval = evaluate(test_samples, test_samples_count);
        // Evaluation initial_eval = split_evaluate(test_samples, test_samples_count, 4);

        // std::cout << "Initial accuracy: " << initial_eval.accuracy * 100 << "%" << std::endl;
        // std::cout << "Initial loss: " << initial_eval.loss << std::endl;

        for (int epoch = 0; epoch < epochs; epoch++) {

            // set_is_inferece(false);

            auto start = chrono::high_resolution_clock::now();

            shuffle_pointers((void**)samples, samples_count);
            // setup_mini_batch_size(mini_batch_size);
            // loss->set_compute_loss_flag(false);
            int mini_batch_tracker = 0;
            NodePtr total_loss;
            float loss_val = 0;
            while (mini_batch_tracker + mini_batch_size <= samples_count) {
                total_loss = construct_forward_to_total_loss(samples + mini_batch_tracker, samples[0]->get_input_seq_one_hots().getRows(), mini_batch_size);

                Node::reset_markers(sorted_nodes);
                Node::zero_grad(sorted_nodes);
                sorted_nodes.clear();
                total_loss->topo_sort(sorted_nodes);
                Node::zero_grad(sorted_nodes);
                
                feedforward(total_loss);
                backprop();
                update(lr, mini_batch_size);
                mini_batch_tracker += mini_batch_size;

                loss_val += total_loss->getData().getData()[0];
            }

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> train_time = end - start;

            // start = std::chrono::high_resolution_clock::now();

            // // Evaluation train_eval = evaluate(samples, samples_count);
            // Evaluation test_eval = split_evaluate(test_samples, test_samples_count, 8);

            // end = std::chrono::high_resolution_clock::now();
            // std::chrono::duration<double, std::milli> test_time = end - start;

            // double total_time = train_time.count() + test_time.count();

            std::cout << std::fixed << std::setprecision(2) << "Epoch " << epoch << " completed - Train_time: " << train_time.count() / 1000;
            // std::cout << std::fixed << std::setprecision(2) << "Epoch " << epoch << " completed - Train_time: "<< train_time.count() / 1000
                // << "s - Test time: " << test_time.count() / 1000
                // << "s - Total time: " << train_time / 1000 << "s";

            std::cout.unsetf(std::ios::fixed); // Remove the fixed format flag
            std::cout.precision(6); // Reset precision to the default value (commonly 6)

            float loss = loss_val / (samples_count * seq_length);
            
            std::cout << " - Loss: " << loss << std::endl;
        }
    }

    vector<string> generate(char initial_char, int seq_length, int count) {
        vector<string> generated;

        for (int i = 0; i < count; i++) {
            NodePtr state = make_shared<Node>(Wh->getData().getCols(), 1, false);
            state->getData().fill(0);

            string seq{initial_char};
            char last_char = initial_char;

            for (int j = 0; j < seq_length; j++) {
                string char_str{last_char};
                SampleRNN s(char_str, 1);
                NodePtr inputs = make_shared<Node>(s.get_input_seq_one_hots(), false);
                inputs = Node::transpose(inputs, false);

                std::pair<NodePtr, NodePtr> logits_and_state = construct_forward_to_logits_and_state(inputs, state, 1);
                NodePtr logits = logits_and_state.first;
                state = logits_and_state.second;

                NodePtr sum_logits = logits->sum(false);
                NodePtr sum_state = state->sum(false);
                NodePtr sum = Node::plus(sum_logits, sum_state, false);
                sorted_nodes.clear();
                sum->topo_sort(sorted_nodes);
                Node::forwardPass(sorted_nodes);
                Node::reset_markers(sorted_nodes);

                Mat softm(logits->getData().getRows(), logits->getData().getCols());
                Mat::softmax(softm, logits->getData(), false);

                // Generate a random number between 0 and 1
                float random = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
                float cumulative_prob = 0.0f;
                int idx = -1;

                // Sample based on the probability distribution
                for (int k = 0; k < softm.getRows(); ++k) {
                    cumulative_prob += softm.getElement(k, 0);
                    if (random < cumulative_prob) {
                        idx = k;
                        break;
                    }
                }

                // If we somehow didn't select an index (shouldn't happen), default to the last one
                if (idx == -1) idx = softm.getRows() - 1;

                last_char = SampleRNN::itoc(idx);
                seq.push_back(last_char);
            }

            generated.push_back(seq);
        }

        return generated;
    }

    void zero_grad() {
        // Wx->zero_grad();
        // Wh->zero_grad();
        // bs->zero_grad();
        // Wy->zero_grad();
        // by->zero_grad();
        Node::zero_grad(sorted_nodes);
    }

private:

    NodePtr Wx;
    NodePtr Wh;
    NodePtr bs;
    NodePtr Wy;
    NodePtr by;

    deque<NodePtr> sorted_nodes;

};

#endif