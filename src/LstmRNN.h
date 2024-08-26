#ifndef LSTMRNN_RNN_H
#define LSTMRNN_RNN_H

#include "SampleRNN.h"
#include "Node.h"
#include "utils.h"
#include "activation_functions.h"

#include <utility>
#include <tuple>
#include <memory>
#include <chrono>
#include <iomanip>

using namespace std;

class LstmRNN {

public:

    LstmRNN(int input_size, int state_size, int output_size) {
        Wf = make_shared<Node>(state_size, input_size, true);
        Uf = make_shared<Node>(state_size, state_size, true);
        bf = make_shared<Node>(state_size, 1, true);

        Wi = make_shared<Node>(state_size, input_size, true);
        Ui = make_shared<Node>(state_size, state_size, true);
        bi = make_shared<Node>(state_size, 1, true);

        Wc = make_shared<Node>(state_size, input_size, true);
        Uc = make_shared<Node>(state_size, state_size, true);
        bc = make_shared<Node>(state_size, 1, true);

        Wo = make_shared<Node>(state_size, input_size, true);
        Uo = make_shared<Node>(state_size, state_size, true);
        bo = make_shared<Node>(state_size, 1, true);

        Wlogits = make_shared<Node>(output_size, state_size, true);
        blogits = make_shared<Node>(output_size, 1, true);

        Wf->zero_grad();
        Uf->zero_grad();
        bf->zero_grad();

        Wi->zero_grad();
        Ui->zero_grad();
        bi->zero_grad();

        Wc->zero_grad();
        Uc->zero_grad();
        bc->zero_grad();

        Wo->zero_grad();
        Uo->zero_grad();
        bo->zero_grad();

        Wlogits->zero_grad();
        blogits->zero_grad();
    }

    // returns <logits, cell state, hidden state>
    std::tuple<NodePtr, NodePtr, NodePtr> construct_forward_to_logits_and_state(NodePtr inputs, NodePtr c_state, NodePtr h_state, int mini_batch_size) {
        NodePtr ft = Node::mat_plus_vec(Node::plus(Node::matmul(Wf, inputs, true), Node::matmul(Uf, h_state, true), true), bf, true);
        ft = Node::sigmoid(ft, true);

        NodePtr it = Node::mat_plus_vec(Node::plus(Node::matmul(Wi, inputs, true), Node::matmul(Ui, h_state, true), true), bi, true);
        it = Node::sigmoid(it, true);

        NodePtr c = Node::mat_plus_vec(Node::plus(Node::matmul(Wc, inputs, true), Node::matmul(Uc, h_state, true), true), bc, true);
        c = Node::tanh(c, true);

        NodePtr ct = Node::plus(Node::hadamard_product(ft, c_state, true), Node::hadamard_product(it, c, true), true);
        
        NodePtr ot = Node::mat_plus_vec(Node::plus(Node::matmul(Wo, inputs, true), Node::matmul(Uo, h_state, true), true), bo, true);
        ot = Node::sigmoid(ot, true);

        NodePtr ht = Node::hadamard_product(ot, Node::tanh(ct, true), true);

        NodePtr logits = Node::mat_plus_vec(Node::matmul(Wlogits, ht, true), blogits, true);

        return std::make_tuple(logits, ct, ht);
    }

    NodePtr construct_forward_to_total_loss(SampleRNN* samples[], int sequence_length, int mini_batch_size) {
        NodePtr cell_state = make_shared<Node>(Ui->getData().getCols(), mini_batch_size, false);
        cell_state->getData().fill(0);

        NodePtr hidden_state = make_shared<Node>(Ui->getData().getCols(), mini_batch_size, false);
        hidden_state->getData().fill(0);

        NodePtr total_loss = make_shared<Node>(1, 1, false);
        total_loss->getData().fill(0);

        for (int i = 0; i < sequence_length; i++) {
            std::pair<NodePtr, NodePtr> inputs_and_outputs = load_mini_batch_inputs_and_outputs(samples, i, mini_batch_size);
            NodePtr inputs = inputs_and_outputs.first;
            NodePtr outputs = inputs_and_outputs.second;
            inputs = Node::transpose(inputs, false);
            outputs = Node::transpose(outputs, false);

            std::tuple<NodePtr, NodePtr, NodePtr> logits_and_state = construct_forward_to_logits_and_state(inputs, cell_state, hidden_state, mini_batch_size);
            NodePtr logits = std::get<0>(logits_and_state);
            cell_state = std::get<1>(logits_and_state);
            hidden_state = std::get<2>(logits_and_state);

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
        initialize_weights(Wf);
        initialize_weights(Uf);
        bf->getData().fill(0);

        initialize_weights(Wi);
        initialize_weights(Ui);
        bi->getData().fill(0);

        initialize_weights(Wc);
        initialize_weights(Uc);
        bc->getData().fill(0);

        initialize_weights(Wo);
        initialize_weights(Uo);
        bo->getData().fill(0);

        initialize_weights(Wlogits);
        blogits->getData().fill(0);
    }

    void initialize_weights(NodePtr w) {
        float mean = 0;
        float std = sqrt(1.0 / w->getData().getCols());
        float args[2] = {mean, std};
        Mat::apply(w->getData(), w->getData(), normal_sample_applier, static_cast<void*>(args));
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
        float coeff = lr / (float)mini_batch_size;

        Wf->getData() -= Mat::scale(Wf->getGrad(), coeff);
        Uf->getData() -= Mat::scale(Uf->getGrad(), coeff);
        bf->getData() -= Mat::scale(bf->getGrad(), coeff);

        Wi->getData() -= Mat::scale(Wi->getGrad(), coeff);
        Ui->getData() -= Mat::scale(Ui->getGrad(), coeff);
        bi->getData() -= Mat::scale(bi->getGrad(), coeff);

        Wc->getData() -= Mat::scale(Wc->getGrad(), coeff);
        Uc->getData() -= Mat::scale(Uc->getGrad(), coeff);
        bc->getData() -= Mat::scale(bc->getGrad(), coeff);

        Wo->getData() -= Mat::scale(Wo->getGrad(), coeff);
        Uo->getData() -= Mat::scale(Uo->getGrad(), coeff);
        bo->getData() -= Mat::scale(bo->getGrad(), coeff);

        Wlogits->getData() -= Mat::scale(Wlogits->getGrad(), coeff);
        blogits->getData() -= Mat::scale(blogits->getGrad(), coeff);
    }

    void sgd(SampleRNN* samples[], int samples_count, float lr, int epochs, int mini_batch_size) {
        std::cout << "Starting LstmRNN SGD." << std::endl;

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
            NodePtr cell_state = make_shared<Node>(Ui->getData().getCols(), 1, false);
            cell_state->getData().fill(0);

            NodePtr hidden_state = make_shared<Node>(Ui->getData().getCols(), 1, false);
            hidden_state->getData().fill(0);

            string seq{initial_char};
            char last_char = initial_char;

            for (int j = 0; j < seq_length; j++) {
                string char_str{last_char};
                SampleRNN s(char_str, 1);
                NodePtr inputs = make_shared<Node>(s.get_input_seq_one_hots(), false);
                inputs = Node::transpose(inputs, false);

                std::tuple<NodePtr, NodePtr, NodePtr> logits_and_state = construct_forward_to_logits_and_state(inputs, cell_state, hidden_state, 1);
                NodePtr logits = std::get<0>(logits_and_state);
                cell_state = std::get<1>(logits_and_state);
                hidden_state = std::get<2>(logits_and_state);

                NodePtr sum_logits = logits->sum(false);
                NodePtr sum_cell_state = cell_state->sum(false);
                NodePtr sum_hidden_state = hidden_state->sum(false);
                NodePtr sum = Node::plus(sum_logits, sum_cell_state, false);
                sum = Node::plus(sum, sum_hidden_state, false);
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
    NodePtr Wf;
    NodePtr Uf;
    NodePtr bf;

    NodePtr Wi;
    NodePtr Ui;
    NodePtr bi;

    NodePtr Wc;
    NodePtr Uc;
    NodePtr bc;

    NodePtr Wo;
    NodePtr Uo;
    NodePtr bo;

    NodePtr Wlogits;
    NodePtr blogits;

    deque<NodePtr> sorted_nodes;

};

#endif