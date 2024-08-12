#ifndef SAMPLE_RNN_H
#define SAMPLE_RNN_H

#include "Mat.h"

#include <iostream>
#include <string>
#include <map>
#include <memory>

using namespace std;

class SampleRNN {
public:
    SampleRNN(string sequence, int seq_length)
    : seq_str(sequence), seq_length(seq_length),
    input_seq_one_hots(seq_length, _ctoi.size()), output_seq_one_hots(seq_length, _ctoi.size()) {
        init_input_seq();
        init_output_seq();
    }

    string getSequence() {
        return seq_str;
    }

    static void set_ctoi_and_itoc(map<char, int> ctoi, map<int, char> itoc) {
        SampleRNN::_ctoi = ctoi;
        SampleRNN::_itoc = itoc;
    }

    float* get_input_sample_at_pos(int pos) {
        if (pos < 0 || pos >= seq_length)
            throw runtime_error("get_input_sample_at_pos: pos out of bounds");

        return input_seq_one_hots.getData() + pos * input_seq_one_hots.getDown();
    }

    float* get_output_sample_at_pos(int pos) {
        if (pos < 0 || pos >= seq_length)
            throw runtime_error("get_output_sample_at_pos: pos out of bounds");

        return output_seq_one_hots.getData() + pos * output_seq_one_hots.getDown();
    }

    void init_input_seq() {

        input_seq_one_hots.zero();

        size_t i = 0;
        float* input_seq_one_hots_data = input_seq_one_hots.getData();
        int down = input_seq_one_hots.getDown();
        for (; i < (size_t)seq_length && i < seq_str.length(); i++) {
            input_seq_one_hots_data[i * down + _ctoi[seq_str[i]]] = 1.0;
        }

        for (; i < (size_t)seq_length; i++) {
            input_seq_one_hots_data[i * down + _ctoi['_']] = 1.0;
        }
    }

    void init_output_seq() {

        output_seq_one_hots.zero();

        size_t i = 1;
        float* output_seq_one_hots_data = output_seq_one_hots.getData();
        int down = output_seq_one_hots.getDown();
        for (; i < (size_t)seq_length && i < seq_str.length(); i++) {
            output_seq_one_hots_data[(i-1) * down + _ctoi[seq_str[i]]] = 1.0;
        }

        for (; i < (size_t)seq_length + 1; i++) {
            output_seq_one_hots_data[(i-1) * down + _ctoi['_']] = 1.0;
        }
    }

    static string one_hots_to_string(Mat& one_hots) {
        if ((size_t)one_hots.getCols() != _ctoi.size()) {
            throw runtime_error("one_hots_to_string: one_hots dimensions do not match the sequence dimensions");
        }

        string str;
        for (int i = 0; i < one_hots.getRows(); i++) {
            for (size_t j = 0; j < _ctoi.size(); j++) {
                if (one_hots.getElement(i, j) == 1.0) {
                    str += _itoc[j];
                    break;
                }
            }
        }

        return str;
    }

    Mat& get_input_seq_one_hots() {
        return input_seq_one_hots;
    }

    Mat& get_output_seq_one_hots() {
        return output_seq_one_hots;
    }

protected:

    string seq_str;
    int seq_length;
    Mat input_seq_one_hots;
    Mat output_seq_one_hots;
    static map<char, int> _ctoi;
    static map<int, char> _itoc;

};

#endif