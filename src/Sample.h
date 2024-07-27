#ifndef SAMPLE_H
#define SAMPLE_H

#include "Mat.h"
#include "mnist_loader.h"

#include <cstring>
#include <memory>
#include <malloc.h>

class Sample {
    public:
    Sample(const float* data, int data_size, const float* label, int label_size) {
        this->data = (float*)malloc(data_size * sizeof(float));
        this->label = (float*)malloc(label_size * sizeof(float));
        this->data_size = data_size;
        this->label_size = label_size;
        std::memcpy(this->data, data, data_size * sizeof(float));
        std::memcpy(this->label, label, label_size * sizeof(float));
    }

    ~Sample() {
        free(data);
        free(label);
    }

    float* getData() { return data; }
    float* getLabel() { return label; }

    int index_from_label() {
        for (int i = 0; i < 10; i++) {
            if (label[i] == 1) {
                return i;
            }
        }
        return -1;
    }

    void print_data() {
        for (int i = 0; i < data_size; i++) {
            cout << data[i] << " ";
        }
        cout << endl;
    }

    void print_label() {
        for (int i = 0; i < label_size; i++) {
            cout << label[i] << " ";
        }
        cout << endl;
    }

    static Sample* from_mnist_sample(const MnistSample& sample) {
        float label[10] = {0};
        label[sample.label] = 1;
        return new Sample(sample.data, 28*28, label, 10);
    }
    
    private:
    float* data;
    int data_size;
    float* label;
    int label_size;
    
};

#endif