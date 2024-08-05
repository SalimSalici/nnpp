#include <iostream>
#include <memory>
#include <cstdlib> // For rand() and srand()
#include <ctime>   // For time()
#include <chrono> // For timing

#include "Mat.h"
#include "Node.h"
#include "Layers.h"
#include "NeuralNetwork.h"
#include "mnist_loader.h"
#include "Sample.h"
#include "utils.h"

using namespace std;

int main(int argc, char const *argv[]) {

    std::srand(std::time(0));

    int training_samples_count = 600;
    int test_samples_count = 1000;
    float black = 0;
    float white = 1;


    // MNIST
    MnistSample* training_data = mnist_load_samples("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", 0, training_samples_count, black, white);
    MnistSample* test_data = mnist_load_samples("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte", 0, test_samples_count, black, white);

    Sample* training_samples[training_samples_count];
    Sample* test_samples[test_samples_count];

    for (int i = 0; i < training_samples_count; i++) {
        training_samples[i] = Sample::from_mnist_sample(training_data[i]);
    }

    for (int i = 0; i < test_samples_count; i++) {
        test_samples[i] = Sample::from_mnist_sample(test_data[i]);
    }

    free(training_data);
    free(test_data);

    float lr = 0.1;
    int epochs = 30;


    int minibatch_size = 10;

    NeuralNetwork nn(28*28);

    int c_out_1 = 20;
    int c_out_2 = 40;

    shared_ptr<Conv2d_mec> conv_1 = make_shared<Conv2d_mec>(minibatch_size, 28, 28, 1, c_out_1, 5, 5, 1, 1, 0, 0);
    shared_ptr<Conv2d_mec> conv_2 = make_shared<Conv2d_mec>(minibatch_size, 12, 12, c_out_1, c_out_2, 5, 5, 1, 1, 0, 0);

    shared_ptr<Linear> linear_1 = make_shared<Linear>(4*4*c_out_2, 100);
    shared_ptr<Linear> linear_2 = make_shared<Linear>(100, 10);

    nn.add_layer(conv_1);
    nn.add_layer(make_shared<ReLU>());
    nn.add_layer(make_shared<Maxpool_hnwc_to_nhwc>(minibatch_size, 24, 24, c_out_1, 2, 2, 2, 2));

    nn.add_layer(conv_2);
    nn.add_layer(make_shared<ReLU>());
    nn.add_layer(make_shared<Maxpool_hnwc_to_nhwc>(minibatch_size, 8, 8, c_out_2, 2, 2, 2, 2));

    nn.add_layer(linear_1);
    nn.add_layer(make_shared<ReLU>());
    nn.add_layer(linear_2);
    nn.set_loss_type(LossType::CCE);

    nn.initialize_layers();

    // nn.sgd(training_samples, training_samples_count, lr, epochs, minibatch_size, test_samples, test_samples_count);
    nn.sgd(training_samples, training_samples_count, lr, epochs, minibatch_size, training_samples, training_samples_count);


    cin.get();
    linear_1->freeze_params();
    linear_2->freeze_params();

    conv_1->initialize();
    conv_2->initialize();

    // conv_1->freeze_params();
    // conv_2->freeze_params();

    nn.sgd(training_samples, training_samples_count, lr, epochs, minibatch_size, training_samples, training_samples_count);

    return 0;
}
