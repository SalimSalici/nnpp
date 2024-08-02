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

    int training_samples_count = 100;
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

    // for (int i = 0; i < 10; i++) {
    //     mnist_print_image(training_samples[i].getData());
    //     cout << "Label: " << training_samples[i].index_from_label() << endl;
    // }

    // NeuralNetwork nn(28*28);

    // nn.add_layer(make_shared<Linear>(28*28, 30));
    // nn.add_layer(make_shared<Sigmoid>());
    // nn.add_layer(make_shared<Linear>(30, 10));
    // nn.add_layer(make_shared<Sigmoid>());
    // // nn.set_loss_type(LossType::CCE);
    // nn.set_loss_type(LossType::MSE);

    float lr = 0.1;
    int epochs = 30;

    int minibatch_size = 10;

    NeuralNetwork nn(28*28);

    shared_ptr<Layer> conv2d_im2row_1 = make_shared<Conv2d_im2row>(minibatch_size, 28, 28, 1, 20, 5, 5, 1, 1, 0, 0);
    shared_ptr<Layer> conv2d_im2row_2 = make_shared<Conv2d_im2row>(minibatch_size, 24, 24, 20, 40, 5, 5, 1, 1, 0, 0);
    shared_ptr<Layer> linear_1 = make_shared<Linear>(20*20*40, 100);
    shared_ptr<Layer> linear_2 = make_shared<Linear>(100, 10);

    nn.add_layer(conv2d_im2row_1);
    // nn.add_layer(make_shared<Conv2d_im2row>(minibatch_size, 28, 28, 1, 20, 5, 5, 1, 1, 0, 0));
    nn.add_layer(make_shared<ReLU>());
    // nn.add_layer(make_shared<Conv2d_im2row>(minibatch_size, 24, 24, 20, 40, 5, 5, 1, 1, 0, 0));
    nn.add_layer(conv2d_im2row_2);
    nn.add_layer(make_shared<ReLU>());
    nn.add_layer(linear_1);
    // nn.add_layer(make_shared<Linear>(20*20*40, 100));
    nn.add_layer(make_shared<ReLU>());
    nn.add_layer(linear_2);
    // nn.add_layer(make_shared<Linear>(100, 10));
    // nn.add_layer(make_shared<Sigmoid>());
    nn.set_loss_type(LossType::CCE);

    nn.initialize_layers();


    
    // nn.sgd(training_samples, training_samples_count, lr, epochs, minibatch_size, test_samples, test_samples_count);
    nn.sgd(training_samples, training_samples_count, lr, epochs, minibatch_size, training_samples, training_samples_count);

    cin.get();

    // nn.freeze_params();
    linear_1->freeze_params();
    linear_2->freeze_params();
    conv2d_im2row_1->initialize();
    conv2d_im2row_2->initialize();

    nn.sgd(training_samples, training_samples_count, lr, epochs, minibatch_size, training_samples, training_samples_count);

    return 0;
}
