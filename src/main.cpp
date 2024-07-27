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

#include <iostream>
#include <memory>
#include <cstdlib> // For rand() and srand()
#include <ctime>   // For time()

using namespace std;

int main(int argc, char const *argv[]) {

    std::srand(std::time(0));

    InputLayer input(4, 3);

    float first_sample[4] = {11, 12, 13, 14};
    float first_label[3] = {1, 0, 0};

    float second_sample[4] = {21, 22, 23, 24};
    float second_label[3] = {0, 1, 0};

    float third_sample[4] = {31, 32, 33, 34};
    float third_label[3] = {0, 0, 1};

    Sample samples[3] = {
        Sample(first_sample, 4, first_label, 3),
        Sample(second_sample, 4, second_label, 3),
        Sample(third_sample, 4, third_label, 3)
    };

    input.load_train_samples(samples, 3);

    input.get_output()->getData().print();

    NeuralNetwork nn(4);

    nn.add_layer(make_shared<Linear>(4, 100));
    nn.add_layer(make_shared<Linear>(100, 10));
    nn.add_layer(make_shared<ReLU>());

    nn.initialize_layers();
    nn.setup_mini_batch_size(3);

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 1; i++) {
        nn.feed_forward(samples, 3);
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end - start;

    nn.get_output_data().print();
    nn.get_loss()->print();
    std::cout << "Took " << duration.count() << " ms" << std::endl;

    return 0;

    int training_samples_count = 100;
    int test_samples_count = 100;
    float black = 0;
    float white = 1;

    // MNIST
    MnistSample* training_data = mnist_load_samples("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", 0, training_samples_count, black, white);
    MnistSample* test_data = mnist_load_samples("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte", 0, test_samples_count, black, white);

    shared_ptr<Sample> training_samples[training_samples_count];
    shared_ptr<Sample> test_samples[test_samples_count];

    for (int i = 0; i < training_samples_count; i++) {
        training_samples[i] = shared_ptr<Sample>(Sample::from_mnist_sample(training_data[i]));
    }

    for (int i = 0; i < test_samples_count; i++) {
        test_samples[i] = shared_ptr<Sample>(Sample::from_mnist_sample(test_data[i]));
    }

    free(training_data);
    free(test_data);

    for (int i = 0; i < 10; i++) {
        mnist_print_image(training_samples[i]->getData());
        cout << "Label: " << training_samples[i]->index_from_label() << endl;
    }

    return 0;
    return 0;
}
