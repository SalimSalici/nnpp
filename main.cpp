#include <iostream>
#include <memory>

#include "Mat.h"
#include "Node.h"
#include "Layers.h"
#include "NeuralNetwork.h"
#include "mnist_loader.h"

using namespace std;

int main(int argc, char const *argv[]) {

    int training_samples_count = 100;
    int test_samples_count = 100;
    float black = 0;
    float white = 1;

    // MNIST
    MnistSample* training_data = mnist_load_samples("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", 0, training_samples_count, black, white);
    MnistSample* test_data = mnist_load_samples("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte", 0, test_samples_count, black, white);

    for (int i = 0; i < 10; i++) {
        mnist_print_image(training_data[i].data);
        cout << "Label: " << (int)training_data[i].label << endl;
    }
    return 0;

    NeuralNetwork nn;

    nn.add_layer(make_shared<InputLayer>(10, 15));
    // (10 x 15)

    nn.add_layer(make_shared<Linear>(10, 8));
    // (8 x 10)

    nn.add_layer(make_shared<Linear>(8, 3));
    // (3 x 8)

    nn.add_layer(make_shared<Sigmoid>());

    // (8 x 10) * (10 x 15) = (8 x 15)
    // (3 x 8) * (8 x 15) = (3 x 15)

    nn.add_layer(make_shared<Summation>());

    nn.print();

    cout << "\nlol 1\n";

    nn.initialize();

    cout << "\nlol 2\n";

    nn.print();

    nn.feed_forward();
    nn.backprop();

    cout << "\nlol 3\n";

    nn.print();

    cout << "\nlol 4\n";

    nn.get_output_data().print();

    return 0;

    Linear linear(8, 5);
    cout << "\n\n\n";
    linear.print();

    cout << "\n\n\n";
    linear.initialize_xavier();
    linear.print();

    return 0;

    Mat mat1(3, 3);
    Mat mat2(3, 3);
    Mat mat_vec(3, 1);

    float arr[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    float arr2[] = {1, 1, 1, 1, 1, 1, 2, 2, 2};

    float arr_vec[] = {1, 1, 1};

    mat1.copy_from(arr);
    mat1.print();

    mat2.copy_from(arr2);
    mat2.print();

    mat_vec.copy_from(arr_vec);
    mat_vec.print();

    Mat mat3 = mat1 + mat2;
    mat3.print();

    Mat mat4(mat1 + mat2);
    mat4.print();

    Mat::matmul(mat4, mat1, mat2);
    mat4.print();

    mat4 = mat1;
    mat4.print();

    mat4 = mat1 + mat2;
    mat4.print();

    std::cout << "\nresults\n";

    Mat mat5 = Mat::matmul(mat1, mat2);
    mat5.print();

    mat5.zero();
    mat5.print();

    cout << "\n----------------------------\n\n";

    auto a = make_shared<Node>(mat1);
    auto b = make_shared<Node>(mat2);

    auto c = Node::matmul(a, b);
    c->compute();

    a->getData().print();
    b->getData().print();
    c->getData().print();

    float sevens[9] = {7, 7, 7, 19, 19, 19, 7, 7, 7};
    Mat mat7(3, 3);
    mat7.copy_from(sevens);

    auto d = make_shared<Node>(mat7);
    
    auto e_pre = *c - d;

    auto vec = make_shared<Node>(mat_vec);

    auto e = Node::mat_plus_vec(e_pre, vec);

    auto f = Node::sigmoid(e);

    // PowNode g = Node::pow(f, 2);
    auto g = *Node::pow(f, 2) + a;

    // auto k = Node::pow(*f, 2);
    
    auto j = g->sum();

    // j->backprop();
    deque<NodePtr> sorted_nodes;
    j->topo_sort(sorted_nodes);

    Node::forwardPass(sorted_nodes);
    Node::backwardPass(sorted_nodes);

    Node::reset(sorted_nodes);
    Node::backwardPass(sorted_nodes);

    cout << "\nJ data\n";
    j->getData().print();
    cout << "\nG data\n";
    g->getData().print();
    cout << "\nF data\n";
    f->getData().print();
    cout << "\nE data\n";
    e->getData().print();
    cout << "\nVEC data\n";
    vec->getData().print();
    cout << "\nE-pre data\n";
    e_pre->getData().print();
    cout << "\nD data\n";
    d->getData().print();
    cout << "\nC data\n";
    c->getData().print();
    cout << "\nB data\n";
    b->getData().print();
    cout << "\nA data\n";
    a->getData().print();

    cout << "\nJ grad\n";
    j->getGrad().print();
    cout << "\nG grad\n";
    g->getGrad().print();
    cout << "\nF grad\n";
    f->getGrad().print();
    cout << "\nE grad\n";
    e->getGrad().print();
    cout << "\nVEC grad\n";
    vec->getGrad().print();
    cout << "\nE-pre grad\n";
    e_pre->getGrad().print();
    cout << "\nD grad\n";
    d->getGrad().print();
    cout << "\nC grad\n";
    c->getGrad().print();
    cout << "\nB grad\n";
    b->getGrad().print();
    cout << "\nA grad\n";
    a->getGrad().print();

    return 0;
}
