#ifndef NODE_H
#define NODE_H

#include <deque>
#include <iostream>
#include <stdexcept>

#include "Mat.h"
#include "activation_functions.h"

class Node;
class UnaryNode;
class BinaryNode;
class SumNode;
class PlusNode;
class MinusNode;
class MatMulNode;
class ActivationNode;
class SigmoidNode;
class PowNode;
class MatPlusVecNode;
class RSSNode;

using namespace std;
using NodePtr = std::shared_ptr<Node>;

class Node : public std::enable_shared_from_this<Node> {
   public:
    Node(int rows, int cols) : data(rows, cols), grad(rows, cols) {
        grad.zero();
    }

    Node(const Mat& toLoad) : Node(toLoad.getRows(), toLoad.getCols()) {
        loadData(toLoad);
    }

    virtual ~Node() = default;

    virtual void backprop() {
        throw domain_error("Cannot backpropagate on a non-scalar matrix.");
    };

    virtual void backprop(deque<NodePtr> sorted_nodes) {
        throw domain_error("Cannot backpropagate on a non-scalar matrix.");
    };

    virtual void compute() {};

    virtual void back() {
        // cout << "NODE BACK\n"; 
    };

    Mat& getData() {
        return data;
    }

    Mat& getGrad() { return grad; }

    void zero_grad() {
        grad.zero();
    }

    void loadData(const Mat& other) {
        if (other.getRows() != data.getRows() || other.getCols() != data.getCols())
            throw invalid_argument("Cannot load data from a matrix of different size.");
        data.copy_from(other.getData());
    }

    virtual void topo_sort(deque<NodePtr>& sorted_nodes) {
        // cout << "NODE TOPO\n";
        if (permMarker) return;
        permMarker = true;
        sorted_nodes.push_front(shared_from_this());
    }

    void reset_markers() {
        permMarker = false;
        tempMarker = false;
    }

    NodePtr operator+(NodePtr other);
    NodePtr operator-(NodePtr other);
    NodePtr sum();

    static NodePtr matmul(NodePtr a, NodePtr b);
    static NodePtr activation(NodePtr a, float (*act)(float), float (*act_derivative)(float));
    static NodePtr sigmoid(NodePtr a);
    static NodePtr pow(NodePtr a, float pow);
    static NodePtr rss(NodePtr y, NodePtr y_hat);
    static NodePtr mat_plus_vec(NodePtr a, NodePtr b);

    static void forwardPass(deque<NodePtr>& sorted_nodes) {
        for (auto it = sorted_nodes.rbegin(); it != sorted_nodes.rend(); ++it) {
            (*it)->compute();
        }
    }

    static void backwardPass(deque<NodePtr>& sorted_nodes) {
        for (const auto& node : sorted_nodes) {
            node->back();
        }
    }

    static void zero_grad(deque<NodePtr>& sorted_nodes) {
        for (const auto& node : sorted_nodes) {
            node->zero_grad();
        }
    }

    static void reset_markers(deque<NodePtr>& sorted_nodes) {
        for (const auto& node : sorted_nodes) {
            node->reset_markers();
        }
    }

   protected:
    Mat data;
    Mat grad;
    bool tempMarker = false;
    bool permMarker = false;
};

class UnaryNode : public Node {
   public:
    UnaryNode(NodePtr a, int rows, int cols) : Node(rows, cols), a(a) {}

    void topo_sort(deque<NodePtr>& sorted_nodes) override {
        // cout << "UNARY TOPO\n";
        if (permMarker) return;
        tempMarker = true;
        a->topo_sort(sorted_nodes);
        permMarker = true;
        sorted_nodes.push_front(shared_from_this());
    }

   protected:
    NodePtr a;
};

class BinaryNode : public Node {
   public:
    BinaryNode(NodePtr a, NodePtr b, int rows, int cols) : Node(rows, cols), a(a), b(b) {}

    void topo_sort(deque<NodePtr>& sorted_nodes) override {
        // cout << "BINARY TOPO\n";
        if (permMarker) return;
        tempMarker = true;
        a->topo_sort(sorted_nodes);
        b->topo_sort(sorted_nodes);
        permMarker = true;
        sorted_nodes.push_front(shared_from_this());
    }

   protected:
    NodePtr a;
    NodePtr b;
};

class SumNode : public UnaryNode {
   public:
    SumNode(NodePtr a) : UnaryNode(a, 1, 1) {}

    void compute() override {
        data.fill(a->getData().elementsSum());
    }

    void backprop() override {
        deque<NodePtr> sorted_nodes;
        topo_sort(sorted_nodes);

        grad.fill(1);

        for (NodePtr node : sorted_nodes) node->back();
    };

    void backprop(deque<NodePtr> sorted_nodes) override {
        grad.fill(1);
        for (NodePtr node : sorted_nodes) node->back();
    };

    void back() override {
        // cout << "SUM BACK\n";
        a->getGrad().fill(1);
    }
};

class ActivationNode : public UnaryNode {
   public:
    ActivationNode(NodePtr a, float (*act)(float), float (*act_derivative)(float))
        : UnaryNode(a, a->getData().getRows(), a->getData().getCols()) {
            this->act = act;
            this->act_derivative = act_derivative;
        }

    void compute() override {
        Mat::apply(data, a->getData(), act);
    }

    void back() override {
        // cout << "SIGMOID BACK\n";
        Mat prime = Mat::apply(a->getData(), act_derivative);
        Mat::hadamardProduct(prime, prime, grad);
        Mat::plus(a->getGrad(), a->getGrad(), prime);
        // a->getGrad() +=
        //     grad * Mat::apply(a->getData(), act_derivative);
    }

    private:
    float (*act)(float);
    float (*act_derivative)(float);
};

class SigmoidNode : public ActivationNode {
    public:
     SigmoidNode(NodePtr a) : ActivationNode(a, activation_functions::sigmoid, activation_functions::sigmoid_derivative) {}
};

class PowNode : public UnaryNode {
   public:
   PowNode(NodePtr a, float pow)
        : UnaryNode(a, a->getData().getRows(), a->getData().getCols()), pow(pow) {}

    void compute() override {
        Mat::pow(data, a->getData(), pow);
    }

    void back() override {
        // cout << "POW BACK\n";
        Mat::plus(a->getGrad(), a->getGrad(), grad * Mat::pow(Mat::scale(a->getData(), pow), pow - 1));
        // a->getGrad() += grad * Mat::pow(Mat::scale(a->getData(), pow),
        //                                pow - 1);
    }

   private:
    float pow;
};

class PlusNode : public BinaryNode {
   public:
    PlusNode(NodePtr a, NodePtr b) : BinaryNode(a, b, a->getData().getRows(), a->getData().getCols()) {}

    void compute() override {
        Mat::plus(data, a->getData(), b->getData());
    }

    void back() override {
        // cout << "PLUS BACK\n";
        Mat::plus(a->getGrad(), a->getGrad(), grad);
        Mat::plus(b->getGrad(), b->getGrad(), grad);
        // a->getGrad() += grad;
        // b->getGrad() += grad;
    }
};

class MatPlusVecNode : public BinaryNode {
   public:
    MatPlusVecNode(NodePtr a, NodePtr b) : BinaryNode(a, b, a->getData().getRows(), a->getData().getCols()) {}

    void compute() override {
        Mat::mat_plus_vec(data, a->getData(), b->getData());
    }

    void back() override {
        // cout << "PLUS BACK\n";
        Mat::plus(a->getGrad(), a->getGrad(), grad);
        // a->getGrad() += grad;
        Mat::vec_plus_mat(b->getGrad(), b->getGrad(), grad);
    }
};

class MinusNode : public BinaryNode {
   public:
    MinusNode(NodePtr a, NodePtr b) : BinaryNode(a, b, a->getData().getRows(), a->getData().getCols()) {}

    void compute() override {
        Mat::minus(data, a->getData(), b->getData());
    }

    void back() override {
        // cout << "MINUS BACK\n";

        Mat::plus(a->getGrad(), a->getGrad(), grad);
        Mat::minus(b->getGrad(), b->getGrad(), grad);
    }
};

class MatMulNode : public BinaryNode {
   public:
    MatMulNode(NodePtr a, NodePtr b)
        : BinaryNode(a, b, a->getData().getRows(), b->getData().getCols()) {}

    void compute() override {
        Mat::matmul(data, a->getData(), b->getData());
    }

    void back() override {
        // cout << "MATMUL BACK\n";
        b->getData().transpose();
        Mat::plus(a->getGrad(), a->getGrad(), Mat::matmul(grad, b->getData()));
        // a->getGrad() += Mat::matmul(grad, b->getData());
        b->getData().transpose();

        a->getData().transpose();
        Mat::plus(b->getGrad(), b->getGrad(), Mat::matmul(a->getData(), grad));
        // b->getGrad() += Mat::matmul(a->getData(), grad);
        a->getData().transpose();
    }
};

// Residual Sum of Squares loss. Similar to MSE, but without the division by the number of samples (so without the mean)
class RSSNode : public BinaryNode {
   public:

    // y is the true value, y_hat is the predicted value (output of the model)
    // a                    b
    RSSNode(NodePtr y, NodePtr y_hat) : BinaryNode(y, y_hat, 1, 1),
        difference(y->getData().getRows(), y->getData().getCols()),
        pow(y->getData().getRows(), y->getData().getCols()) {}

    void compute() override {
        // b is the predicted value, a is the true value
        if (!compute_flag) return;
        Mat::minus(difference, b->getData(), a->getData());
        Mat::pow(pow, difference, 2);
        data.fill(pow.elementsSum());
        // std::cout << data.getData()[0] << std::endl;
    }

    void back() override {
        // This nodes assumes it's the last in the computational graph

        // cout << "RSS BACK\n";

        // b is the predicted value, a is the true value    
        if (!compute_flag)
            Mat::minus(difference, b->getData(), a->getData());

        Mat::plus(b->getGrad(), b->getGrad(), difference);
        Mat::plus(a->getGrad(), a->getGrad(), Mat::scale(difference, -1));

        // std::cout << "BACCKOOO\n";
    }

    void set_compute_flag(bool flag) {
        compute_flag = flag;
        if (flag == false)
            data.fill(NAN);
    }

    private:
     Mat difference;
     Mat pow;
     bool compute_flag = true;
};

NodePtr Node::operator+(NodePtr other) { return std::make_shared<PlusNode>(shared_from_this(), other); }

NodePtr Node::operator-(NodePtr other) { return std::make_shared<MinusNode>(shared_from_this(), other); }

NodePtr Node::matmul(NodePtr a, NodePtr b) { return std::make_shared<MatMulNode>(a, b); }

NodePtr Node::sum() { return std::make_shared<SumNode>(shared_from_this()); }

NodePtr Node::activation(NodePtr a, float (*act)(float), float (*act_derivative)(float)) {
    return std::make_shared<ActivationNode>(a, act, act_derivative);
}

NodePtr Node::sigmoid(NodePtr a) { return std::make_shared<SigmoidNode>(a); }

NodePtr Node::pow(NodePtr a, float pow) { return std::make_shared<PowNode>(a, pow); }

NodePtr Node::mat_plus_vec(NodePtr a, NodePtr b) { return std::make_shared<MatPlusVecNode>(a, b); }

#endif