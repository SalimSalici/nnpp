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
    Node(int rows, int cols, bool requires_grad) : data(rows, cols), grad(rows, cols), requires_grad(requires_grad) {
        grad.zero();
    }

    Node(const Mat& toLoad, bool requires_grad) : Node(toLoad.getRows(), toLoad.getCols(), requires_grad) {
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

    bool get_requires_grad() {
        return requires_grad;
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
    bool requires_grad;
    bool tempMarker = false;
    bool permMarker = false;
};

class UnaryNode : public Node {
   public:
    UnaryNode(NodePtr a, int rows, int cols, bool requires_grad) : Node(rows, cols, requires_grad), a(a) {}

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
    BinaryNode(NodePtr a, NodePtr b, int rows, int cols, bool requires_grad) : Node(rows, cols, requires_grad), a(a), b(b) {}

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
    SumNode(NodePtr a, bool requires_grad) : UnaryNode(a, 1, 1, requires_grad) {}

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
        if (a->get_requires_grad())
            a->getGrad().fill(1);
    }
};

class ActivationNode : public UnaryNode {
   public:
    ActivationNode(NodePtr a, float (*act)(float), float (*act_derivative)(float), bool requires_grad)
        : UnaryNode(a, a->getData().getRows(), a->getData().getCols(), requires_grad) {
            this->act = act;
            this->act_derivative = act_derivative;
        }

    void compute() override {
        Mat::apply(data, a->getData(), act);
    }

    void back() override {
        // cout << "SIGMOID BACK\n";
        if (a->get_requires_grad()) {
            Mat prime = Mat::apply(a->getData(), act_derivative);
            Mat::hadamardProduct(prime, prime, grad);
            Mat::plus(a->getGrad(), a->getGrad(), prime);
        }
    }

    private:
    float (*act)(float);
    float (*act_derivative)(float);
};

class SigmoidNode : public ActivationNode {
    public:
     SigmoidNode(NodePtr a, bool requires_grad) :
        ActivationNode(a, activation_functions::sigmoid, activation_functions::sigmoid_derivative, requires_grad) {}
};

class PowNode : public UnaryNode {
   public:
   PowNode(NodePtr a, float pow, bool requires_grad)
        : UnaryNode(a, a->getData().getRows(), a->getData().getCols(), requires_grad), pow(pow) {}

    void compute() override {
        Mat::pow(data, a->getData(), pow);
    }

    void back() override {
        // cout << "POW BACK\n";
        if (a->get_requires_grad()) {
            Mat::plus(a->getGrad(), a->getGrad(), grad * Mat::pow(Mat::scale(a->getData(), pow), pow - 1));
        }
    }

   private:
    float pow;
};

class PlusNode : public BinaryNode {
   public:
    PlusNode(NodePtr a, NodePtr b, bool requires_grad)
        : BinaryNode(a, b, a->getData().getRows(), a->getData().getCols(), requires_grad) {}

    void compute() override {
        Mat::plus(data, a->getData(), b->getData());
    }

    void back() override {
        // cout << "PLUS BACK\n";
        if (a->get_requires_grad()) {
            Mat::plus(a->getGrad(), a->getGrad(), grad);
        }
        if (b->get_requires_grad()) {
            Mat::plus(b->getGrad(), b->getGrad(), grad);
        }
    }
};

class MatPlusVecNode : public BinaryNode {
   public:
    MatPlusVecNode(NodePtr a, NodePtr b, bool requires_grad) 
        : BinaryNode(a, b, a->getData().getRows(), a->getData().getCols(), requires_grad) {}

    void compute() override {
        Mat::mat_plus_vec(data, a->getData(), b->getData());
    }

    void back() override {
        // cout << "PLUS BACK\n";
        if (a->get_requires_grad()) {
            Mat::plus(a->getGrad(), a->getGrad(), grad);
        }
        if (b->get_requires_grad()) {
            Mat::vec_plus_mat(b->getGrad(), b->getGrad(), grad);
        }
    }
};

class MinusNode : public BinaryNode {
   public:
    MinusNode(NodePtr a, NodePtr b, bool requires_grad)
        : BinaryNode(a, b, a->getData().getRows(), a->getData().getCols(), requires_grad) {}

    void compute() override {
        Mat::minus(data, a->getData(), b->getData());
    }

    void back() override {
        // cout << "MINUS BACK\n";
        if (a->get_requires_grad()) {
            Mat::plus(a->getGrad(), a->getGrad(), grad);
        }
        if (b->get_requires_grad()) {
            Mat::minus(b->getGrad(), b->getGrad(), grad);
        }
    }
};

class MatMulNode : public BinaryNode {
   public:
    MatMulNode(NodePtr a, NodePtr b, bool requires_grad)
        : BinaryNode(a, b, a->getData().getRows(), b->getData().getCols(), requires_grad) {}

    void compute() override {
        Mat::matmul(data, a->getData(), b->getData());
    }

    void back() override {
        // cout << "MATMUL BACK\n";
        if (a->get_requires_grad()) {
            b->getData().transpose();
            Mat::plus(a->getGrad(), a->getGrad(), Mat::matmul(grad, b->getData()));
            b->getData().transpose();
        }
        if (b->get_requires_grad()) {
            a->getData().transpose();
            Mat::plus(b->getGrad(), b->getGrad(), Mat::matmul(a->getData(), grad));
            a->getData().transpose();
        }
    }
};

// Residual Sum of Squares loss. Similar to MSE, but without the division by the number of samples (so without the mean)
class RSSNode : public BinaryNode {
   public:

    // y is the true value, y_hat is the predicted value (output of the model)
    // a                    b
    RSSNode(NodePtr y, NodePtr y_hat, bool requires_grad) : BinaryNode(y, y_hat, 1, 1, requires_grad),
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

        if (b->get_requires_grad()) {
            Mat::plus(b->getGrad(), b->getGrad(), difference);
        }
        if (a->get_requires_grad()) {
            Mat::plus(a->getGrad(), a->getGrad(), Mat::scale(difference, -1));
        }

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

NodePtr Node::operator+(NodePtr other) { return std::make_shared<PlusNode>(shared_from_this(), other, true); }

NodePtr Node::operator-(NodePtr other) { return std::make_shared<MinusNode>(shared_from_this(), other, true); }

NodePtr Node::matmul(NodePtr a, NodePtr b) { return std::make_shared<MatMulNode>(a, b, true); }

NodePtr Node::sum() { return std::make_shared<SumNode>(shared_from_this(), true); }

NodePtr Node::activation(NodePtr a, float (*act)(float), float (*act_derivative)(float)) {
    return std::make_shared<ActivationNode>(a, act, act_derivative, true);
}

NodePtr Node::sigmoid(NodePtr a) { return std::make_shared<SigmoidNode>(a, true); }

NodePtr Node::pow(NodePtr a, float pow) { return std::make_shared<PowNode>(a, pow, true); }

NodePtr Node::mat_plus_vec(NodePtr a, NodePtr b) { return std::make_shared<MatPlusVecNode>(a, b, true); }

#endif