#ifndef NODE_H
#define NODE_H

#include <deque>
#include <iostream>
#include <stdexcept>

#include "Mat.h"
#include "activations.h"

class Node;
class UnaryNode;
class BinaryNode;
class SumNode;
class PlusNode;
class MinusNode;
class MatMulNode;
class SigmoidNode;
class PowNode;

using namespace std;
using NodePtr = std::shared_ptr<Node>;

class Node : public std::enable_shared_from_this<Node> {
   public:
    Node(const Mat& data) : data(data), grad(data.getRows(), data.getCols()) {
        grad.zero();
    }
    virtual ~Node() = default;

    virtual void backprop() {
        throw domain_error("Cannot backpropagate on a non-scalar matrix.");
    };

    virtual void back() { cout << "NODE BACK\n"; };
    const Mat& getData() const { return data; }
    Mat& getGrad() { return grad; }

    virtual void topo_sort(deque<NodePtr>& sorted_nodes) {
        cout << "NODE TOPO\n";
        if (permMarker) return;
        permMarker = true;
        sorted_nodes.push_front(shared_from_this());
    }

    NodePtr operator+(NodePtr other);
    NodePtr operator-(NodePtr other);
    NodePtr sum();

    static NodePtr matmul(NodePtr a, NodePtr b);
    static NodePtr sigmoid(NodePtr a);
    static NodePtr pow(NodePtr a, float pow);

   protected:
    Mat data;
    Mat grad;
    bool tempMarker = false;
    bool permMarker = false;
};

class UnaryNode : public Node {
   public:
    UnaryNode(NodePtr a, const Mat& data) : Node(data), a(a) {}

    void topo_sort(deque<NodePtr>& sorted_nodes) override {
        cout << "UNARY TOPO\n";
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
    BinaryNode(NodePtr a, NodePtr b, const Mat& data) : Node(data), a(a), b(b) {}

    void topo_sort(deque<NodePtr>& sorted_nodes) override {
        cout << "BINARY TOPO\n";
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
    SumNode(NodePtr a) : UnaryNode(a, Mat(1, 1)) {
        data.fill(a->getData().elementsSum());
    }

    void backprop() override {
        deque<NodePtr> sorted_nodes;
        topo_sort(sorted_nodes);

        grad.fill(1);

        for (NodePtr node : sorted_nodes) node->back();
    };

    void back() override {
        cout << "SUM BACK\n";
        a->getGrad().fill(1);
    }
};

class SigmoidNode : public UnaryNode {
   public:
    SigmoidNode(NodePtr a)
        : UnaryNode(a, Mat::apply(a->getData(), activations::sigmoid)) {}

    void back() override {
        cout << "SIGMOID BACK\n";
        a->getGrad() +=
            grad * Mat::apply(a->getData(), activations::sigmoid_derivative);
    }
};

class PowNode : public UnaryNode {
   public:
    PowNode(NodePtr a, float pow)
        : UnaryNode(a, Mat::pow(a->getData(), pow)), pow(pow) {}

    void back() override {
        cout << "POW BACK\n";
        a->getGrad() += grad * Mat::pow(Mat::scale(a->getData(), pow),
                                       pow - 1);
    }

   private:
    float pow;
};

class PlusNode : public BinaryNode {
   public:
    PlusNode(NodePtr a, NodePtr b) : BinaryNode(a, b, a->getData() + b->getData()) {}

    void back() override {
        cout << "PLUS BACK\n";
        a->getGrad() += grad;
        b->getGrad() += grad;
    }
};

class MinusNode : public BinaryNode {
   public:
    MinusNode(NodePtr a, NodePtr b) : BinaryNode(a, b, a->getData() - b->getData()) {}

    void back() override {
        cout << "MINUS BACK\n";
        a->getGrad() += grad;
        b->getGrad() -= grad;
    }
};

class MatMulNode : public BinaryNode {
   public:
    MatMulNode(NodePtr a, NodePtr b)
        : BinaryNode(a, b, Mat::matmul(a->getData(), b->getData())) {}

    void back() override {
        cout << "MATMUL BACK\n";
        a->getGrad() += Mat::matmul(grad, false, b->getData(), true);
        b->getGrad() += Mat::matmul(a->getData(), true, grad, false);
    }
};

NodePtr Node::operator+(NodePtr other) { return std::make_shared<PlusNode>(shared_from_this(), other); }

NodePtr Node::operator-(NodePtr other) { return std::make_shared<MinusNode>(shared_from_this(), other); }

NodePtr Node::matmul(NodePtr a, NodePtr b) { return std::make_shared<MatMulNode>(a, b); }

NodePtr Node::sum() { return std::make_shared<SumNode>(shared_from_this()); }

NodePtr Node::sigmoid(NodePtr a) { return std::make_shared<SigmoidNode>(a); }

NodePtr Node::pow(NodePtr a, float pow) { return std::make_shared<PowNode>(a, pow); }

#endif