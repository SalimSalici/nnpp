#ifndef NODE_H
#define NODE_H

#define LOG_OPERATIONS 0

#include <deque>
#include <iostream>
#include <vector>
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
class DropoutNode;
class HadamardProductNode;
class ActivationNode;
class ReLUNode;
class SigmoidNode;
class TanhNode;
class PowNode;
class MatPlusVecNode;
class MatPlusRowVecNode;
class RSSNode;
class TransposeNode;
class Im2rowNode;
class ReshapeNode;
class MecLowerNode;
class MecLowerSeparatedNode;
class MecConvNode;
class Maxpool_hnwc_to_nhwc_Node;

using namespace std;
using NodePtr = std::shared_ptr<Node>;

class Node : public std::enable_shared_from_this<Node> {
   public:
    Node(int rows, int cols, bool requires_grad) : data(rows, cols), grad(rows, cols), requires_grad(requires_grad) {
        // if (requires_grad)
        //     grad.zero();
    }

    Node(const Mat& toLoad, bool requires_grad) : Node(toLoad.getRows(), toLoad.getCols(), requires_grad) {
        loadData(toLoad);
    }

    void print_data() {
        cout << "Name: " << typeid(*this).name() << endl;
        // data.print_ascii_greyscale();
        data.print();
        cout << endl;
    }

    virtual ~Node() = default;

    virtual void backprop() {
        throw domain_error("Cannot backpropagate on a non-scalar matrix.");
    };

    virtual void backprop(deque<NodePtr> sorted_nodes) {
        throw domain_error("Cannot backpropagate on a non-scalar matrix.");
    };

    virtual void compute() {
        #if LOG_OPERATIONS
        std::cout << "Node compute" << std::endl;
        #endif
    };

    virtual void back() {
        #if LOG_OPERATIONS
        std::cout << "Node back. Rows: " << data.getRows() << ", Cols: " << data.getCols() << std::endl; 
        #endif
    };

    virtual Mat& getData() {
        return data;
    }

    Mat& getGrad() { return grad; }

    void zero_grad() {
        if (!requires_grad) return;
        grad.zero();
    }

    void loadData(const Mat& other) {
        if (other.getRows() != data.getRows() || other.getCols() != data.getCols())
            throw invalid_argument("Cannot load data from a matrix of different size.");
        data.copy_from(other.getData());
    }

    virtual void topo_sort(deque<NodePtr>& sorted_nodes) {
        #if LOG_OPERATIONS
        std::cout << "Node topo_sort" << std::endl;
        #endif
        if (permMarker) return;
        permMarker = true;
        sorted_nodes.push_front(shared_from_this());
    }

    void reset_markers() {
        permMarker = false;
        tempMarker = false;
    }

    virtual bool get_requires_grad() {
        return requires_grad;
    }

    bool get_enabled() {
        return enabled;
    }

    virtual void set_enabled(bool enabled) {
        enabled = enabled;
    }

    virtual int getRows() {
        return data.getRows();
    }

    virtual int getCols() {
        return data.getCols();
    }

    NodePtr operator+(NodePtr other);
    NodePtr operator-(NodePtr other);
    NodePtr sum();

    static NodePtr plus(NodePtr a, NodePtr b);
    static NodePtr matmul(NodePtr a, NodePtr b);
    static NodePtr activation(NodePtr a, float (*act)(float), float (*act_derivative)(float));
    static NodePtr relu(NodePtr a);
    static NodePtr sigmoid(NodePtr a);
    static NodePtr tanh(NodePtr a);
    static NodePtr pow(NodePtr a, float pow);
    static NodePtr rss(NodePtr y, NodePtr y_hat);
    static NodePtr mat_plus_vec(NodePtr a, NodePtr b);
    static NodePtr mat_plus_row_vec(NodePtr a, NodePtr b);
    static NodePtr dropout(int rows, int cols, float dropout_rate, bool requires_grad);
    static NodePtr hadamard_product(NodePtr a, NodePtr b);
    static NodePtr transpose(NodePtr a);
    static NodePtr im2row(NodePtr a, int n, int h, int w, int c,
        int k_h, int k_w, int s_h, int s_w, int p_h, int p_w, bool requires_grad = true);
    static NodePtr reshape(NodePtr a, int reshaped_rows, int reshaped_cols, bool requires_grad = true);
    static NodePtr mec_lower(NodePtr a, int n, int h, int w, int c,
        int k_h, int k_w, int s_h, int s_w, int p_h, int p_w, bool requires_grad = true);
    static NodePtr mec_lower_separated(std::vector<shared_ptr<Mat>>& inputs, int n, int h, int w, int c, int k_h, int k_w, int s_h, int s_w, bool requires_grad = true);
    static NodePtr mec_conv_mm(NodePtr lowered, NodePtr kernels, int n, int h, int w, int c,
                int k_h, int k_w, int s_h, int s_w, bool requires_grad = true);
    static NodePtr maxpool_hnwc_to_nhwc(NodePtr a, int n, int h, int w, int c, int k_h, int k_w, int s_h, int s_w, bool requires_grad = true);

    static void forwardPass(deque<NodePtr>& sorted_nodes) {
        for (auto it = sorted_nodes.rbegin(); it != sorted_nodes.rend(); ++it) {
            NodePtr node = *it;
            if (!node->get_enabled()) return;
            node->compute();
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
    bool enabled = true;
    bool tempMarker = false;
    bool permMarker = false;
};

class UnaryNode : public Node {
   public:
    UnaryNode(NodePtr a, int rows, int cols, bool requires_grad) : Node(rows, cols, requires_grad), a(a) {}

    void topo_sort(deque<NodePtr>& sorted_nodes) override {
        #if LOG_OPERATIONS
        std::cout << "UnaryNode topo_sort" << std::endl;
        #endif
        if (permMarker) return;
        tempMarker = true;
        a->topo_sort(sorted_nodes);
        permMarker = true;
        sorted_nodes.push_front(shared_from_this());
    }

    Mat& getData() override {
        if (get_enabled()) return data;
        return a->getData();
    }

   protected:
    NodePtr a;
};

class BinaryNode : public Node {
   public:
    BinaryNode(NodePtr a, NodePtr b, int rows, int cols, bool requires_grad) : Node(rows, cols, requires_grad), a(a), b(b) {}

    void topo_sort(deque<NodePtr>& sorted_nodes) override {
        #if LOG_OPERATIONS
        std::cout << "BinaryNode topo_sort" << std::endl;
        #endif
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
        #if LOG_OPERATIONS
        std::cout << "SumNode compute" << std::endl;
        #endif
        data.fill(a->getData().elementsSum());
    }

    void backprop() override {
        #if LOG_OPERATIONS
        std::cout << "SumNode backprop" << std::endl;
        #endif
        deque<NodePtr> sorted_nodes;
        topo_sort(sorted_nodes);

        grad.fill(1);

        for (NodePtr node : sorted_nodes) node->back();
    };

    void backprop(deque<NodePtr> sorted_nodes) override {
        #if LOG_OPERATIONS
        std::cout << "SumNode backprop with sorted nodes" << std::endl;
        #endif
        grad.fill(1);
        for (NodePtr node : sorted_nodes) node->back();
    };

    void back() override {
        #if LOG_OPERATIONS
        std::cout << "SumNode back" << std::endl;
        #endif
        if (a->get_requires_grad())
            a->getGrad().fill(1);
    }
};

class ActivationNode : public UnaryNode {
   public:
    ActivationNode(NodePtr a, float (*act)(float), float (*act_derivative)(float), bool requires_grad) : 
        UnaryNode(a, a->getRows(), a->getCols(), requires_grad),
        prime(a->getRows(), a->getCols())
    {
        this->act = act;
        this->act_derivative = act_derivative;
    }

    void compute() override {
        #if LOG_OPERATIONS
        std::cout << "ActivationNode compute" << std::endl;
        #endif
        Mat::apply(data, a->getData(), act);
    }

    virtual void back() override {
        #if LOG_OPERATIONS
        std::cout << "ActivationNode back" << std::endl;
        #endif
        if (a->get_requires_grad()) {
            Mat::apply(prime, a->getData(), act_derivative);
            // Mat::hadamardProduct(prime, prime, grad);
            // Mat::plus(a->getGrad(), a->getGrad(), prime);
            Mat::hadamardProduct_keep_res(a->getGrad(), grad, prime, 1);
        }
    }

    protected:
    float (*act)(float);
    float (*act_derivative)(float);
    Mat prime;
};

class ReLUNode : public UnaryNode {
    public:
     ReLUNode(NodePtr a, bool requires_grad) :
        UnaryNode(a, a->getRows(), a->getCols(), requires_grad), prime(a->getRows(), a->getCols()) {}

    void compute() override {
        #if LOG_OPERATIONS
        std::cout << "ReLUNode compute" << std::endl;
        #endif
        Mat::apply_relu(data, a->getData());
    }

    void back() override {
        #if LOG_OPERATIONS
        std::cout << "SigmoidNode back" << std::endl;
        #endif
        if (a->get_requires_grad()) {
            Mat::apply_relu_derivative(prime, a->getData());
            // Mat::hadamardProduct(prime, prime, grad);
            // Mat::plus(a->getGrad(), a->getGrad(), prime);
            Mat::hadamardProduct_keep_res(a->getGrad(), grad, prime, 1);
        }
    }

    protected:

    Mat prime;
};

class SigmoidNode : public ActivationNode {
    public:
     SigmoidNode(NodePtr a, bool requires_grad) :
        ActivationNode(a, activation_functions::sigmoid, activation_functions::sigmoid_derivative, requires_grad) {}

    void back() override {
        #if LOG_OPERATIONS
        std::cout << "SigmoidNode back" << std::endl;
        #endif
        if (a->get_requires_grad()) {
            Mat::apply(prime, data, activation_functions::sigmoid_derivative_with_sig);
            // Mat::hadamardProduct(prime, prime, grad);
            // Mat::plus(a->getGrad(), a->getGrad(), prime);
            Mat::hadamardProduct_keep_res(a->getGrad(), grad, prime, 1);
        }
    }
};

class TanhNode : public ActivationNode {
    public:
     TanhNode(NodePtr a, bool requires_grad) :
        ActivationNode(a, activation_functions::tanh, activation_functions::tanh_derivative, requires_grad) {}

    void back() override {
        #if LOG_OPERATIONS
        std::cout << "TanhNode back" << std::endl;
        #endif
        if (a->get_requires_grad()) {
            Mat::apply(prime, data, activation_functions::tanh_derivative_with_tanh);
            // Mat::hadamardProduct(prime, prime, grad);
            // Mat::plus(a->getGrad(), a->getGrad(), prime);
            Mat::hadamardProduct_keep_res(a->getGrad(), grad, prime, 1);
        }
    }
};

// Element-wise power
class PowNode : public UnaryNode {
   public:
   PowNode(NodePtr a, float pow, bool requires_grad)
        : UnaryNode(a, a->getRows(), a->getCols(), requires_grad), pow(pow) {}

    void compute() override {
        #if LOG_OPERATIONS
        std::cout << "PowNode compute" << std::endl;
        #endif
        Mat::pow(data, a->getData(), pow);
    }

    void back() override {
        #if LOG_OPERATIONS
        std::cout << "PowNode back" << std::endl;
        #endif
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
        : BinaryNode(a, b, a->getRows(), a->getCols(), requires_grad) {}

    void compute() override {
        #if LOG_OPERATIONS
        std::cout << "PlusNode compute" << std::endl;
        #endif
        Mat::plus(data, a->getData(), b->getData());
    }

    void back() override {
        #if LOG_OPERATIONS
        std::cout << "PlusNode back" << std::endl;
        #endif
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
        : BinaryNode(a, b, a->getRows(), a->getCols(), requires_grad) {}

    void compute() override {
        #if LOG_OPERATIONS
        std::cout << "MatPlusVecNode compute" << std::endl;
        #endif
        Mat::mat_plus_vec(data, a->getData(), b->getData());
    }

    void back() override {
        #if LOG_OPERATIONS
        std::cout << "MatPlusVecNode back" << std::endl;
        #endif
        if (a->get_requires_grad()) {
            Mat::plus(a->getGrad(), a->getGrad(), grad);
        }
        if (b->get_requires_grad()) {
            Mat::vec_plus_mat(b->getGrad(), b->getGrad(), grad);
        }
    }
};

class MatPlusRowVecNode : public BinaryNode {
   public:
    MatPlusRowVecNode(NodePtr a, NodePtr b, bool requires_grad) 
        : BinaryNode(a, b, a->getRows(), a->getCols(), requires_grad) {}

    void compute() override {
        #if LOG_OPERATIONS
        std::cout << "MatPlusRowVecNode compute" << std::endl;
        std::cout << "a: " << a->getData().getRows() << "x" << a->getData().getCols() << std::endl;
        std::cout << "b: " << b->getData().getRows() << "x" << b->getData().getCols() << std::endl << std::endl;
        #endif

        Mat::mat_plus_row_vec(data, a->getData(), b->getData());
    }

    void back() override {
        #if LOG_OPERATIONS
        std::cout << "MatPlusRowVecNode back" << std::endl;
        #endif
        if (a->get_requires_grad()) {
            Mat::plus(a->getGrad(), a->getGrad(), grad);
        }
        if (b->get_requires_grad()) {
            Mat::row_vec_plus_mat(b->getGrad(), b->getGrad(), grad);
        }
    }
};

class MinusNode : public BinaryNode {
   public:
    MinusNode(NodePtr a, NodePtr b, bool requires_grad)
        : BinaryNode(a, b, a->getRows(), a->getCols(), requires_grad) {}

    void compute() override {
        #if LOG_OPERATIONS
        std::cout << "MinusNode compute" << std::endl;
        #endif
        Mat::minus(data, a->getData(), b->getData());
    }

    void back() override {
        #if LOG_OPERATIONS
        std::cout << "MinusNode back" << std::endl;
        #endif
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
        : BinaryNode(a, b, a->getRows(), b->getCols(), requires_grad) {}

    void compute() override {
        #if LOG_OPERATIONS
        std::cout << "MatMulNode compute" << std::endl;
        #endif
        Mat::matmul(data, a->getData(), b->getData());
    }

    void back() override {
        #if LOG_OPERATIONS
        std::cout << "MatMulNode back" << std::endl;
        #endif
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

// Inverse dropout (neuron scaling is applied during training, not during inference)
class DropoutNode : public Node {
    public:

    DropoutNode(int rows, int cols, float dropout_rate, bool requires_grad) :
        Node(rows, cols, requires_grad), p(dropout_rate) {}

    void compute() override {
        #if LOG_OPERATIONS
        std::cout << "DropoutNode compute" << std::endl;
        #endif
        if (p == 1) {
            throw invalid_argument("Cannot apply dropout with rate 1.");
        }
        data.fill_rand_rate(0, 1.0f / (1.0f - p), p);
    }

    float get_dropout_rate() {
        return p;
    }

    void back() {
        #if LOG_OPERATIONS
        std::cout << "DropoutNode back" << std::endl;
        #endif
    };
        
    private:

    float p;
};

class HadamardProductNode : public BinaryNode {
   public:
    HadamardProductNode(NodePtr a, NodePtr b, bool requires_grad)
        : BinaryNode(a, b, a->getRows(), a->getCols(), requires_grad) {}

    void compute() override {
        #if LOG_OPERATIONS
        std::cout << "HadamardProductNode compute" << std::endl;
        #endif
        Mat::hadamardProduct(data, a->getData(), b->getData());
    }

    void back() override {
        #if LOG_OPERATIONS
        std::cout << "HadamardProductNode back" << std::endl;
        #endif
        if (a->get_requires_grad())
            Mat::hadamardProduct_keep_res(a->getGrad(), grad, b->getData(), 1);

        if (b->get_requires_grad())
            Mat::hadamardProduct_keep_res(b->getGrad(), grad, a->getData(), 1);
    }
};

// Residual Sum of Squares loss. Similar to MSE, but without the division by the number of samples (so without the mean)
class RSSNode : public BinaryNode {
   public:

    // y is the true value, y_hat is the predicted value (output of the model)
    // a                    b
    RSSNode(NodePtr y, NodePtr y_hat, bool requires_grad) : BinaryNode(y, y_hat, 1, 1, requires_grad),
        difference(y->getRows(), y->getCols()),
        pow(y->getRows(), y->getCols()) {}

    void compute() override {
        #if LOG_OPERATIONS
        std::cout << "RSSNode compute" << std::endl;
        #endif
        // b is the predicted value, a is the true value
        if (!enabled) return;
        Mat::minus(difference, b->getData(), a->getData());
        Mat::pow(pow, difference, 2);
        data.fill(pow.elementsSum());
    }

    void back() override {
        #if LOG_OPERATIONS
        std::cout << "RSSNode back" << std::endl;
        #endif
        // This nodes assumes it's the last in the computational graph

        // b is the predicted value, a is the true value    
        if (!enabled)
            Mat::minus(difference, b->getData(), a->getData());

        if (b->get_requires_grad()) {
            Mat::plus(b->getGrad(), b->getGrad(), difference);
        }
        if (a->get_requires_grad()) {
            Mat::plus(a->getGrad(), a->getGrad(), Mat::scale(difference, -1));
        }
    }

    void set_enabled(bool flag) override {
        enabled = flag;
        if (flag == false)
            data.fill(NAN);
    }

    private:
     Mat difference;
     Mat pow;
};

// Categorical Cross Entropy Node
class CCENode : public BinaryNode {
   public:
    // "y" is the true value (one-hot encoded), "logits" is the output of the last layer w/o activation function
    //  a                                       b
    CCENode(NodePtr y, NodePtr logits, bool requires_grad) :
        BinaryNode(y, logits, 1, 1, requires_grad),
        predictions(y->getRows(), y->getCols()),
        log_predictions(y->getRows(), y->getCols()) {}

    void compute() override {
        #if LOG_OPERATIONS
        std::cout << "CCENode compute" << std::endl;
        #endif
        if (!enabled) return;

        // Apply softmax to logits to get predictions
        Mat::softmax(predictions, b->getData());

        // Compute log of predictions
        Mat::apply_log(log_predictions, predictions);

        // Compute loss: -sum(y * log(y_hat))
        Mat::hadamardProduct(log_predictions, a->getData(), log_predictions);
        data.fill(-log_predictions.elementsSum());
    }

    void back() override {
        #if LOG_OPERATIONS
        std::cout << "CCENode back" << std::endl;
        #endif
        // b is the predicted value, a is the true value    
        if (!enabled && b->get_requires_grad())
            Mat::softmax(predictions, b->getData());

        // Gradient of CCE with respect to softmax output
        // grad = predictions - y
        if (b->get_requires_grad()) {
            Mat::minus(b->getGrad(), predictions, a->getData());
        }

        // We don't compute gradient w.r.t. y (a) as it's usually the ground truth
    }

    void set_enabled(bool flag) override {
        enabled = flag;
        if (flag == false)
            data.fill(NAN);
    }

   private:
    Mat predictions;
    Mat log_predictions;
};


// 
class BCENode : public BinaryNode {
   public:

    // "y" is the true value, "logits" is the output of the last layer w/o activation function
    //  a                      b
    BCENode(NodePtr y, NodePtr logits, bool requires_grad) :
        BinaryNode(y, logits, 1, 1, requires_grad),
        intermediate1(y->getRows(), y->getCols()), 
        intermediate2(y->getRows(), y->getCols()), 
        intermediate3(y->getRows(), y->getCols()), 
        predictions(y->getRows(), y->getCols()) {}

    void compute() override {
        #if LOG_OPERATIONS
        std::cout << "BCENode compute" << std::endl;
        #endif
        // b is the predicted value, a is the true value
        if (!enabled) return;

        // "predictions" is y_hat, the output of the last layer after applying the sigmoid activation function
        Mat::apply(predictions, b->getData(), activation_functions::sigmoid);

        // log(y_hat)
        Mat::apply_log(intermediate1, predictions);

        // y * log(y_hat) 
        Mat::hadamardProduct(intermediate1, a->getData(), intermediate1);

        // 1 - y_hat
        Mat::mat_plus_scalar(intermediate2, predictions, 1, -1);

        // log(1 - y_hat)
        Mat::apply_log(intermediate2, intermediate2);

        // 1 - y
        Mat::mat_plus_scalar(intermediate3, a->getData(), 1, -1);

        // (1 - y) * log(1 - y_hat)
        Mat::hadamardProduct(intermediate2, intermediate3, intermediate2);

        // y * log(y_hat)  + (1 - y) * log(1 - y_hat)
        Mat::plus(intermediate1, intermediate1, intermediate2);
        
        data.fill(-intermediate1.elementsSum());
    }

    void back() override {
        #if LOG_OPERATIONS
        std::cout << "BCENode back" << std::endl;
        #endif
        // This nodes assumes it's the last in the computational graph

        // b is the predicted value, a is the true value    
        if (!enabled && b->get_requires_grad())
            Mat::apply(predictions, b->getData(), activation_functions::sigmoid);

        if (b->get_requires_grad())
            Mat::minus(b->getGrad(), predictions, a->getData());

        // We are assuming the labels don't need gradient, so we don't backpropagate to them
        // If we did, I think it should be -ln(y_hat/(1-y_hat)) = ln((1-y_hat)/y_hat)
    }

    void set_enabled(bool flag) override {
        enabled = flag;
        if (flag == false)
            data.fill(NAN);
    }

    private:
     Mat intermediate1;
     Mat intermediate2;
     Mat intermediate3;
     Mat predictions;
};

class TransposeNode : public UnaryNode {
    public:

    TransposeNode(NodePtr a, bool requires_grad) : UnaryNode(a, a->getRows(), a->getCols(), requires_grad) {
        data.view(a->getData());
        grad.view(a->getGrad());

        data.transpose();
        grad.transpose();
    }

    bool get_requires_grad() override {
        return a->get_requires_grad();
    }

    int getRows() override {
        return a->getCols();
    }

    int getCols() override {
        return a->getRows();
    }
};

class ReshapeNode : public UnaryNode {
    public:

    ReshapeNode(NodePtr a, int reshaped_rows, int reshaped_cols, bool requires_grad)
        : UnaryNode(
            a, 
            calculateRows(a, reshaped_rows, reshaped_cols),         
            calculateCols(a, reshaped_rows, reshaped_cols), 
            requires_grad
        ) {
            if (reshaped_rows <= 0 && reshaped_cols <= 0) {
                throw invalid_argument("At least one of the reshaped dimensions must be positive.");
            }

            data.view(a->getData());
            grad.view(a->getGrad());
        }

    void compute() override {
        #if LOG_OPERATIONS
        std::cout << "ReshapeNode compute" << std::endl;
        std::cout << "a: " << a->getData().getRows() << "x" << a->getData().getCols() << std::endl;
        std::cout << "into " << data.getRows() << "x" << data.getCols() << std::endl << std::endl;
        #endif

        // data.view(a->getData());
        // grad.view(a->getGrad());
    }

    bool get_requires_grad() override {
        return a->get_requires_grad();
    }

    private:

    static int calculateRows(NodePtr a, int reshaped_rows, int reshaped_cols) {
        int _size = a->getData().getSize();
        return reshaped_rows == -1 ? _size / reshaped_cols : reshaped_rows;
    }

    static int calculateCols(NodePtr a, int reshaped_rows, int reshaped_cols) {
        int _size = a->getData().getSize();
        return reshaped_cols == -1 ? _size / reshaped_rows : reshaped_cols;
    }

};

class Im2rowNode : public UnaryNode {
    public:

    Im2rowNode(NodePtr a, 
        int n, int h, int w, int c,
        int k_h, int k_w, int s_h, int s_w, int p_h, int p_w, bool requires_grad
    )
    : UnaryNode(
        a,
        ((h + 2 * p_h - k_h) / s_h + 1) * ((w + 2 * p_w - k_w) / s_w + 1) * n, 
        k_h * k_w * c, requires_grad
    ) {    
        this->n = n;
        this->h = h;
        this->w = w;
        this->c = c;
        this->k_h = k_h;
        this->k_w = k_w;
        this->s_h = s_h;
        this->s_w = s_w;
        this->p_h = p_h;
        this->p_w = p_w;

        int im_padded_h = h + 2 * p_h;
        int im_padded_w = w + 2 * p_w;

        int out_h = (im_padded_h - k_h) / s_h + 1;
        int out_w = (im_padded_w - k_w) / s_w + 1;

        this->lowered_h = out_h * out_w * n;
        this->lowered_w = k_h * k_w * c;
    }
    
    
    void compute() override {
        #if LOG_OPERATIONS
        std::cout << "Im2rowNode compute" << std::endl;
        std::cout << "a: " << a->getData().getRows() << "x" << a->getData().getCols() << std::endl;
        std::cout << "out: " << data.getRows() << "x" << data.getCols() << std::endl << std::endl << std::endl;
        #endif

        Mat::im2row_nhwc(data, a->getData(), n, h, w, c, k_h, k_w, s_h, s_w, p_h, p_w);
    }
    
    void back() override {
        #if LOG_OPERATIONS
        std::cout << "Im2rowNode back" << std::endl;
        #endif
        if (a->get_requires_grad()) {
            Mat::row2img_nhwc_additive(a->getGrad(), grad, n, h, w, c, k_h, k_w, s_h, s_w, p_h, p_w);
        }
    }
    
    private:

    int n, h, w, c;
    int k_h, k_w;
    int s_h, s_w;
    int p_h, p_w;
    int lowered_h, lowered_w;
};

class MecLowerNode : public UnaryNode {
    public:

    MecLowerNode(NodePtr a, 
        int n, int h, int w, int c,
        int k_h, int k_w, int s_h, int s_w, int p_h, int p_w, bool requires_grad
    )
    : UnaryNode(
        a,
        calculate_lower_h(n, h, w, c, k_h, k_w, s_h, s_w, p_h, p_w),
        calculate_lower_w(n, h, w, c, k_h, k_w, s_h, s_w, p_h, p_w),
        requires_grad
        // ((h + 2 * p_h - k_h) / s_h + 1) * ((w + 2 * p_w - k_w) / s_w + 1) * n, 
        // k_h * k_w * c, requires_grad
    ) {    
        this->n = n;
        this->h = h;
        this->w = w;
        this->c = c;
        this->k_h = k_h;
        this->k_w = k_w;
        this->s_h = s_h;
        this->s_w = s_w;
        this->p_h = p_h;
        this->p_w = p_w;

        int im_padded_h = h + 2 * p_h;
        int im_padded_w = w + 2 * p_w;

        // int out_h = (im_padded_h - k_h) / s_h + 1;
        int out_w = (im_padded_w - k_w) / s_w + 1;

        this->lowered_h = out_w * n;
        this->lowered_w = im_padded_h * k_w * c;
    }
    
    
    void compute() override {
        #if LOG_OPERATIONS
        std::cout << "MecLowerNode compute" << std::endl;
        std::cout << "a: " << a->getData().getRows() << "x" << a->getData().getCols() << std::endl;
        std::cout << "out: " << data.getRows() << "x" << data.getCols() << std::endl << std::endl;
        #endif

        Mat::mec_lower(data, a->getData(), n, h, w, c, k_h, k_w, s_h, s_w, p_h, p_w);
    }
    
    void back() override {
        #if LOG_OPERATIONS
        std::cout << "MecLowerNode back" << std::endl;
        #endif
        if (a->get_requires_grad()) {
            Mat::mec_lower_to_img_additive(a->getGrad(), grad, n, h, w, c, k_h, k_w, s_h, s_w, p_h, p_w);
        }
    }
    
    private:

    int n, h, w, c;
    int k_h, k_w;
    int s_h, s_w;
    int p_h, p_w;
    int lowered_h, lowered_w;

    static int calculate_lower_h(int n, int h, int w, int c,
        int k_h, int k_w, int s_h, int s_w, int p_h, int p_w) {
        
        // int im_padded_h = h + 2 * p_h;
        int im_padded_w = w + 2 * p_w;

        // int out_h = (im_padded_h - k_h) / s_h + 1;
        int out_w = (im_padded_w - k_w) / s_w + 1;

        int _lower_h = out_w * n;
        return _lower_h;
    }

    static int calculate_lower_w(int n, int h, int w, int c,
        int k_h, int k_w, int s_h, int s_w, int p_h, int p_w) {
        
        int im_padded_h = h + 2 * p_h;
        // int im_padded_w = w + 2 * p_w;

        // int out_h = (im_padded_h - k_h) / s_h + 1;
        // int out_w = (im_padded_w - k_w) / s_w + 1;

        int _lower_w = im_padded_h * k_w * c;
        return _lower_w;
    }
};

class MecLowerSeparatedNode : public Node {
    public:

    MecLowerSeparatedNode(vector<shared_ptr<Mat>>& inputs, 
        int n, int h, int w, int c,
        int k_h, int k_w, int s_h, int s_w, bool requires_grad
    )
    : Node(
        calculate_lower_h(n, h, w, c, k_h, k_w, s_h, s_w),
        calculate_lower_w(n, h, w, c, k_h, k_w, s_h, s_w),
        requires_grad
    ), inputs(inputs) {    
        this->n = n;
        this->h = h;
        this->w = w;
        this->c = c;
        this->k_h = k_h;
        this->k_w = k_w;
        this->s_h = s_h;
        this->s_w = s_w;

        // int out_h = (im_padded_h - k_h) / s_h + 1;
        int out_w = (w - k_w) / s_w + 1;

        this->lowered_h = out_w * n;
        this->lowered_w = h * k_w * c;
    }
    
    
    void compute() override {
        #if LOG_OPERATIONS
        std::cout << "MecLowerSeparatedNode compute" << std::endl;
        std::cout << "out: " << data.getRows() << "x" << data.getCols() << std::endl << std::endl;
        #endif

        Mat::mec_lower_separated(data, inputs, n, h, w, c, k_h, k_w, s_h, s_w);
    }
    
    // void back() override {
    //     throw std::runtime_error("Can't go back on MecLowerSeparatedNode");
    // }
    
    private:

    int n, h, w, c;
    int k_h, k_w;
    int s_h, s_w;
    int lowered_h, lowered_w;

    std::vector<shared_ptr<Mat>>& inputs;


    static int calculate_lower_h(int n, int h, int w, int c,
        int k_h, int k_w, int s_h, int s_w) {
        
        // int im_padded_h = h + 2 * p_h;
        // int im_padded_w = w + 2 * p_w;

        // int out_h = (im_padded_h - k_h) / s_h + 1;
        int out_w = (w - k_w) / s_w + 1;

        int _lower_h = out_w * n;
        return _lower_h;
    }

    static int calculate_lower_w(int n, int h, int w, int c,
        int k_h, int k_w, int s_h, int s_w) {
        
        // int im_padded_h = h + 2 * p_h;
        // int im_padded_w = w + 2 * p_w;

        // int out_h = (im_padded_h - k_h) / s_h + 1;
        // int out_w = (im_padded_w - k_w) / s_w + 1;

        int _lower_w = h * k_w * c;
        return _lower_w;
    }
};

class MecConvNode : public BinaryNode {
public:
    MecConvNode(NodePtr lowered, NodePtr kernels, 
                int n, int h, int w, int c,
                int k_h, int k_w, int s_h, int s_w, bool requires_grad)
    : BinaryNode(
        lowered, kernels,
        ((h - k_h) / s_h + 1),
        n * ((w - k_w) / s_w + 1) * kernels->getCols(), requires_grad
    ) {
        this->n = n;
        this->h = h;
        this->w = w;
        this->c = c;
        this->k_h = k_h;
        this->k_w = k_w;
        this->s_h = s_h;
        this->s_w = s_w;
    }

    void compute() override {
        #if LOG_OPERATIONS
        std::cout << "MecConvNode compute" << std::endl;
        std::cout << "lowered: " << a->getData().getRows() << "x" << a->getData().getCols() << std::endl;
        std::cout << "kernels: " << b->getData().getRows() << "x" << b->getData().getCols() << std::endl;
        std::cout << "out: " << data.getRows() << "x" << data.getCols() << std::endl << std::endl;
        #endif

        Mat::mec_conv(data, a->getData(), b->getData(), n, h, w, c, k_h, k_w, s_h, s_w);
    }

    void back() override {
        #if LOG_OPERATIONS
        std::cout << "MecConvNode back" << std::endl;
        #endif
        if (a->get_requires_grad()) {
            Mat::mec_back_into_lowered(a->getGrad(), grad, b->getData(), n, h, w, c, k_h, k_w, s_h, s_w);
        }
    }

private:
    int n, h, w, c;
    int k_h, k_w;
    int s_h, s_w;
};

class Maxpool_hnwc_to_nhwc_Node : public UnaryNode {
    public:

    Maxpool_hnwc_to_nhwc_Node(NodePtr a, 
        int n, int h, int w, int c,
        int k_h, int k_w, int s_h, int s_w, bool requires_grad
    )
    : UnaryNode(
        a,
        ((h - k_h) / s_h + 1) * n, 
        ((w - k_w) / s_w + 1) * c, requires_grad
    ) {    
        this->n = n;
        this->h = h;
        this->w = w;
        this->c = c;
        this->k_h = k_h;
        this->k_w = k_w;
        this->s_h = s_h;
        this->s_w = s_w;

        this->out_h = (h - k_h) / s_h + 1;
        this->out_w = (w - k_w) / s_w + 1;

        this->indeces = new int[out_h * out_w * n * c];
    }

    ~Maxpool_hnwc_to_nhwc_Node() {
        delete[] indeces;
    }
    
    void compute() override {
        #if LOG_OPERATIONS
        std::cout << "Maxpool_hnwc_to_nhwc_Node compute" << std::endl;
        std::cout << "a: " << a->getData().getRows() << "x" << a->getData().getCols() << std::endl;
        std::cout << "out: " << data.getRows() << "x" << data.getCols() << std::endl << std::endl;
        #endif

        Mat::maxpool_hnwc_to_nhwc(data, a->getData(), n, h, w, c, k_h, k_w, s_h, s_w, indeces);
    }
    
    void back() override {
        #if LOG_OPERATIONS
        std::cout << "Maxpool_hnwc_to_nhwc_Node back" << std::endl;
        #endif
        if (a->get_requires_grad()) {
            float* a_grad_data = a->getGrad().getData();
            float* grad_data = grad.getData();

            // for (int i = 0; i < out_h * out_w * n * c; i++) {
            //     std::cout << i << ": " << indeces[i] << "\n";
            // }

            for (int i = 0; i < out_h * out_w * n * c; i++) {
                a_grad_data[indeces[i]] += grad_data[i];
            }
        }
    }
    
    private:
    int* indeces;
    int n, h, w, c;
    int k_h, k_w;
    int s_h, s_w;
    int out_h, out_w;
};

NodePtr Node::operator+(NodePtr other) { return std::make_shared<PlusNode>(shared_from_this(), other, true); }

NodePtr Node::plus(NodePtr a, NodePtr b) { return std::make_shared<PlusNode>(a, b, true); }

NodePtr Node::operator-(NodePtr other) { return std::make_shared<MinusNode>(shared_from_this(), other, true); }

NodePtr Node::matmul(NodePtr a, NodePtr b) { return std::make_shared<MatMulNode>(a, b, true); }

NodePtr Node::sum() { return std::make_shared<SumNode>(shared_from_this(), true); }

NodePtr Node::activation(NodePtr a, float (*act)(float), float (*act_derivative)(float)) {
    return std::make_shared<ActivationNode>(a, act, act_derivative, true);
}

NodePtr Node::relu(NodePtr a) { return std::make_shared<ReLUNode>(a, true); }

NodePtr Node::sigmoid(NodePtr a) { return std::make_shared<SigmoidNode>(a, true); }

NodePtr Node::tanh(NodePtr a) { return std::make_shared<TanhNode>(a, true); }

NodePtr Node::pow(NodePtr a, float pow) { return std::make_shared<PowNode>(a, pow, true); }

NodePtr Node::mat_plus_vec(NodePtr a, NodePtr b) { return std::make_shared<MatPlusVecNode>(a, b, true); }

NodePtr Node::mat_plus_row_vec(NodePtr a, NodePtr b) { return std::make_shared<MatPlusRowVecNode>(a, b, true); }

NodePtr Node::dropout(int rows, int cols, float dropout_rate, bool requires_grad) {
    return std::make_shared<DropoutNode>(rows, cols, dropout_rate, requires_grad);
}

NodePtr Node::hadamard_product(NodePtr a, NodePtr b) { return std::make_shared<HadamardProductNode>(a, b, true); }

NodePtr Node::transpose(NodePtr a) { return std::make_shared<TransposeNode>(a, true); }

NodePtr Node::im2row(NodePtr a, int n, int h, int w, int c, int k_h, int k_w, int s_h, int s_w,
    int p_h, int p_w, bool requires_grad) {
    return std::make_shared<Im2rowNode>(a, n, h, w, c, k_h, k_w, s_h, s_w, p_h, p_w, requires_grad);
}

NodePtr Node::reshape(NodePtr a, int reshaped_rows, int reshaped_cols, bool requires_grad) {
    return std::make_shared<ReshapeNode>(a, reshaped_rows, reshaped_cols, requires_grad);
}

NodePtr Node::mec_lower(NodePtr a, int n, int h, int w, int c, int k_h, int k_w, int s_h, int s_w,
        int p_h, int p_w, bool requires_grad) {
    return std::make_shared<MecLowerNode>(a, n, h, w, c, k_h, k_w, s_h, s_w, p_h, p_w, requires_grad);
}

NodePtr Node::mec_lower_separated(std::vector<shared_ptr<Mat>>& inputs, int n, int h, int w, int c, int k_h, int k_w, int s_h, int s_w, bool requires_grad) {
    return std::make_shared<MecLowerSeparatedNode>(inputs, n, h, w, c, k_h, k_w, s_h, s_w, requires_grad);
}

NodePtr Node::mec_conv_mm(NodePtr lowered, NodePtr kernels, int n, int h, int w, int c,
                int k_h, int k_w, int s_h, int s_w, bool requires_grad) {
    return std::make_shared<MecConvNode>(lowered, kernels, n, h, w, c, k_h, k_w, s_h, s_w, requires_grad);
}

NodePtr Node::maxpool_hnwc_to_nhwc(NodePtr a, int n, int h, int w, int c, int k_h, int k_w, int s_h, int s_w, bool requires_grad) {
    return std::make_shared<Maxpool_hnwc_to_nhwc_Node>(a, n, h, w, c, k_h, k_w, s_h, s_w, requires_grad);
}

#endif