#include <cmath>

namespace activation_functions {

float sigmoid(float x) { return 1.0 / (1.0 + std::exp(-x)); }

float sigmoid_derivative(float x) {
    float sig = sigmoid(x);
    return sig * (1 - sig);
}

float sigmoid_derivative_with_sig(float sig) {
    return sig * (1 - sig);
}

float tanh(float x) { return std::tanh(x); }

float tanh_derivative(float x) {
    float tanh_x = std::tanh(x);
    return 1 - (tanh_x * tanh_x);
}

float tanh_derivative_with_tanh(float tanh_x) {
    return 1 - (tanh_x * tanh_x);
}

float relu(float x) { return std::fmax(0.0, x); }

float relu_derivative(float x) { return (x > 0) ? 1.0 : 0.0; }

float leaky_relu(float x, float alpha = 0.01) {
    return (x > 0) ? x : alpha * x;
}

float leaky_relu_derivative(float x, float alpha = 0.01) {
    return (x > 0) ? 1.0 : alpha;
}

}  // namespace activations