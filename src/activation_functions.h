#include <cmath>

namespace activation_functions {

inline float sigmoid(float x) { return 1.0 / (1.0 + std::exp(-x)); }

inline float sigmoid_derivative(float x) {
    float sig = sigmoid(x);
    return sig * (1 - sig);
}

inline float sigmoid_derivative_with_sig(float sig) {
    return sig * (1 - sig);
}

inline float tanh(float x) { return std::tanh(x); }

inline float tanh_derivative(float x) {
    float tanh_x = std::tanh(x);
    return 1 - (tanh_x * tanh_x);
}

inline float tanh_derivative_with_tanh(float tanh_x) {
    return 1 - (tanh_x * tanh_x);
}

inline float relu(float x) { return x > 0.0f ? x : 0.0f; }

inline float relu_derivative(float x) {
    // return (x > 0) ? 1.0 : 0.0;
    return (float)(x > 0);
}

inline float leaky_relu(float x, float alpha = 0.01) {
    return (x > 0) ? x : alpha * x;
}

inline float leaky_relu_derivative(float x, float alpha = 0.01) {
    return (x > 0) ? 1.0 : alpha;
}

}  // namespace activations