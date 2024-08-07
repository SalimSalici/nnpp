#include "Mat.h"
#include "SubMat.h"
#include "activation_functions.h"

#include <math.h>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <functional>
#include <limits>
#include <vector>
#include <immintrin.h>  // AVX2

extern "C" {
#include <cblas.h>
}

Mat::Mat(int rows, int cols, bool alloc_data) : rows(rows), cols(cols), size(rows * cols), is_transposed(false), right(1), down(cols), is_view(false) {
    if (alloc_data)
        data = new float[size];
    else data = nullptr;
}

// Copy constructor
Mat::Mat(const Mat& other)
    : rows(other.rows), cols(other.cols), size(other.size),
        is_transposed(other.is_transposed), right(other.right), down(other.down) {
    // std::cout << "Copy constructor of Mat called";
    data = new float[size];
    std::memcpy(data, other.data, size * sizeof(float));
}

// Move constructor
Mat::Mat(Mat&& other) noexcept
    : rows(other.rows), cols(other.cols), size(other.size),
        is_transposed(other.is_transposed), right(other.right), down(other.down) {
    // std::cout << "Move constructor of Mat called";
    data = other.data;
    other.data = nullptr;
}

// Copy assignment
Mat& Mat::operator=(const Mat& other) {
    // std::cout << "Copy assignment of Mat called";
    if (this != &other) {
        float* newData = new float[other.size];
        std::memcpy(newData, other.data, size * sizeof(float));
        delete[] data;
        rows = other.rows;
        cols = other.cols;
        size = other.size;
        data = newData;
        is_transposed = other.is_transposed;
        right = other.right;
        down = other.down;
    }
    return *this;
}

// Move assignment
Mat& Mat::operator=(Mat&& other) {
    // std::cout << "Move assignment of Mat called";
    if (this != &other) {
        delete[] data;
        data = other.data;
        other.data = nullptr;
        rows = other.rows;
        cols = other.cols;
        size = other.size;
        is_transposed = other.is_transposed;
        right = other.right;
        down = other.down;
    }
    return *this;
}

bool Mat::operator==(const Mat& other) {
    if (this->is_transposed || other.is_transposed) {
        throw std::runtime_error("Comparison of transposed matrices is not supported.");
    }

    if (this->rows != other.rows || this->cols != other.cols) {
        return false;
    }

    for (int i = 0; i < this->size; ++i) {
        if (this->data[i] != other.data[i]) {
            return false;
        }
    }

    return true;
}

Mat::~Mat() {
    if (!is_view && data != nullptr)
        delete[] data;
}

void Mat::print() const {
    std::cout << "Rows: " << rows << ", Cols: " << cols << ", Transposed: " << is_transposed << "\n";
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) std::cout << data[r * cols + c] << " ";
        std::cout << "\n";
    }
}

void Mat::print_ascii_greyscale() const {
    const char* shades = " .:-=+*#%@";
    const int num_shades = 10;

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            float value = data[r * cols + c];
            value = std::max(0.0f, std::min(1.0f, value));
            int shade_index = static_cast<int>(value * (num_shades - 1));
            std::cout << shades[shade_index];
        }
        std::cout << "\n";
    }
    std::cout << std::flush;
}

int Mat::getRows() const {
    if (is_transposed) return cols;
    return rows;
}

int Mat::getCols() const {
    if (is_transposed) return rows;
    return cols;
}

int Mat::getSize() const {
    return size;
}

bool Mat::isTransposed() const {
    return is_transposed;
}

int Mat::getRight() const {
    return right;
}

int Mat::getDown() const {
    return down;
}

Mat& Mat::copy_from(const float* other_data) {
    std::memcpy(data, other_data, size * sizeof(float));
    return *this;
}

Mat& Mat::put(float value, int row, int col) {
    if (is_transposed) {
        int temp = row;
        row = col;
        col = temp;
    }
    int i = row * cols + col;
    if (i >= size) throw std::invalid_argument("Index out of bounds.");
    data[i] = value;
    return *this;
}

Mat& Mat::zero() {
    memset(data, 0, size * sizeof(float));
    return *this;
}

Mat& Mat::fill(float value) { 
    for (int i = 0; i < size; i++)
        data[i] = value;
    return *this;
}

Mat& Mat::fill_rand_rate(float value_p, float value_not_p, float p) { 
    for (int i = 0; i < size; i++)
        data[i] = ((float)rand() / RAND_MAX) < p ? value_p : value_not_p;
    return *this;
}

Mat& Mat::transpose() {
    is_transposed = !is_transposed;
    right = is_transposed ? cols : 1;
    down = is_transposed ? 1 : cols;
    return *this;
}

float* Mat::getData() const { return data; }

float Mat::elementsSum() const {
    float result = 0;
    for (int i = 0; i < size; i++) result += data[i];
    return result;
}

void Mat::plus(Mat& result, const Mat& a, const Mat& b) {
    if (result.isTransposed() || a.isTransposed() || b.isTransposed()) {
        Mat::element_op_tr_supp(result, a, b, std::plus<float>());
        return;
    }

    if (
        result.rows != a.rows || result.cols != a.cols ||
        result.rows != b.rows || result.cols != b.cols
    ) throw std::invalid_argument("Matrix mismatch in Mat::plus. Dimensions: result(" + std::to_string(result.rows) + "," + std::to_string(result.cols) + "), a(" + std::to_string(a.rows) + "," + std::to_string(a.cols) + "), b(" + std::to_string(b.rows) + "," + std::to_string(b.cols) + ")");
    for (int i = 0; i < result.size; i++) result.data[i] = a.data[i] + b.data[i];
}

void Mat::element_op_tr_supp(Mat& result, const Mat& a, const Mat& b, std::function<float(float, float)> op) {
    int r_rows = result.getRows();
    int r_cols = result.getCols();
    int a_rows = a.getRows();
    int a_cols = a.getCols();
    int b_rows = b.getRows();
    int b_cols = b.getCols();

    int r_right = result.getRight();
    int r_down = result.getDown();

    int a_right = a.getRight();
    int a_down = a.getDown();

    int b_right = b.getRight();
    int b_down = b.getDown();

    if (r_rows != a_rows || r_cols != a_cols || r_rows != b_rows || r_cols != b_cols)
        throw std::invalid_argument("Matrix mismatch in Mat::element_op_tr_supp. Dimensions: result(" + std::to_string(r_rows) + "," + std::to_string(r_cols) + "), a(" + std::to_string(a_rows) + "," + std::to_string(a_cols) + "), b(" + std::to_string(b_rows) + "," + std::to_string(b_cols) + ")");

    float* r_data = result.getData();
    float* a_data = a.getData();
    float* b_data = b.getData();

    for (int r = 0; r < r_rows; r++) {
        float *r_curr = r_data + r * r_down;
        float *a_curr = a_data + r * a_down;
        float *b_curr = b_data + r * b_down;
        for (int c = 0; c < r_cols; c++) {
            *r_curr = op(*a_curr, *b_curr);
            r_curr += r_right;
            a_curr += a_right;
            b_curr += b_right;
        }
    }
}

void Mat::element_op_tr_supp_keep_res(Mat& result, const Mat& a, const Mat& b, float result_scaling, std::function<float(float, float)> op) {
    int r_rows = result.getRows();
    int r_cols = result.getCols();
    int a_rows = a.getRows();
    int a_cols = a.getCols();
    int b_rows = b.getRows();
    int b_cols = b.getCols();

    int r_right = result.getRight();
    int r_down = result.getDown();

    int a_right = a.getRight();
    int a_down = a.getDown();

    int b_right = b.getRight();
    int b_down = b.getDown();

    if (r_rows != a_rows || r_cols != a_cols || r_rows != b_rows || r_cols != b_cols)
        throw std::invalid_argument("Matrix mismatch in Mat::element_op_tr_supp_keep_res. Dimensions: result(" + std::to_string(r_rows) + "," + std::to_string(r_cols) + "), a(" + std::to_string(a_rows) + "," + std::to_string(a_cols) + "), b(" + std::to_string(b_rows) + "," + std::to_string(b_cols) + ")");

    float* r_data = result.getData();
    float* a_data = a.getData();
    float* b_data = b.getData();

    for (int r = 0; r < r_rows; r++) {
        float *r_curr = r_data + r * r_down;
        float *a_curr = a_data + r * a_down;
        float *b_curr = b_data + r * b_down;
        for (int c = 0; c < r_cols; c++) {
            *r_curr = op(*a_curr, *b_curr) + (*r_curr * result_scaling);
            r_curr += r_right;
            a_curr += a_right;
            b_curr += b_right;
        }
    }
}

Mat Mat::operator+(const Mat& other) const {
    if (rows != other.rows || cols != other.cols)
        throw std::invalid_argument("Matrix mismatch in Mat::operator+. Dimensions: this(" + std::to_string(rows) + "," + std::to_string(cols) + "), other(" + std::to_string(other.rows) + "," + std::to_string(other.cols) + ")");

    Mat result(rows, cols);
    Mat::plus(result, *this, other);
    return result;
}

void Mat::minus(Mat& result, const Mat& a, const Mat& b) {
    if (result.isTransposed() || a.isTransposed() || b.isTransposed()) {
        Mat::element_op_tr_supp(result, a, b, std::minus<float>());
        return;
    }

    if (
        result.rows != a.rows || result.cols != a.cols ||
        result.rows != b.rows || result.cols != b.cols
    ) throw std::invalid_argument("Matrix mismatch in Mat::minus. Dimensions: result(" + std::to_string(result.rows) + "," + std::to_string(result.cols) + "), a(" + std::to_string(a.rows) + "," + std::to_string(a.cols) + "), b(" + std::to_string(b.rows) + "," + std::to_string(b.cols) + ")");
    for (int i = 0; i < result.size; i++) result.data[i] = a.data[i] - b.data[i];
}

Mat Mat::operator-(const Mat& other) const {
    if (rows != other.rows || cols != other.cols)
        throw std::invalid_argument("Matrix mismatch in Mat::operator-. Dimensions: this(" + std::to_string(rows) + "," + std::to_string(cols) + "), other(" + std::to_string(other.rows) + "," + std::to_string(other.cols) + ")");

    Mat result(rows, cols);
    Mat::minus(result, *this, other);
    return result;
}

void Mat::hadamardProduct(Mat& result, const Mat& a, const Mat& b) {
    if (result.isTransposed() || a.isTransposed() || b.isTransposed()) {
        Mat::element_op_tr_supp(result, a, b, std::multiplies<float>());
        return;
    }

    if (
        result.rows != a.rows || result.cols != a.cols ||
        result.rows != b.rows || result.cols != b.cols
    ) throw std::invalid_argument("Matrix mismatch in Mat::hadamardProduct. Dimensions: result(" + std::to_string(result.rows) + "," + std::to_string(result.cols) + "), a(" + std::to_string(a.rows) + "," + std::to_string(a.cols) + "), b(" + std::to_string(b.rows) + "," + std::to_string(b.cols) + ")");
    
    // for (int i = 0; i < result.size; i++) result.data[i] = a.data[i] * b.data[i];
    
    const int simd_width = 8; // AVX2 processes 8 floats at a time
    int i = 0;
    
    // Process 8 elements at a time using AVX2
    for (; i < result.size - simd_width; i += simd_width) {
        __m256 a_vec = _mm256_loadu_ps(&a.data[i]);
        __m256 b_vec = _mm256_loadu_ps(&b.data[i]);
        __m256 result_vec = _mm256_mul_ps(a_vec, b_vec);
        _mm256_storeu_ps(&result.data[i], result_vec);
    }
    
    // Handle remaining elements
    for (; i < result.size; i++) {
        result.data[i] = a.data[i] * b.data[i];
    }
}

void Mat::hadamardProduct_keep_res(Mat& result, const Mat& a, const Mat& b, float result_scaling) {
    if (result.isTransposed() || a.isTransposed() || b.isTransposed()) {
        Mat::element_op_tr_supp_keep_res(result, a, b, result_scaling, std::multiplies<float>());
        return;
    }

    if (
        result.rows != a.rows || result.cols != a.cols ||
        result.rows != b.rows || result.cols != b.cols
    ) throw std::invalid_argument("Matrix mismatch in Mat::hadamardProduct_keep_res. Dimensions: result(" + std::to_string(result.rows) + "," + std::to_string(result.cols) + "), a(" + std::to_string(a.rows) + "," + std::to_string(a.cols) + "), b(" + std::to_string(b.rows) + "," + std::to_string(b.cols) + ")");
    
    // for (int i = 0; i < result.size; i++) result.data[i] = a.data[i] * b.data[i] + result.data[i] * result_scaling;
    
    const int simd_width = 8; // AVX2 processes 8 floats at a time
    int i = 0;

    float* result_data = result.getData();
    float* a_data = a.getData();
    float* b_data = b.getData();
    
    if (result_scaling != 1) {
        // Process 8 elements at a time using AVX2
        __m256 scaling_vec = _mm256_set1_ps(result_scaling);
        for (; i < result.size - simd_width; i += simd_width) {
            __m256 a_vec = _mm256_loadu_ps(a_data + i);
            __m256 b_vec = _mm256_loadu_ps(b_data + i);
            __m256 result_vec = _mm256_loadu_ps(result_data + i);
            __m256 product = _mm256_mul_ps(a_vec, b_vec);
            __m256 scaled_result = _mm256_mul_ps(result_vec, scaling_vec);
            __m256 final_result = _mm256_add_ps(product, scaled_result);
            _mm256_storeu_ps(result_data + i, final_result);
        }
    } else {
        for (; i < result.size - simd_width; i += simd_width) {
            __m256 a_vec = _mm256_loadu_ps(a_data + i);
            __m256 b_vec = _mm256_loadu_ps(b_data + i);
            __m256 result_vec = _mm256_loadu_ps(result_data + i);
            __m256 product = _mm256_mul_ps(a_vec, b_vec);
            __m256 final_result = _mm256_add_ps(product, result_vec);
            _mm256_storeu_ps(result_data + i, final_result);
        }
    }
    
    // Handle remaining elements
    for (; i < result.size; i++) {
        result.data[i] = a.data[i] * b.data[i] + result.data[i] * result_scaling;
    }
}

Mat Mat::operator*(const Mat& other) const {
    if (rows != other.rows || cols != other.cols)
        throw std::invalid_argument("Matrix mismatch in Mat::operator*. Dimensions: this(" + std::to_string(rows) + "," + std::to_string(cols) + "), other(" + std::to_string(other.rows) + "," + std::to_string(other.cols) + ")");

    Mat result(rows, cols);
    Mat::hadamardProduct(result, *this, other);
    return result;
}

Mat& Mat::operator+=(const Mat& other) {
    if (rows != other.rows || cols != other.cols)
        throw std::invalid_argument("Matrix mismatch in Mat::operator+=. Dimensions: this(" + std::to_string(rows) + "," + std::to_string(cols) + "), other(" + std::to_string(other.rows) + "," + std::to_string(other.cols) + ")");

    for (int i = 0; i < size; i++) data[i] += other.data[i];
    return *this;
}

Mat& Mat::operator-=(const Mat& other) {
    if (rows != other.rows || cols != other.cols)
        throw std::invalid_argument("Matrix mismatch in Mat::operator-=. Dimensions: this(" + std::to_string(rows) + "," + std::to_string(cols) + "), other(" + std::to_string(other.rows) + "," + std::to_string(other.cols) + ")");

    for (int i = 0; i < size; i++) data[i] -= other.data[i];
    return *this;
}

float Mat::operator[](int idx) const {
    if (idx >= size) throw std::invalid_argument("Index out of bounds.");
    return data[idx];
}

float Mat::getElement(int row, int col) const {
    if (is_transposed) {
        int temp = row;
        row = col;
        col = temp;
    }
    int i = row * cols + col;
    if (i >= size) throw std::invalid_argument("Index out of bounds.");
    return data[i];
}

float Mat::get_element_fallback(int row, int col, float fallback) const {
    if (is_transposed) {
        int temp = row;
        row = col;
        col = temp;
    }
    int i = row * cols + col;
    if (i >= size) return fallback;
    return data[i];
}

// void Mat::matmul(Mat& result, const Mat& a, const Mat& b) {
//     int a_rows = a.getRows();
//     int a_cols = a.getCols();
//     int b_rows = b.getRows();
//     int b_cols = b.getCols();

//     if (a_cols != b_rows || result.rows != a_rows || result.cols != b_cols) {
//         throw std::invalid_argument("Matrix dimensions mismatch for multiplication.");
//     }

//     for (int i = 0; i < a_rows; ++i) {
//         for (int j = 0; j < b_cols; ++j) {
//             float sum = 0.0f;
//             for (int k = 0; k < a_cols; ++k) {
//                 float a_val = a.getElement(i, k);
//                 float b_val = b.getElement(k, j);
//                 sum += a_val * b_val;
//             }
//             result.put(sum, i, j);
//         }
//     }
// }

void Mat::matmul(Mat& result, const Mat& a, const Mat& b) {
    Mat::matmul_mm(result, a, b, 1.0f, 0.0f);
}

void Mat::matmul_mm(Mat& result, const Mat& a, const Mat& b, float ab_s, float c_s) {
    int a_rows = a.getRows();
    int a_cols = a.getCols();
    int b_rows = b.getRows();
    int b_cols = b.getCols();

    if (a_cols != b_rows || result.rows != a_rows || result.cols != b_cols) {
        std::cout << "a_rows: " << a_rows << ", a_cols: " << a_cols << ", b_rows: " << b_rows << ", b_cols: " << b_cols << ", result_rows: " << result.rows << ", result_cols: " << result.cols << std::endl;
        throw std::invalid_argument("Matrix dimensions mismatch for multiplication.");
    }

    CBLAS_TRANSPOSE a_transposed = a.isTransposed() ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE b_transposed = b.isTransposed() ? CblasTrans : CblasNoTrans;

    cblas_sgemm(CblasRowMajor, 
                a_transposed, b_transposed,
                a_rows, b_cols, a_cols,
                ab_s, a.getData(), a.cols, b.getData(), b.cols,
                c_s, result.getData(), result.cols);
}

void Mat::matmul_mv(Mat& result, const Mat& a, const Mat& b, float ab_s, float c_s) {
    int a_rows = a.getRows();
    int a_cols = a.getCols();
    int b_rows = b.getRows();
    int b_cols = b.getCols();

    if (a_cols != b_rows || result.rows != a_rows || result.cols != b_cols)
        throw std::invalid_argument("Matrix dimensions mismatch for multiplication.");

    if (b_cols != 1)
        throw std::invalid_argument("Matrix b must be a column vector for matmul_mv.");

    CBLAS_TRANSPOSE a_transposed = a.isTransposed() ? CblasTrans : CblasNoTrans;

    cblas_sgemv(CblasRowMajor, 
            a_transposed,
            a.rows, a.cols,
            ab_s, a.data, a.cols, b.data, 1, 
            c_s, result.data, 1);
}

Mat Mat::matmul(const Mat& a, const Mat& b) {
    Mat result(a.getRows(), b.getCols());
    Mat::matmul(result, a, b);
    return result;
}

Mat& Mat::apply(float (*act)(float)) {
    for (int i = 0; i < size; i++)
        data[i] = act(data[i]);
    return *this;
}

Mat Mat::apply(const Mat& a, float (*act)(float)) {
    Mat result(a.rows, a.cols);
    Mat::apply(result, a, act);
    return result;
}

void Mat::apply(Mat& result, const Mat& a, float (*act)(float)) {
    if (result.rows != a.rows || result.cols != a.cols)
        throw std::invalid_argument("Matrix dimensions mismatch for apply.");
    for (int i = 0; i < result.size; i++)
        result.data[i] = act(a.data[i]);
}

void Mat::apply(Mat& result, const Mat& a, float (*act)(float, void*), void* args) {
    if (result.rows != a.rows || result.cols != a.cols)
        throw std::invalid_argument("Matrix dimensions mismatch for apply.");
    for (int i = 0; i < result.size; i++)
        result.data[i] = act(a.data[i], args);
}

void Mat::apply_log(Mat& result, const Mat& a) {
    if (result.rows != a.rows || result.cols != a.cols)
        throw std::invalid_argument("Matrix dimensions mismatch for apply_log.");
    for (int i = 0; i < result.size; i++)
        result.data[i] = std::log(a.data[i] + 0.00000000001);
}

void Mat::apply_relu(Mat& result, const Mat& a) {
    if (result.rows != a.rows || result.cols != a.cols)
        throw std::invalid_argument("Matrix dimensions mismatch for apply.");

    float* const a_data = a.getData();
    float* const result_data = result.getData();
    for (int i = 0; i < result.size; i++)
        result_data[i] = activation_functions::relu(a_data[i]);

}

void Mat::apply_relu_derivative(Mat& result, const Mat& a) {
    if (result.rows != a.rows || result.cols != a.cols)
        throw std::invalid_argument("Matrix dimensions mismatch for apply.");

    float* const a_data = a.getData();
    float* const result_data = result.getData();
    for (int i = 0; i < result.size; i++)
        result_data[i] = activation_functions::relu_derivative(a_data[i]);

    // for (int i = 0; i < result.size; i++) {
    //     result.data[i] = a.data[i] > 0.0f ? 1.0f : 0.0f;
    // }
}

Mat& Mat::raiseEach(int power) {
    for (int i = 0; i < size; i++)
        data[i] = std::pow(data[i], power);
    return *this;
}

Mat& Mat::raiseEach(float power) {
    for (int i = 0; i < size; i++)
        data[i] = std::pow(data[i], power);
    return *this;
}

Mat Mat::pow(const Mat& a, int power) {
    Mat result(a.rows, a.cols);
    Mat::pow(result, a, power);
    return result;
}

Mat Mat::pow(const Mat& a, float power) {
    Mat result(a.rows, a.cols);
    Mat::pow(result, a, power);
    return result;
}

void Mat::pow(Mat& result, const Mat& a, float power) {
    if (result.rows != a.rows || result.cols != a.cols)
        throw std::invalid_argument("Matrix dimensions mismatch for apply.");
    for (int i = 0; i < result.size; i++)
        result.data[i] = std::pow(a.data[i], power);
}

void Mat::pow(Mat& result, const Mat& a, int power) {
    if (result.rows != a.rows || result.cols != a.cols)
        throw std::invalid_argument("Matrix dimensions mismatch for apply.");
    for (int i = 0; i < result.size; i++)
        result.data[i] = std::pow(a.data[i], power);
}

Mat Mat::scale(const Mat& a, float factor) {
    Mat result(a.rows, a.cols);
    Mat::scale(result, a, factor);
    return result;
}

void Mat::scale(Mat& result, const Mat& a, float factor) {
    if (result.rows != a.rows || result.cols != a.cols)
        throw std::invalid_argument("Matrix dimensions mismatch for apply.");
    for (int i = 0; i < result.size; i++)
        result.data[i] = a.data[i] * factor;
}

void Mat::softmax(Mat& result, const Mat& a, bool rows) {
    if (result.rows != a.rows || result.cols != a.cols)
        throw std::invalid_argument("Matrix dimensions mismatch for softmax.");

    int r = a.getRows();
    int c = a.getCols();
    float* a_data = a.getData();
    float* result_data = result.getData();

    if (rows) {
        for (int i = 0; i < r; ++i) {
            float max_val = a_data[i * c];
            for (int j = 1; j < c; ++j) {
                max_val = std::max(max_val, a_data[i * c + j]);
            }

            float sum = 0.0f;
            for (int j = 0; j < c; ++j) {
                float exp_val = std::exp(a_data[i * c + j] - max_val);
                result_data[i * c + j] = exp_val;
                sum += exp_val;
            }

            for (int j = 0; j < c; ++j) {
                result_data[i * c + j] /= sum;
            }
        }
    } else {

        // Claude Sonnet 3.5 approach (tested to be slower than my approach)
        // for (int j = 0; j < c; ++j) {
        //     float max_val = a_data[j];
        //     for (int i = 1; i < r; ++i) {
        //         max_val = std::max(max_val, a_data[i * c + j]);
        //     }

        //     float sum = 0.0f;
        //     for (int i = 0; i < r; ++i) {
        //         float exp_val = std::exp(a_data[i * c + j] - max_val);
        //         result_data[i * c + j] = exp_val;
        //         sum += exp_val;
        //     }

        //     for (int i = 0; i < r; ++i) {
        //         result_data[i * c + j] /= sum;
        //     }
        // }

        // My approach (tested to be faster than Claude Sonnet 3.5 approach)
        // Key insight: we process the matrix row after row, instead of column after column.
        // To do this we need the two supporting arrays "max_vals" and "sums"

        float max_vals[c];
        float sums[c] = {0};

        std::memcpy(max_vals, a_data, c * sizeof(float));

        for (int i = 1; i < r; ++i) {
            for (int j = 0; j < c; ++j) {
                max_vals[j] = std::max(max_vals[j], a_data[i * c + j]);
            }
        }

        for (int i = 0; i < r; ++i) {
            for (int j = 0; j < c; ++j) {
                float exp_val = std::exp(a_data[i * c + j] - max_vals[j]);
                result_data[i * c + j] = exp_val;
                sums[j] += exp_val;
            }
        }

        for (int i = 0; i < r; ++i) {
            for (int j = 0; j < c; ++j) {
                result_data[i * c + j] /= sums[j];
            }
        }
    }
}

// void Mat::mat_plus_scalar(Mat& result, const Mat& mat, float scalar, float mat_scaling) {

//     std::cout << result.rows << " " << result.cols << " | " << mat.rows << " " << mat.cols << std::endl;

//     if (result.rows != mat.rows || result.cols != mat.cols)
//         throw std::invalid_argument("Matrix dimensions mismatch for mat_plus_scalar.");

//     for (int i = 0; i < result.size; i++)
//         result.data[i] = mat_scaling * mat.data[i] + scalar;
// }

void Mat::mat_plus_scalar(Mat& result, const Mat& mat, float scalar, float mat_scaling) {
    if (result.getRows() != mat.getRows() || result.getCols() != mat.getCols())
        throw std::invalid_argument("Matrix dimensions mismatch for mat_plus_scalar.");

    int r_rows = result.getRows();
    int r_cols = result.getCols();
    int r_right = result.getRight();
    int r_down = result.getDown();
    int m_right = mat.getRight();
    int m_down = mat.getDown();

    float* r_data = result.getData();
    float* m_data = mat.getData();

    for (int r = 0; r < r_rows; r++) {
        float* r_curr = r_data + r * r_down;
        float* m_curr = m_data + r * m_down;
        for (int c = 0; c < r_cols; c++) {
            *r_curr = mat_scaling * (*m_curr) + scalar;
            r_curr += r_right;
            m_curr += m_right;
        }
    }
}

void Mat::mat_plus_vec(Mat& result, const Mat& mat, const Mat& vec) {
    if (result.rows != mat.rows || result.cols != mat.cols || vec.cols != 1 || vec.rows != mat.rows)
        throw std::invalid_argument("Matrix dimensions mismatch for mat_plus_vec.");

    int mat_cols = mat.cols;
    for (int r = 0; r < vec.rows; r++) {
        float vec_val = vec.data[r];
        for (int c = 0; c < mat.cols; c++) {
            int mat_idx = r * mat_cols + c;
            result.data[mat_idx] = mat.data[mat_idx] + vec_val;
        }
    }
}

// result[i] = vec[i] + sum(mat[i,:])
void Mat::vec_plus_mat(Mat& result, const Mat& vec, const Mat& mat) {
    if (result.rows != vec.rows || result.cols != vec.cols || vec.cols != 1 || vec.rows != mat.rows)
        throw std::invalid_argument("Matrix dimensions mismatch for vec_plus_mat.");

    int mat_cols = mat.cols;
    for (int r = 0; r < vec.rows; r++) {
        float sum = 0;
        for (int c = 0; c < mat.cols; c++) {
            int mat_idx = r * mat_cols + c;
            sum += mat.data[mat_idx];
        }
        result.data[r] = vec[r] + sum;
    }
}

void Mat::mat_plus_row_vec(Mat& result, const Mat& mat, const Mat& vec) {
    if (result.rows != mat.rows || result.cols != mat.cols || vec.rows != 1 || vec.cols != mat.cols)
        throw std::invalid_argument("Matrix dimensions mismatch for mat_plus_row_vec.");

    const int mat_rows = mat.rows;
    const int mat_cols = mat.cols;

    float* const result_data = result.getData();
    float* const mat_data = mat.getData();
    int disp = 0;
    for (int r = 0; r < mat_rows; r++) {
        for (int c = 0; c < mat_cols; c++) {
            *(result_data + disp) = *(mat_data + disp) + vec.data[c];
            disp++;
            // const int mat_idx = r * mat_cols + c;
            // result.data[mat_idx] = mat.data[mat_idx] + vec.data[c];
        }
    }
}

// result[i] = vec[i] + sum(mat[:,i])
void Mat::row_vec_plus_mat(Mat& result, const Mat& vec, const Mat& mat) {
    if (result.rows != 1 || result.cols != vec.cols || vec.rows != 1 || vec.cols != mat.cols)
        throw std::invalid_argument("Matrix dimensions mismatch for row_vec_plus_mat.");

    result.zero();

    const int mat_rows = mat.rows;
    const int mat_cols = mat.cols;

    float* result_data = result.getData();
    float* mat_data = mat.getData();

    for (int r = 0; r < mat_rows; r++) {
        for (int c = 0; c < mat_cols; c++) {
            result_data[c] += mat_data[r * mat.cols + c];
        }
    }
}

void Mat::im2row_nhwc(Mat& result, Mat& im, int n, int h, int w, int c, int k_h, int k_w, int s_h, int s_w, int p_h, int p_w) {

    // int im_cols = w * c;
    // int im_right = 1;
    // int im_down = im_cols;

    int im_padded_h = h + 2 * p_h;
    int im_padded_w = w + 2 * p_w;

    int out_h = (im_padded_h - k_h) / s_h + 1;
    int out_w = (im_padded_w - k_w) / s_w + 1;

    int lowered_h = out_h * out_w * n;
    int lowered_w = k_h * k_w * c;

    if (result.size != lowered_h * lowered_w)
        throw std::invalid_argument("Matrix dimensions mismatch for im2row_nhwc.");

    if (result.isTransposed() || im.isTransposed())
        throw std::invalid_argument("Matrix must not be transposed for im2row_nhwc.");

    for (int cur_n = 0; cur_n < n; cur_n++) {
        SubMat cur_im(im, cur_n * h, 0, h, w * c);
        
        for (int cur_h = 0; cur_h < out_h; cur_h++) {
            for (int cur_w = 0; cur_w < out_w; cur_w++) {
                int row_idx = (cur_n * out_h * out_w + cur_h * out_w + cur_w) * lowered_w;
                for (int kh = 0; kh < k_h; kh++) {
                    for (int kw = 0; kw < k_w; kw++) {
                        for (int cc = 0; cc < c; cc++) {
                            int im_h = cur_h * s_h + kh - p_h;
                            int im_w = cur_w * s_w + kw - p_w;
                            float val = cur_im.get_element_fallback(im_h, im_w * c + cc, 0.0f);
                            result.data[row_idx++] = val;
                        }
                    }
                }
            }
        }
    }
}

void Mat::row2img_nhwc_additive(Mat& result, Mat& lowered, int n, int h, int w, int c, int k_h, int k_w, int s_h, int s_w, int p_h, int p_w) {
    int im_padded_h = h + 2 * p_h;
    int im_padded_w = w + 2 * p_w;

    int out_h = (im_padded_h - k_h) / s_h + 1;
    int out_w = (im_padded_w - k_w) / s_w + 1;

    int lowered_h = out_h * out_w * n;
    int lowered_w = k_h * k_w * c;

    if (lowered.getSize() != lowered_h * lowered_w)
        throw std::invalid_argument("Matrix dimensions mismatch for row2img_nhwc_additive.");

    if (result.getSize() != n * h * w * c)
        throw std::invalid_argument("Result matrix dimensions mismatch for row2img_nhwc_additive.");

    if (result.isTransposed() || lowered.isTransposed())
        throw std::invalid_argument("Matrices must not be transposed for row2img_nhwc_additive.");

    float* result_data = result.getData();
    float* lowered_data = lowered.getData();

    // Initialize result matrix with zeros
    // std::fill(result_data, result_data + result.getSize(), 0.0f);

    for (int cur_n = 0; cur_n < n; cur_n++) {
        for (int cur_h = 0; cur_h < out_h; cur_h++) {
            for (int cur_w = 0; cur_w < out_w; cur_w++) {
                int row_idx = (cur_n * out_h * out_w + cur_h * out_w + cur_w) * lowered_w;
                for (int kh = 0; kh < k_h; kh++) {
                    for (int kw = 0; kw < k_w; kw++) {
                        for (int cc = 0; cc < c; cc++) {
                            int im_h = cur_h * s_h + kh - p_h;
                            int im_w = cur_w * s_w + kw - p_w;
                            if (im_h >= 0 && im_h < h && im_w >= 0 && im_w < w) {
                                int result_idx = ((cur_n * h + im_h) * w + im_w) * c + cc;
                                result_data[result_idx] += lowered_data[row_idx];
                            }
                            row_idx++;
                        }
                    }
                }
            }
        }
    }
}

void Mat::im2col_nhwc(Mat& result, Mat& im, int n, int h, int w, int c, int k_h, int k_w, int s_h, int s_w, int p_h, int p_w) {
    int im_padded_h = h + 2 * p_h;
    int im_padded_w = w + 2 * p_w;

    int out_h = (im_padded_h - k_h) / s_h + 1;
    int out_w = (im_padded_w - k_w) / s_w + 1;

    int lowered_h = k_h * k_w * c;
    int lowered_w = out_h * out_w * n;

    if (result.size != lowered_h * lowered_w)
        throw std::invalid_argument("Matrix dimensions mismatch for im2col_nhwc.");

    if (result.isTransposed() || im.isTransposed())
        throw std::invalid_argument("Matrix must not be transposed for im2col_nhwc.");

    for (int cur_n = 0; cur_n < n; cur_n++) {
        SubMat cur_im(im, cur_n * h, 0, h, w * c);
        
        for (int cur_h = 0; cur_h < out_h; cur_h++) {
            for (int cur_w = 0; cur_w < out_w; cur_w++) {
                int col_idx = (cur_h * out_w + cur_w) * n + cur_n;
                for (int kh = 0; kh < k_h; kh++) {
                    for (int kw = 0; kw < k_w; kw++) {
                        for (int cc = 0; cc < c; cc++) {
                            int im_h = cur_h * s_h + kh - p_h;
                            int im_w = cur_w * s_w + kw - p_w;
                            float val = cur_im.get_element_fallback(im_h, im_w * c + cc, 0.0f);
                            result.data[(kh * k_w * c + kw * c + cc) * lowered_w + col_idx] = val;
                        }
                    }
                }
            }
        }
    }
}

void Mat::col2img_nhwc_additive(Mat& result, Mat& lowered, int n, int h, int w, int c, int k_h, int k_w, int s_h, int s_w, int p_h, int p_w) {
    int im_padded_h = h + 2 * p_h;
    int im_padded_w = w + 2 * p_w;

    int out_h = (im_padded_h - k_h) / s_h + 1;
    int out_w = (im_padded_w - k_w) / s_w + 1;

    int lowered_h = k_h * k_w * c;
    int lowered_w = out_h * out_w * n;

    if (lowered.getSize() != lowered_h * lowered_w)
        throw std::invalid_argument("Matrix dimensions mismatch for col2img_nhwc_additive.");

    if (result.getSize() != n * h * w * c)
        throw std::invalid_argument("Result matrix dimensions mismatch for col2img_nhwc_additive.");

    if (result.isTransposed() || lowered.isTransposed())
        throw std::invalid_argument("Matrices must not be transposed for col2img_nhwc_additive.");

    float* result_data = result.getData();
    float* lowered_data = lowered.getData();

    for (int cur_n = 0; cur_n < n; cur_n++) {
        for (int cur_h = 0; cur_h < out_h; cur_h++) {
            for (int cur_w = 0; cur_w < out_w; cur_w++) {
                int col_idx = (cur_h * out_w + cur_w) * n + cur_n;
                for (int kh = 0; kh < k_h; kh++) {
                    for (int kw = 0; kw < k_w; kw++) {
                        for (int cc = 0; cc < c; cc++) {
                            int im_h = cur_h * s_h + kh - p_h;
                            int im_w = cur_w * s_w + kw - p_w;
                            if (im_h >= 0 && im_h < h && im_w >= 0 && im_w < w) {
                                int result_idx = ((cur_n * h + im_h) * w + im_w) * c + cc;
                                result_data[result_idx] += lowered_data[(kh * k_w * c + kw * c + cc) * lowered_w + col_idx];
                            }
                        }
                    }
                }
            }
        }
    }
}

Mat Mat::view(int rows, int cols) {

    if (rows == -1 && cols == -1)
        throw std::invalid_argument("Mat::view - At least one of the dimensions must be specified for view.");

    if (is_transposed)
        throw std::invalid_argument("Matrix must not be transposed for view.");

    if (size != rows * cols)
        throw std::invalid_argument("Matrix dimensions mismatch for view.");

    // int _rows = rows == -1 ? size / cols : rows;
    // int _cols = cols == -1 ? size / rows : cols;

    Mat result(rows, cols, false);
    result.is_view = true;
    result.data = data;

    return result;
}

void Mat::view(const Mat& other) {

    if (other.is_transposed)
        throw std::invalid_argument("Matrix must not be transposed for view.");

    if (size != other.size)
        throw std::invalid_argument("Matrix dimensions mismatch for view.");

    if (is_view) {
        data = other.data;
    } else {
        delete[] data;
        data = other.data;
        is_view = true;
    }
}

void Mat::view(float* data) {
    if (is_view) {
        this->data = data;
    } else {
        delete[] this->data;
        this->data = data;
        is_view = true;
    }
}

void Mat::mec_lower(Mat& result, Mat& im, int n, int h, int w, int c, int k_h, int k_w, int s_h, int s_w, int p_h, int p_w) {
    int im_padded_h = h + 2 * p_h;
    int im_padded_w = w + 2 * p_w;

    int o_w = (im_padded_w - k_w) / s_w + 1;
    // int o_h = (im_padded_h - k_h) / s_h + 1;

    if (result.size != n * o_w * im_padded_h * k_w * c)
        throw std::invalid_argument("Matrix 'result' dimensions mismatch for mec_lower.");

    if (im.size != n * h * w * c)
        throw std::invalid_argument("Matrix 'im' dimensions mismatch for mec_lower.");

    float *res_data = result.getData();
    float *im_data = im.getData();

    if (p_h == 0 && p_w == 0) {

        // std::cout << "mec_lower: Padding is 0" << std::endl;
        int img_size = h * w * c;

        for (int nn = 0; nn < n; nn++) {
            float* cur_im_data = im_data + nn * img_size;
            for (int ww = 0; ww < o_w; ww++) {
                int col_idx = ww * s_w * c;
                for (int hh = 0; hh < h; hh++) {
                    std::memcpy(res_data, cur_im_data + col_idx + hh * w * c, k_w * c * sizeof(float));
                    res_data += k_w * c;
                }
            }
        }
    } else {

        // std::cout << "mec_lower: Padding is NOOOT 0" << std::endl;

        for (int nn = 0; nn < n; nn++) {
            SubMat cur_im(im, nn * h, 0, h, w * c);
            // float* cur_im_data = im_data + nn * h * w * c;
            for (int ww = 0; ww < o_w; ww++) {
                int col_idx = ww * s_w * c - p_w;
                float* cur_res_data = res_data + (nn * im_padded_h * k_w * c * o_w + ww * im_padded_h * k_w * c);
                for (int hh = -p_h; hh < h + p_h; hh++) {
                    for (int kw = 0; kw < k_w; kw++) {
                        int im_w = col_idx + kw * c;
                        for (int cc = 0; cc < c; cc++) {
                            float val = cur_im.get_element_fallback(hh, im_w + cc, 0.0f);
                            *cur_res_data = val;
                            cur_res_data++;
                        }
                    }
                }
            }
        }
    }
}

void Mat::mec_lower_separated(Mat& result, std::vector<std::shared_ptr<Mat>>& inputs, int n, int h, int w, int c, int k_h, int k_w, int s_h, int s_w) {
    int o_w = (w - k_w) / s_w + 1;
    // int o_h = (h - k_h) / s_h + 1;

    if (result.size != n * o_w * h * k_w * c)
        throw std::invalid_argument("Matrix 'result' dimensions mismatch for mec_lower.");

    if (inputs.size() != (size_t)n || inputs[0]->getSize() != h * w * c)
        throw std::invalid_argument("Mat::mec_lower_separated - Matrix 'im' dimensions mismatch for mec_lower.");

    float *res_data = result.getData();

    for (int nn = 0; nn < n; nn++) {
        float* cur_im_data = inputs[nn]->getData();
        for (int ww = 0; ww < o_w; ww++) {
            int col_idx = ww * s_w * c;
            for (int hh = 0; hh < h; hh++) {
                std::memcpy(res_data, cur_im_data + col_idx + hh * w * c, k_w * c * sizeof(float));
                res_data += k_w * c;
            }
        }
    }
}

void Mat::mec_lower_to_img_additive(Mat& im, Mat& lowered, int n, int h, int w, int c, int k_h, int k_w, int s_h, int s_w, int p_h, int p_w) {
    int im_padded_h = h + 2 * p_h;
    int im_padded_w = w + 2 * p_w;

    int o_w = (im_padded_w - k_w) / s_w + 1;
    // int o_h = (im_padded_h - k_h) / s_h + 1;

    if (lowered.getSize() != n * o_w * im_padded_h * k_w * c)
        throw std::invalid_argument("Matrix 'lowered' dimensions mismatch for mec_lower.");

    if (im.getSize() != n * h * w * c)
        throw std::invalid_argument("Matrix 'im' dimensions mismatch for mec_lower.");

    float* lowered_data = lowered.getData();
    float* im_data = im.getData();

    if (p_h == 0 && p_w == 0) {
        for (int nn = 0; nn < n; nn++) {
            float* cur_im_data = im_data + nn * h * w * c;
            for (int ww = 0; ww < o_w; ww++) {
                int col_idx = ww * s_w * c;
                for (int hh = 0; hh < h; hh++) {
                    for (int kw = 0; kw < k_w * c; kw++) {
                        cur_im_data[col_idx + hh * w * c + kw] += *lowered_data;
                        lowered_data++;
                    }
                }
                    
            }
        }
    } else {
        for (int nn = 0; nn < n; nn++) {
            SubMat cur_im(im, nn * h, 0, h, w * c);
            // float* cur_im_data = im_data + nn * h * w * c;
            for (int ww = 0; ww < o_w; ww++) {
                int col_idx = ww * s_w * c - p_w;
                float* cur_lowered_data = lowered_data + (nn * im_padded_h * k_w * c * o_w + ww * im_padded_h * k_w * c);
                for (int hh = -p_h; hh < h + p_h; hh++) {
                    for (int kw = 0; kw < k_w; kw++) {
                        int im_w = col_idx + kw * c;
                        for (int cc = 0; cc < c; cc++) {
                            float val = cur_im.get_element_fallback(hh, im_w + cc, 0.0f);
                            cur_im.put_ignore_overflow(val + *cur_lowered_data, hh, im_w + cc);
                            // cur_im.put_ignore_overflow(*cur_lowered_data, hh, im_w + cc);
                            cur_lowered_data++;
                        }
                    }
                }
            }
        }
    }
}

void Mat::maxpool_hnwc_to_nhwc(Mat& result, Mat& im, int n, int h, int w, int c, int k_h, int k_w, int s_h, int s_w, int* indeces) {
    const int o_h = (h - k_h) / s_h + 1;
    const int o_w = (w - k_w) / s_w + 1;

    if (result.getSize() != n * o_h * o_w * c)
        throw std::invalid_argument("Matrix 'result' dimensions mismatch for maxpool_hnwc.");

    if (im.getSize() != n * h * w * c)
        throw std::invalid_argument("Matrix 'im' dimensions mismatch for maxpool_hnwc.");

    float maxes[c];
    int indeces_[c];

    float* res_data = result.getData();
    float* im_data = im.getData();

    const int im_down = im.getDown();

    const float minus_inf = -std::numeric_limits<float>::infinity();

    for (int nn = 0; nn < n; nn++) {
        float* start_inputs_n = im_data + nn * w * c;

        for (int hh = 0; hh < o_h; hh++) {
            float* start_inputs = start_inputs_n + hh * s_h * im_down;

            for (int ww = 0; ww < o_w; ww++) {
                float* start_inputs_w = start_inputs + ww * s_w * c;

                for (int i = 0; i < c; i++)
                    maxes[i] = minus_inf;

                for (int mw = 0; mw < k_w; mw++) {
                    float* start_inputs_s = start_inputs_w + mw * c;

                    for (int mh = 0; mh < k_h; mh++) {
                        start_inputs_s += mh * im_down;
                        const int cached = start_inputs_s - im_data;

                        for (int cc = 0; cc < c; cc++) {
                            if (start_inputs_s[cc] >= maxes[cc]) {
                                maxes[cc] = start_inputs_s[cc]; 
                                indeces_[cc] = cached + cc;
                            }
                        }
                    }
                }

                std::memcpy(res_data, maxes, sizeof(float) * c);
                std::memcpy(indeces, indeces_, sizeof(int) * c);
                res_data += c;
                indeces += c;
            }
        }
    }

    return;
}

void Mat::mec_conv(Mat& result, Mat& lowered, Mat& kernels, int n, int h, int w, int c, int k_h, int k_w, int s_h, int s_w) {
    int o_h = (h - k_h) / s_h + 1;
    int o_w = (w - k_w) / s_w + 1;

    if (result.getSize() != n * o_h * o_w * kernels.getCols())
        throw std::invalid_argument("Matrix 'result' dimensions mismatch for mec_mm.");

    if (lowered.getSize() != n * o_w * h * k_w * c)
        throw std::invalid_argument("Matrix 'lowered' dimensions mismatch for mec_mm.");

    if (kernels.getRows() != k_h * k_w * c)
        throw std::invalid_argument("Matrix 'kernels' dimensions mismatch for mec_mm.");

    int lowered_rows = lowered.getRows();
    int kc = kernels.getCols();

    // #pragma omp parallel for
    for (int h = 0; h < o_h; h++) {
        float* lowered_start = lowered.getData() + h * s_h * k_w * c;
        float* result_start = result.getData() + h * o_w * kc * n;

        cblas_sgemm(CblasRowMajor, 
                    CblasNoTrans, CblasNoTrans,
                    lowered_rows, kc, kernels.getRows(),
                    1.0f, lowered_start, lowered.getCols(),
                    kernels.getData(), kernels.getCols(),
                    0.0f, result_start, kernels.getCols());
    }
}

void Mat::mec_back_into_lowered(Mat& result, Mat& grad, Mat& kernels, int n, int h, int w, int c, int k_h, int k_w, int s_h, int s_w) {
    int o_h = (h - k_h) / s_h + 1;
    int o_w = (w - k_w) / s_w + 1;

    int k_c = kernels.getCols();

    float* grad_data = grad.getData();
    float* result_data = result.getData();
    float* kernels_data = kernels.getData();

    for (int h = 0; h < o_h; h++) {

        // float* other_start = other->mat->data + h * stride * C_in * kw;
        float* next_grad_start = grad_data + h * o_w * k_c * n;
        float* result_start = result_data + h * s_h * k_w * c;

        cblas_sgemm(CblasRowMajor,
                CblasNoTrans, CblasTrans,
                result.getRows(), kernels.getRows(), k_c,
                1.0, next_grad_start, k_c, kernels_data, k_c,
                1.0, result_start, result.getCols());
    }
}
