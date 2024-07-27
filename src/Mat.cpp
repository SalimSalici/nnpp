#include "Mat.h"

#include <math.h>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <functional>

Mat::Mat(int rows, int cols) : rows(rows), cols(cols), size(rows * cols), is_transposed(false), right(1), down(cols) {
    data = new float[size];
}

// Copy constructor
Mat::Mat(const Mat& other)
    : rows(other.rows), cols(other.cols), size(other.size),
        is_transposed(other.is_transposed), right(other.right), down(other.down) {
    std::cout << "Copy constructor of Mat called";
    data = new float[size];
    std::memcpy(data, other.data, size * sizeof(float));
}

// Move constructor
Mat::Mat(Mat&& other) noexcept
    : rows(other.rows), cols(other.cols), size(other.size),
        is_transposed(other.is_transposed), right(other.right), down(other.down) {
    std::cout << "Move constructor of Mat called";
    data = other.data;
    other.data = nullptr;
}

// Copy assignment
Mat& Mat::operator=(const Mat& other) {
    std::cout << "Copy assignment of Mat called";
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
    std::cout << "Move assignment of Mat called";
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

Mat::~Mat() { delete[] data; }

void Mat::print() const {
    std::cout << "Rows: " << rows << ", Cols: " << cols << ", Transposed: " << is_transposed << "\n";
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) std::cout << data[r * cols + c] << " ";
        std::cout << "\n";
    }
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
    ) throw std::invalid_argument("Matrix mismatch.");
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
        throw std::invalid_argument("Matrix mismatch.");

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

void Mat::plus_tr_supp(Mat& result, const Mat& a, const Mat& b) {
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
        throw std::invalid_argument("Matrix mismatch.");

    float* r_data = result.getData();
    float* a_data = a.getData();
    float* b_data = b.getData();

    for (int r = 0; r < r_rows; r++) {
        float *r_curr = r_data + r * r_down;
        float *a_curr = a_data + r * a_down;
        float *b_curr = b_data + r * b_down;
        for (int c = 0; c < r_cols; c++) {
            *r_curr = *a_curr + *b_curr;
            r_curr += r_right;
            a_curr += a_right;
            b_curr += b_right;
        }
    }
}

Mat Mat::operator+(const Mat& other) const {
    if (rows != other.rows || cols != other.cols)
        throw std::invalid_argument("Matrix mismatch.");

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
    ) throw std::invalid_argument("Matrix mismatch.");
    for (int i = 0; i < result.size; i++) result.data[i] = a.data[i] - b.data[i];
}

Mat Mat::operator-(const Mat& other) const {
    if (rows != other.rows || cols != other.cols)
        throw std::invalid_argument("Matrix mismatch.");

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
    ) throw std::invalid_argument("Matrix mismatch.");
    for (int i = 0; i < result.size; i++) result.data[i] = a.data[i] * b.data[i];
}

Mat Mat::operator*(const Mat& other) const {
    if (rows != other.rows || cols != other.cols)
        throw std::invalid_argument("Matrix mismatch.");

    Mat result(rows, cols);
    Mat::hadamardProduct(result, *this, other);
    return result;
}

Mat& Mat::operator+=(const Mat& other) {
    if (rows != other.rows || cols != other.cols)
        throw std::invalid_argument("Matrix mismatch.");

    for (int i = 0; i < size; i++) data[i] += other.data[i];
    return *this;
}

Mat& Mat::operator-=(const Mat& other) {
    if (rows != other.rows || cols != other.cols)
        throw std::invalid_argument("Matrix mismatch.");

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

void Mat::matmul(Mat& result, const Mat& a, const Mat& b) {
    int a_rows = a.getRows();
    int a_cols = a.getCols();
    int b_rows = b.getRows();
    int b_cols = b.getCols();

    if (a_cols != b_rows || result.rows != a_rows || result.cols != b_cols) {
        throw std::invalid_argument("Matrix dimensions mismatch for multiplication.");
    }

    for (int i = 0; i < a_rows; ++i) {
        for (int j = 0; j < b_cols; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < a_cols; ++k) {
                float a_val = a.getElement(i, k);
                float b_val = b.getElement(k, j);
                sum += a_val * b_val;
            }
            result.put(sum, i, j);
        }
    }
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