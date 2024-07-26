#include "Mat.h"

#include <math.h>
#include <cstring>
#include <iostream>
#include <stdexcept>

Mat::Mat(int rows, int cols) : rows(rows), cols(cols), size(rows * cols) {
    data = new float[size];
}

// Copy constructor
Mat::Mat(const Mat& other)
    : rows(other.rows), cols(other.cols), size(other.size) {
    std::cout << "Copy constructor of Mat called";
    data = new float[size];
    std::memcpy(data, other.data, size * sizeof(float));
}

// Move constructor
Mat::Mat(Mat&& other) noexcept
    : rows(other.rows), cols(other.cols), size(other.size) {
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
    }
    return *this;
}

Mat::~Mat() { delete[] data; }

void Mat::print() const {
    std::cout << "Rows: " << rows << ", Cols: " << cols << "\n";
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) std::cout << data[r * cols + c] << " ";
        std::cout << "\n";
    }
}

int Mat::getRows() const {
    return rows;
}

int Mat::getCols() const {
    return cols;
}

int Mat::getSize() const {
    return size;
}

void Mat::copy_from(const float* other_data) {
    std::memcpy(data, other_data, size * sizeof(float));
}

void Mat::put(float value, int row, int col) {
    int i = row * cols + col;
    if (i >= size) throw std::invalid_argument("Index out of bounds.");
    data[i] = value;
}

void Mat::zero() { memset(data, 0, size * sizeof(float)); }

void Mat::fill(float value) { 
    for (int i = 0; i < size; i++)
        data[i] = value;
}

float* Mat::getData() const { return data; }

float Mat::elementsSum() const {
    float result = 0;
    for (int i = 0; i < size; i++) result += data[i];
    return result;
}

void Mat::plus(Mat& result, const Mat& a, const Mat& b) {
    if (
        result.rows != a.rows || result.cols != a.cols ||
        result.rows != b.rows || result.cols != b.cols
    ) throw std::invalid_argument("Matrix mismatch.");
    for (int i = 0; i < result.size; i++) result.data[i] = a.data[i] + b.data[i];
}

Mat Mat::operator+(const Mat& other) const {
    if (rows != other.rows || cols != other.cols)
        throw std::invalid_argument("Matrix mismatch.");

    Mat result(rows, cols);
    Mat::plus(result, *this, other);
    return result;
}

void Mat::minus(Mat& result, const Mat& a, const Mat& b) {
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
    int i = row * cols + col;
    if (i >= size) throw std::invalid_argument("Index out of bounds.");
    return data[i];
}

void Mat::matmul(Mat& result, const Mat& a, bool a_transpose, const Mat& b, bool b_transpose) {
    int a_rows = a_transpose ? a.cols : a.rows;
    int a_cols = a_transpose ? a.rows : a.cols;
    int b_rows = b_transpose ? b.cols : b.rows;
    int b_cols = b_transpose ? b.rows : b.cols;

    if (a_cols != b_rows || result.rows != a_rows || result.cols != b_cols) {
        throw std::invalid_argument("Matrix dimensions mismatch for multiplication.");
    }

    for (int i = 0; i < a_rows; ++i) {
        for (int j = 0; j < b_cols; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < a_cols; ++k) {
                float a_val = a_transpose ? a.getElement(k, i) : a.getElement(i, k);
                float b_val = b_transpose ? b.getElement(j, k) : b.getElement(k, j);
                sum += a_val * b_val;
            }
            result.put(sum, i, j);
        }
    }
}

Mat Mat::matmul(const Mat& a, bool a_transpose, const Mat& b, bool b_transpose) {
    int a_rows = a_transpose ? a.cols : a.rows;
    int b_cols = b_transpose ? b.rows : b.cols;

    Mat result(a_rows, b_cols);

    Mat::matmul(result, a, a_transpose, b, b_transpose);

    return result;
}

void Mat::matmul(Mat& result, const Mat& a, const Mat& b) {
    Mat::matmul(result, a, false, b, false);
}

Mat Mat::matmul(const Mat& a, const Mat& b) {
    return matmul(a, false, b, false);
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
            result.data[mat_idx] = mat.data[mat_idx] + vec.data[r];
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