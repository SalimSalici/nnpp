#include "Mat.h"

#include <math.h>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <functional>

extern "C" {
#include <cblas.h>
}

Mat::Mat(int rows, int cols, bool alloc_data) : rows(rows), cols(cols), size(rows * cols), is_transposed(false), right(1), down(cols), is_view(false) {
    if (alloc_data)
        data = new float[size];
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
        throw std::invalid_argument("Matrix mismatch.");

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

void Mat::hadamardProduct_keep_res(Mat& result, const Mat& a, const Mat& b, float result_scaling) {
    if (result.isTransposed() || a.isTransposed() || b.isTransposed()) {
        Mat::element_op_tr_supp_keep_res(result, a, b, result_scaling, std::multiplies<float>());
        return;
    }

    if (
        result.rows != a.rows || result.cols != a.cols ||
        result.rows != b.rows || result.cols != b.cols
    ) throw std::invalid_argument("Matrix mismatch.");
    for (int i = 0; i < result.size; i++) result.data[i] = a.data[i] * b.data[i] + result.data[i] * result_scaling;
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

    if (a_cols != b_rows || result.rows != a_rows || result.cols != b_cols)
        throw std::invalid_argument("Matrix dimensions mismatch for multiplication.");

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

Mat Mat::view(int rows, int cols) {

    if (rows == -1 && cols == -1)
        throw std::invalid_argument("Mat::view - At least one of the dimensions must be specified for view.");

    if (is_transposed)
        throw std::invalid_argument("Matrix must not be transposed for view.");

    if (size != rows * cols)
        throw std::invalid_argument("Matrix dimensions mismatch for view.");

    int _rows = rows == -1 ? size / cols : rows;
    int _cols = cols == -1 ? size / rows : cols;

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