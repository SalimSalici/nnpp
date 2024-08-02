#ifndef MAT_H
#define MAT_H

#include <functional>

class Mat {
   public:
    Mat(int rows, int cols, bool alloc_data = true);
    // Copy constructor
    Mat(const Mat& other);
    // Move constructor
    Mat(Mat&& other) noexcept;
    // Copy assignment
    Mat& operator=(const Mat& other);
    // Move assignment
    Mat& operator=(Mat&& other);
    ~Mat();
    void print() const;
    Mat& copy_from(const float* other_data);
    Mat& put(float value, int row, int col);
    Mat& zero();
    Mat& fill(float value);
    Mat& fill_rand_rate(float value_p, float value_not_p, float p);
    Mat& transpose();
    Mat& operator+=(const Mat& other);
    Mat& operator-=(const Mat& other);
    Mat& apply(float (*act)(float));
    Mat& raiseEach(int power);
    Mat& raiseEach(float power);
    float elementsSum() const;

    // returns a new Mat object that is a view of the current Mat object
    Mat view(int rows, int cols);
    
    // changes the current Mat object to be a view of another Mat object
    void view(const Mat& other);

    Mat operator+(const Mat& other) const;
    Mat operator-(const Mat& other) const;
    Mat operator*(const Mat& other) const;
    float operator[](int idx) const;
    float getElement(int row, int col) const;
    static Mat matmul(const Mat& a, const Mat& b);
    static Mat apply(const Mat& a, float (*act)(float));
    static Mat pow(const Mat& a, int power);
    static Mat pow(const Mat& a, float power);
    static Mat scale(const Mat& a, float factor);

    static void element_op_tr_supp(Mat& result, const Mat& a, const Mat& b, std::function<float(float, float)> op);
    
    // result = a (op) b + result * result_scaling
    static void element_op_tr_supp_keep_res(Mat& result, const Mat& a, const Mat& b, float result_scaling, std::function<float(float, float)> op);

    static void plus(Mat& result, const Mat& a, const Mat& b);
    static void minus(Mat& result, const Mat& a, const Mat& b);
    static void hadamardProduct(Mat& result, const Mat& a, const Mat& b);

    // result = a * b + result * result_scaling
    static void hadamardProduct_keep_res(Mat& result, const Mat& a, const Mat& b, float result_scaling);
    static void matmul(Mat& result, const Mat& a, const Mat& b);
    static void matmul_mm(Mat& result, const Mat& a, const Mat& b, float ab_s, float c_s);
    static void matmul_mv(Mat& result, const Mat& a, const Mat& b, float ab_s, float c_s);
    static void apply(Mat& result, const Mat& a, float (*act)(float));
    static void apply(Mat& result, const Mat& a, float (*act)(float, void*), void* args);
    static void apply_log(Mat& result, const Mat& a);
    static void pow(Mat& result, const Mat& a, int power);
    static void pow(Mat& result, const Mat& a, float power);
    static void scale(Mat& result, const Mat& a, float factor);
    static void softmax(Mat& result, const Mat& a, bool rows = false);
    static void mat_plus_scalar(Mat& result, const Mat& mat, float scalar, float mat_scaling);
    static void mat_plus_vec(Mat& result, const Mat& mat, const Mat& vec);
    static void vec_plus_mat(Mat& result, const Mat& vec, const Mat& mat);
    static void mat_plus_row_vec(Mat& result, const Mat& mat, const Mat& vec);
    static void row_vec_plus_mat(Mat& result, const Mat& vec, const Mat& mat);
    static void im2row_nhwc(Mat& result, const Mat& im, int n, int h, int w, int c, int k_h, int k_w, int s_h, int s_w, int p_h, int p_w);
    static void row2img_nhwc_additive(Mat& result, const Mat& lowered, int n, int h, int w, int c, int k_h, int k_w, int s_h, int s_w, int p_h, int p_w);
    static void im2col_nhwc(Mat& result, const Mat& im, int n, int h, int w, int c, int k_h, int k_w, int s_h, int s_w, int p_h, int p_w);
    static void col2img_nhwc_additive(Mat& result, const Mat& lowered, int n, int h, int w, int c, int k_h, int k_w, int s_h, int s_w, int p_h, int p_w);
    
    float* getData() const;
    int getRows() const;
    int getCols() const;
    int getSize() const;
    bool isTransposed() const;
    int getRight() const;
    int getDown() const;

   private:
    float* data;
    int rows, cols, size;
    bool is_transposed;
    int right;
    int down;
    bool is_view;
};

#endif  // MAT_H