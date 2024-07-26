#ifndef MAT_H
#define MAT_H

class Mat {
   public:
    Mat(int rows, int cols);
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
    void copy_from(const float* other_data);
    void put(float value, int row, int col);
    void zero();
    void fill(float value);
    Mat& operator+=(const Mat& other);
    Mat& operator-=(const Mat& other);
    Mat& apply(float (*act)(float));
    Mat& raiseEach(int power);
    Mat& raiseEach(float power);
    float elementsSum() const;
    Mat operator+(const Mat& other) const;
    Mat operator-(const Mat& other) const;
    Mat operator*(const Mat& other) const;
    float operator[](int idx) const;
    float getElement(int row, int col) const;
    static Mat matmul(const Mat& a, const Mat& b);
    static Mat matmul(const Mat& a, bool a_transpose, const Mat& b, bool b_transpose);
    static Mat apply(const Mat& a, float (*act)(float));
    static Mat pow(const Mat& a, int power);
    static Mat pow(const Mat& a, float power);
    static Mat scale(const Mat& a, float factor);

    static void plus(Mat& result, const Mat& a, const Mat& b);
    static void minus(Mat& result, const Mat& a, const Mat& b);
    static void hadamardProduct(Mat& result, const Mat& a, const Mat& b);
    static void matmul(Mat& result, const Mat& a, const Mat& b);
    static void matmul(Mat& result, const Mat& a, bool a_transpose, const Mat& b, bool b_transpose);
    static void apply(Mat& result, const Mat& a, float (*act)(float));
    static void apply(Mat& result, const Mat& a, float (*act)(float, void*), void* args);
    static void pow(Mat& result, const Mat& a, int power);
    static void pow(Mat& result, const Mat& a, float power);
    static void scale(Mat& result, const Mat& a, float factor);
    static void mat_plus_vec(Mat& result, const Mat& mat, const Mat& vec);
    static void vec_plus_mat(Mat& result, const Mat& vec, const Mat& mat);
    
    float* getData() const;
    int getRows() const;
    int getCols() const;
    int getSize() const;

   private:
    float* data;
    int rows, cols, size;
};

#endif  // MAT_H