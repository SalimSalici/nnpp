#ifndef SUBMAT_H
#define SUBMAT_H

#include "Mat.h"
#include <stdexcept>
#include <iostream>

class SubMat {
public:
    SubMat(Mat& matrix, int start_row, int start_col, int num_rows, int num_cols)
        : original_matrix(matrix), start_row(start_row), start_col(start_col), num_rows(num_rows), num_cols(num_cols) {
        
        // std::cout << "SubMat: " << start_row << " " << start_col << " " << num_rows << " " << num_cols << std::endl;
        // std::cout << "SubMat: " << matrix.getRows() << " " << matrix.getCols() << std::endl;
        
        if (start_row < 0 || start_col < 0 || 
            start_row + num_rows > matrix.getRows() || 
            start_col + num_cols > matrix.getCols()) {
            throw std::out_of_range("SubMat: Invalid submatrix dimensions");
        }
    }

    float get_element(int row, int col) const {
        if (row < 0 || row >= num_rows || col < 0 || col >= num_cols) {
            throw std::out_of_range("SubMat: Index out of range");
        }
        return original_matrix.getElement(start_row + row, start_col + col);
    }

    void put_ignore_overflow(float value, int row, int col) {
        if (row < 0 || row >= num_rows || col < 0 || col >= num_cols) {
            return;
        }
        original_matrix.put(value, start_row + row, start_col + col);
    }

    float get_element_fallback(int row, int col, float fallback) const {
        if (row < 0 || row >= num_rows || col < 0 || col >= num_cols)
            return fallback;
        return original_matrix.getElement(start_row + row, start_col + col);
    }

    void print() const {
        for (int i = 0; i < num_rows; ++i) {
            for (int j = 0; j < num_cols; ++j) {
                std::cout << get_element(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }

    int get_rows() const { return num_rows; }
    int get_cols() const { return num_cols; }

private:
    Mat& original_matrix;
    int start_row;
    int start_col;
    int num_rows;
    int num_cols;
};

#endif