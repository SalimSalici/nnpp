#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>
#include <algorithm>
#include <random>

template <typename T>
void shuffleArray(T* array, size_t size) {
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(array, array + size, g);
}

#endif