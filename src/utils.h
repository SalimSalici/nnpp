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

int random_range_int(int min, int max) {
    int range = max - min;
    float rf = (float)rand() / RAND_MAX;

    return min + round(rf * (float)range);
}

void shuffle_pointers(void* array[], int count) {
    for (size_t i = 0; i < count - 2; i++) {
        size_t j = random_range_int(i, count - 1);
        
        void* tmp = array[i];
        array[i] = array[j];
        array[j] = tmp;
    }
}

#endif