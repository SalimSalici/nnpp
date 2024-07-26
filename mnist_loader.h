#ifndef _MNIST_LOADER_H
#define _MNIST_LOADER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

typedef struct MnistSample {
    float data[28*28];
    uint8_t label;
} MnistSample;

uint8_t* mnist_load_train_images_raw(const char* file_name);
float* mnist_load_train_images(const char* file_name);
MnistSample* mnist_load_samples(const char* data_file_name, const char* labels_file_name, size_t offset, size_t count, float black, float white);
void mnist_print_image(float* image);

#ifdef __cplusplus
}
#endif

#endif