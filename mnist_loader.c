#include "mnist_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

// #define MNIST_SHADES_LEN 5
#define MNIST_SHADES_LEN 10

const int __mnist_image_size = 28*28;

// const char __mnist_loader_shades[MNIST_SHADES_LEN] = {'.', '-', 'o', '#', '@'};
const char __mnist_loader_shades[MNIST_SHADES_LEN] = {' ','.', ':', '-', '=', '+', '*', '#', '%', '@'};

uint32_t map_uint32(uint32_t in) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    return (
        ((in & 0xFF000000) >> 24) |
        ((in & 0x00FF0000) >>  8) |
        ((in & 0x0000FF00) <<  8) |
        ((in & 0x000000FF) << 24)
    );
#else
    return in;
#endif
}

uint8_t* mnist_load_train_images_raw(const char* file_name) {
    FILE* fileptr;
    fileptr = fopen(file_name, "rb");
    fseek(fileptr, 16, SEEK_SET);
    uint8_t* data = (uint8_t*)malloc(sizeof(uint8_t)*__mnist_image_size*60000);
    fread(data, 1, __mnist_image_size*60000, fileptr);
    fclose(fileptr);
    return data;
}

float* mnist_load_train_images(const char* file_name) {
    uint8_t* raw_data = mnist_load_train_images_raw(file_name);
    float* data = (float*)malloc(sizeof(float)*__mnist_image_size*60000);
    for (int i = 0; i < __mnist_image_size*60000; i++)
        data[i] = (float)raw_data[i] / 255.0;
    free(raw_data);
    return data;
}

MnistSample* mnist_load_samples(const char* data_file_name, const char* labels_file_name, size_t offset, size_t count, float black, float white) {

    uint8_t buffer[28*28];

    FILE* datafileptr = fopen(data_file_name, "rb");
    FILE* labelsfileptr = fopen(labels_file_name, "rb");

    fseek(datafileptr, 16 + offset*__mnist_image_size, SEEK_SET);
    fseek(labelsfileptr, 8 + offset, SEEK_SET);

    MnistSample* samples = (MnistSample*)malloc(sizeof(MnistSample) * count);

    for (size_t i = 0; i < count; i++) {
        fread(&buffer, sizeof(uint8_t)*28*28, 1, datafileptr);
        for (size_t j = 0; j < 28*28; j++) {
            samples[i].data[j] = (float)buffer[j] / 255.0;
            samples[i].data[j] = black + ((white - black) * samples[i].data[j]);
        }

        fread(&(samples[i].label), 1, 1, labelsfileptr);
    }

    fclose(datafileptr);
    fclose(labelsfileptr);

    return samples;
}

void mnist_print_image(float* image) {
    for (int i = 0; i < 28*28; i++) {
        int shade_idx = (int)round(image[i] * (float)(MNIST_SHADES_LEN - 1));
        char c = __mnist_loader_shades[shade_idx];
        printf("%c", c);
        if (i % 28 == 27)
            printf("\n");
    }
}