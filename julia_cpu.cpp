%%writefile julia_cpu.cpp
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

#define DIM 1000

struct SimpleBitmap {
    int width, height;
    unsigned char *pixels;
    SimpleBitmap(int w, int h) : width(w), height(h) {
        pixels = (unsigned char *)malloc(image_size());
    }
    ~SimpleBitmap() { free(pixels); }
    size_t image_size() const { return (size_t)width * (size_t)height * 4; }
    unsigned char *get_ptr() { return pixels; }
    void save_ppm(const char *filename) const {
        FILE *fp = fopen(filename, "wb");
        if (!fp) return;
        fprintf(fp, "P6\n%d %d\n255\n", width, height);
        for (int i = 0; i < width * height; i++) {
            fwrite(&pixels[i * 4], 1, 3, fp);
        }
        fclose(fp);
    }
};

struct cuComplex {
    float r, i;
    cuComplex(float a, float b) : r(a), i(b) {}
    float magnitude2(void) const { return r * r + i * i; }
    cuComplex operator*(const cuComplex &a) const {
        return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
    }
    cuComplex operator+(const cuComplex &a) const {
        return cuComplex(r + a.r, i + a.i);
    }
};

int julia(int x, int y) {
    const float scale = 1.5f;
    float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
    float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);
    cuComplex c(-0.8f, 0.156f);
    cuComplex a(jx, jy);
    for (int i = 0; i < 200; i++) {
        a = a * a + c;
        if (a.magnitude2() > 1000.0f) return 0;
    }
    return 1;
}

void kernel(unsigned char *ptr) {
    for (int y = 0; y < DIM; y++) {
        for (int x = 0; x < DIM; x++) {
            int offset = x + y * DIM;
            int juliaValue = julia(x, y);
            ptr[offset * 4 + 0] = (unsigned char)(255 * juliaValue);
            ptr[offset * 4 + 1] = 0;
            ptr[offset * 4 + 2] = 0;
            ptr[offset * 4 + 3] = 255;
        }
    }
}

int main(void) {
    SimpleBitmap bitmap(DIM, DIM);
    unsigned char *ptr = bitmap.get_ptr();

    // Warmup run
    kernel(ptr);

    const int runs = 10;
    for (int r = 0; r < runs; r++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        kernel(ptr);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf("CPU_RUN,%d,%.4f\n", r + 1, ms);
    }

    bitmap.save_ppm("julia_cpu.ppm");
    return 0;
}
