%%writefile julia_gpu.cu
#include <stdio.h>
#include <stdlib.h>

#define DIM 1000

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))
static void HandleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

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
    __host__ __device__ cuComplex(float a, float b) : r(a), i(b) {}
    __host__ __device__ float magnitude2(void) const { return r * r + i * i; }
    __host__ __device__ cuComplex operator*(const cuComplex &a) const {
        return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
    }
    __host__ __device__ cuComplex operator+(const cuComplex &a) const {
        return cuComplex(r + a.r, i + a.i);
    }
};

__device__ int julia(int x, int y) {
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

__global__ void kernel(unsigned char *ptr) {
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x;
    int juliaValue = julia(x, y);
    ptr[offset * 4 + 0] = (unsigned char)(255 * juliaValue);
    ptr[offset * 4 + 1] = 0;
    ptr[offset * 4 + 2] = 0;
    ptr[offset * 4 + 3] = 255;
}

int main(void) {
    SimpleBitmap bitmap(DIM, DIM);
    unsigned char *dev_bitmap;
    HANDLE_ERROR(cudaMalloc((void **)&dev_bitmap, bitmap.image_size()));

    dim3 grid(DIM, DIM);
    cudaEvent_t start, stop, copy_stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventCreate(&copy_stop));

    // Warmup launch
    kernel<<<grid, 1>>>(dev_bitmap);
    HANDLE_ERROR(cudaDeviceSynchronize());

    const int runs = 10;
    for (int r = 0; r < runs; r++) {
        HANDLE_ERROR(cudaEventRecord(start));
        kernel<<<grid, 1>>>(dev_bitmap);
        HANDLE_ERROR(cudaEventRecord(stop));
        HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaEventRecord(copy_stop));
        HANDLE_ERROR(cudaEventSynchronize(copy_stop));

        float kernel_ms = 0.0f, total_ms = 0.0f;
        HANDLE_ERROR(cudaEventElapsedTime(&kernel_ms, start, stop));
        HANDLE_ERROR(cudaEventElapsedTime(&total_ms, start, copy_stop));
        printf("GPU_RUN,%d,%.4f,%.4f\n", r + 1, kernel_ms, total_ms);
    }

    bitmap.save_ppm("julia_gpu.ppm");
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));
    HANDLE_ERROR(cudaEventDestroy(copy_stop));
    HANDLE_ERROR(cudaFree(dev_bitmap));
    return 0;
}
