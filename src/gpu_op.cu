#include "./c_runtime_api.h"
#include <cassert>
#include <cstdio>
#include <cuda_runtime.h>

/* TODO: Your code here */
const int THREADS_PER_BLOCK = 1024;

/* all your GPU kernel code, e.g. matrix_softmax_cross_entropy_kernel */

__global__ void array_set_kernel(float *arr, float value, int n){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id < n) arr[id] = value;
}
int DLGpuArraySet(DLArrayHandle arr, float value) { 
  int n = 1;
  for(int i = 0; i < arr->ndim; i++) n *= arr->shape[i];
  int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  array_set_kernel<<<blocks, THREADS_PER_BLOCK>>>((float*)arr->data, value, n);
  return 0;
}

__global__ void broadcast_to_kernel(const float *a, float *b, int siz_a, int siz_b){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id < siz_b) b[id] = a[id % siz_a];
}
int DLGpuBroadcastTo(const DLArrayHandle input, DLArrayHandle output) {
  int siz_in = 1, siz_out = 1;
  for(int i = 0; i < input->ndim; i++) siz_in *= input->shape[i];
  for(int i = 0; i < output->ndim; i++) siz_out *= output->shape[i];
  int blocks = (siz_out + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  broadcast_to_kernel<<<blocks, THREADS_PER_BLOCK>>>((const float*)input->data, (float*)output->data, siz_in, siz_out);
  return 0;
}

__global__ void reduce_sum_axis_zero_kernel(const float*a, float *b, int nrow, int siz){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id < siz){
    b[id] = 0;
    for(int i = 0; i < nrow; i++) b[id] += a[i * siz + id];
  }
}
int DLGpuReduceSumAxisZero(const DLArrayHandle input, DLArrayHandle output) {
  int siz = 1;
  for(int i = 1; i < input->ndim; i++) siz *= input->shape[i];
  int blocks = (siz + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  reduce_sum_axis_zero_kernel<<<blocks, THREADS_PER_BLOCK>>>((const float*)input->data, (float*)output->data, input->shape[0], siz);
  return 0;
}

__global__ void matrix_elementwise_add_kernel(const float *a, const float *b, float *c, int n){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id < n) c[id] = a[id] + b[id];
}
int DLGpuMatrixElementwiseAdd(const DLArrayHandle matA,
                              const DLArrayHandle matB, DLArrayHandle output) {
  int n = 1;
  for(int i = 0; i < matA->ndim; i++) n *= matA->shape[i];
  int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  matrix_elementwise_add_kernel<<<blocks, THREADS_PER_BLOCK>>>((const float*)matA->data, (const float*)matB->data, (float*)output->data, n);
  return 0;
}

__global__ void matrix_elementwise_add_by_const_kernel(const float *a, float *b, int n, float val){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id < n) b[id] = a[id] + val;
}
int DLGpuMatrixElementwiseAddByConst(const DLArrayHandle input, float val,
                                     DLArrayHandle output) {
  int n = 1;
  for(int i = 0; i < input->ndim; i++) n *= input->shape[i];
  int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  matrix_elementwise_add_by_const_kernel<<<blocks, THREADS_PER_BLOCK>>>((const float*)input->data, (float*)output->data, n, val);
  return 0;
}

__global__ void matrix_elementwise_multiply_kernel(const float *a, const float *b, float *c, int n){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id < n) c[id] = a[id] * b[id];
}
int DLGpuMatrixElementwiseMultiply(const DLArrayHandle matA,
                                   const DLArrayHandle matB,
                                   DLArrayHandle output) {
  int n = 1;
  for(int i = 0; i < matA->ndim; i++) n *= matA->shape[i];
  int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  matrix_elementwise_multiply_kernel<<<blocks, THREADS_PER_BLOCK>>>((const float*)matA->data, (const float*)matB->data, (float*)output->data, n);
  return 0;
}

__global__ void matrix_elementwise_multiply_by_const_kernel(const float *a, float *b, int n, float val){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id < n) b[id] = a[id] * val;
}
int DLGpuMatrixMultiplyByConst(const DLArrayHandle input, float val,
                               DLArrayHandle output) {
  int n = 1;
  for(int i = 0; i < input->ndim; i++) n *= input->shape[i];
  int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  matrix_elementwise_multiply_by_const_kernel<<<blocks, THREADS_PER_BLOCK>>>((const float*)input->data, (float*)output->data, n, val);
  return 0;
}

__global__ void matrix_multiply_kernel(const float *a, const float *b, float *c, int row_a, int col_a, int row_b, int col_b, int n, bool tA, bool tB){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id < n){
    c[id] = 0;
    int row_c = id / (tB ? row_b : col_b);
    int col_c = id % (tB ? row_b : col_b);
    for(int i = 0, ii = tA ? row_a : col_a; i < ii; i++){
      c[id] += a[tA ? i * col_a + row_c : row_c * col_a + i] * b[tB ? col_c * col_b + i : i * col_b + col_c];
    }
  }
} 
int DLGpuMatrixMultiply(const DLArrayHandle matA, bool transposeA,
                        const DLArrayHandle matB, bool transposeB,
                        DLArrayHandle matC) {
  int n = 1;
  for(int i = 0; i < matC->ndim; i++) n *= matC->shape[i];
  int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  matrix_multiply_kernel<<<blocks, THREADS_PER_BLOCK>>>((const float*)matA->data, (const float*)matB->data, (float*)matC->data, 
                                                          matA->shape[0], matA->shape[1], matB->shape[0], matB->shape[1], n,
                                                          transposeA, transposeB);
  return 0;
}

__global__ void relu_kernel(const float *a, float *b, int n){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id < n) b[id] = a[id] > 0 ? a[id] : 0;
}
int DLGpuRelu(const DLArrayHandle input, DLArrayHandle output) {
  int n = 1;
  for(int i = 0; i < input->ndim; i++) n *= input->shape[i];
  int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  relu_kernel<<<blocks, THREADS_PER_BLOCK>>>((const float*)input->data, (float*)output->data, n);
  return 0;
}

__global__ void relu_gradient_kernel(const float *a, const float *b, float *c, int n){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id < n) c[id] = a[id] > 0 ? b[id] : 0;
}
int DLGpuReluGradient(const DLArrayHandle input, const DLArrayHandle in_grad,
                      DLArrayHandle output) {
  int n = 1;
  for(int i = 0; i < input->ndim; i++) n *= input->shape[i];
  int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  relu_gradient_kernel<<<blocks, THREADS_PER_BLOCK>>>((const float*)input->data, (const float*)in_grad->data, (float*)output->data, n);
  return 0;
}

__global__ void softmax_kernel(const float *a, float *b, int nrow, int ncol){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id < nrow){
    a += ncol * id;
    b += ncol * id;
    float sum = 0, maxval = *a;;
    for(int i = 0; i < ncol; i++) maxval = max(maxval, a[i]);
    for(int i = 0; i < ncol; i++) sum += exp(a[i] - maxval);
    for(int i = 0; i < ncol; i++) b[i] = exp(a[i] - maxval) / sum;
  }
}
int DLGpuSoftmax(const DLArrayHandle input, DLArrayHandle output) {
  int nrow = input->shape[0], ncol = input->shape[1];
  int blocks = (nrow + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  softmax_kernel<<<blocks, THREADS_PER_BLOCK>>>((const float*)input->data, (float *)output->data, nrow, ncol);
  return 0;
}

// y = inputs[0], y_ = inputs[1]
// np.mean(-np.sum(y_ * np.log(softmax(y)), axis=1), keepdims=True)
__global__ void matrix_softmax_cross_entropy_kernel(int nrow, int ncol,
                                                    const float *input_a,
                                                    const float *input_b,
                                                    float *output) {
  // Dynamic shared memory, size provided at kernel launch.
  extern __shared__ float loss_per_row[];
  // Two dimensional thread blocks.
  int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
          threadIdx.x;
  if (y >= nrow) {
    return;
  }
  input_a += y * ncol;
  input_b += y * ncol;
  float maxval = *input_a;
  // Find max for a row.
  for (int x = 1; x < ncol; ++x) {
    maxval = max(maxval, input_a[x]);
  }
  // Deduct by max for a row, and raise to exp.
  float sum = 0;
  for (int x = 0; x < ncol; ++x) {
    sum += exp(input_a[x] - maxval);
  }
  // Compute per-row loss.
  float loss = 0;
  for (int x = 0; x < ncol; ++x) {
    loss -= input_b[x] * log(exp(input_a[x] - maxval) / sum);
  }
  loss_per_row[y] = loss;
  __syncthreads();
  // Compute reduce_mean across rows.
  float mean_loss = 0;
  // Use a single thread to reduce mean across rows.
  if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
    for (int i = 0; i < nrow; ++i) {
      mean_loss += loss_per_row[i];
    }
    mean_loss /= nrow;
    output[0] = mean_loss;
  }
}
int DLGpuSoftmaxCrossEntropy(const DLArrayHandle input_a,
                             const DLArrayHandle input_b,
                             DLArrayHandle output) {
  assert(input_a->ndim == 2);
  assert(input_b->ndim == 2);
  assert(output->ndim == 1);
  assert(input_a->shape[0] == input_b->shape[0] &&
         input_a->shape[1] == input_b->shape[1]);
  int nrow = input_a->shape[0];
  // Maximum x- or y-dimension of a block = 1024
  // But we need 'nrow' shared memory, and max shared memory is 48KB.
  // Conservatively allow max 16KB shared memory.
  assert(nrow <= 1024 * 4);
  int ncol = input_a->shape[1];
  const float *input_data_a = (const float *)input_a->data;
  const float *input_data_b = (const float *)input_b->data;
  float *output_data = (float *)output->data;
  dim3 threads;
  if (nrow <= 1024) {
    threads.x = nrow;
  } else {
    threads.x = 1024;
    threads.y = (nrow + 1023) / 1024;
  }
  // 1 block, each block with 'threads' number of threads with 'nrow' shared
  // memory size
  matrix_softmax_cross_entropy_kernel<<<1, threads, nrow * sizeof(float)>>>(
      nrow, ncol, input_data_a, input_data_b, output_data);
  return 0;
}
