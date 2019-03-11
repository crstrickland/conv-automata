#include <cuda.h>
#include <iostream>

__global__ void add(float* A, float* B, float* C, int n)
{
	int i = threadIdx.x;

	if (i < n) C[i] = A[i] + B[i];
}


void print_cuda_errors(cudaError_t status)
{
	if (status != cudaSuccess)
	{
		std::cout << "error" << cudaGetErrorString(status) << std::endl;
	}
}



int main()
{
	size_t size = 4 * sizeof(float);

	float a[] = {0, 1, 2, 3};
	float b[] = {0, 1, 2, 3};

	float* c;

	print_cuda_errors(cudaMallocHost(&c, size));
	
	float* d_a;
	float* d_b;
	float* d_c;

	print_cuda_errors(cudaMalloc(&d_a, size));
	print_cuda_errors(cudaMalloc(&d_b, size));
	print_cuda_errors(cudaMalloc(&d_c, size));
	
	print_cuda_errors(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));
	print_cuda_errors(cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice));

	int tPB = 256;
	int bPG = (4 + tPB - 1) / tPB;

	add<<<1, 64>>>(d_a, d_b, d_c, 4);

	print_cuda_errors(cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost));

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	for (int i = 0; i < 4; i++)
	{
		std::cout << c[i] << " ";
	}
	std::cout << std::endl;

	cudaFree(c);

	return 0;
}
