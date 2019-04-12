#include <cuda.h>
#include <iostream>

extern "C"{
#include "gifenc/gifenc.h"
}

__global__ void add(float* A, float* B, float* C, int n)
{
	int i = threadIdx.x;

	if (i < n) C[i] = A[i] + B[i];
}


// x col, y row
__global__ void compute_state(int* board, int* new_board, int n){
	int x = threadIdx.x;
	int y = threadIdx.y;
	int a, b;
	int acc = 0;
	
	int states[9] = {0, 0, 0, 1, 0, 0, 0, 0, 0};

	states[2] = board[n * y + x];
	
	for(int c = -1; c < 2; c++){
	for(int r = -1; r < 2; r++){
		a = x + c < 0 ? c + n : (x + c) % n;
		b = y + r < 0 ? r + n : (y + r) % n;
		acc += board[n * b + a];
	}}
	acc -= board[n * y + x];

	new_board[n * y + x] = states[acc];
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
	int n = 256;
       	int i, j;
	int frames = 60;

	size_t size = n * n * sizeof(int);

	int* board;
	int* d_board;
	int* d_new_board;
	int* temp;

	int tPB = 256;
	int blocks = (n - 1) / tPB + 1;


	uint8_t pal[] = {0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF};
	ge_GIF *gif = ge_new_gif("example.gif", n, n, pal, 1, 0);

	board = (int *) calloc(n * n, sizeof(int));

	board[256 * 0 + 3] = 1;
	board[256 * 1 + 3] = 1;
      	board[256 * 1 + 4] = 1;
	board[256 * 1 + 2] = 1;
	board[256 * 2 + 3] = 1;	

	// initial board state
	for (j = 0; j < n * n; j++){gif->frame[j] = board[j];}

	print_cuda_errors(cudaMallocHost(&d_board, size));
	print_cuda_errors(cudaMallocHost(&d_new_board, size));
		
	print_cuda_errors(cudaMemcpy(d_board, board, size, cudaMemcpyHostToDevice));
	
	for (i = 1; i < frames; i++){ 

		compute_state<<<blocks, tPB>>>(d_board, d_new_board, n);
		print_cuda_errors(cudaMemcpy(board, d_new_board, size, cudaMemcpyDeviceToHost));

		for (j = 0; j < n * n; j++){
			gif->frame[j] = board[j];
		}
		ge_add_frame(gif, 25);

		temp = d_board;
		d_board = d_new_board;
		d_new_board = temp;
	}

	ge_close_gif(gif);

	free(board);
	cudaFree(d_board);
	cudaFree(d_new_board);

	return 0;
}

int main2(){
	
	int i, j;
	int frames = 50;
	int n = 256;
	int *board, *new_board, *temp;

	board = (int *) calloc(n * n, sizeof(int));
	new_board = (int *) calloc(n * n, sizeof(int));

	uint8_t pal[] = {0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF}; 
	ge_GIF *gif = ge_new_gif("example.gif", n, n, pal, 1, 0);

	board[255 * 0 + 3] = 1U;
	board[255 * 1 + 3] = 1U;
	board[255 * 1 + 4] = 1U;
	board[255 * 1 + 2] = 1U;
	board[255 * 2 + 3] = 1U;

	for (i = 0; i < frames; i++){ 

		for (j = 0; j < n * n; j++){
			gif->frame[j] = new_board[j];
		}
	
		ge_add_frame(gif, 25);
	
		temp = board;
		board = new_board;
		new_board = temp;

	}

	ge_close_gif(gif);

	return 0;
}

