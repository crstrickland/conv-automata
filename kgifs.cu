#include <cuda.h>
#include <iostream>

extern "C"{
#include "gifenc/gifenc.h"
}

// x col, y row
__global__ void compute_state(int* board, int* new_board, int n){
	int x = threadIdx.x;
	int y = threadIdx.y;
	int a, b;
	int acc = 0;
	
	//int states[9] = {0, 0, 0, 1, 0, 0, 0, 0, 0};

	//states[2] = board[n * y + x];
	
	for(int c = -1; c < 2; c++){
	for(int r = -1; r < 2; r++){
		a = x + c < 0 ? c + n : (x + c) % n;
		b = y + r < 0 ? r + n : (y + r) % n;
		acc += board[n * b + a];
	}}
	acc -= board[n * y + x];

	//new_board[n * y + x] = states[acc];

	if (acc > 3) 		new_board[y * n + x] = 0;
	else if (acc == 3) 	new_board[y * n + x] = 1;
	else if (acc > 1) 	new_board[y * n + x] = board[n * y + x];
	else			new_board[y * n + x] = 0;

	if(acc > 0) printf("%d \n", new_board[y * n + x]);
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
	int n = 32;
       	int i, j;
	int frames = 60;

	size_t size = n * n * sizeof(int);

	int* board;
	int* d_board;
	int* d_new_board;
	int* temp;

	dim3 tPB(32, 32);
	dim3 nB((n - 1) / tPB.x + 1, (n - 1) / tPB.y + 1);

	uint8_t pal[] = {0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF};
	ge_GIF *gif = ge_new_gif("example.gif", n, n, pal, 1, 0);

	print_cuda_errors(cudaMallocHost(&board, size));
	print_cuda_errors(cudaMemset(board, 0, size));

	board[n * 1 + 3] = 1;
	board[n * 2 + 3] = 1;
      	board[n * 2 + 4] = 1;
	board[n * 2 + 2] = 1;
	board[n * 3 + 3] = 1;	

	board[n * 20 + 5] = 1;
	board[n * 21 + 6] = 1;
	board[n * 22 + 4] = 1;
	board[n * 22 + 5] = 1;
	board[n * 22 + 6] = 1;


	// add initial board state to the gif
	for (j = 0; j < n * n; j++){gif->frame[j] = board[j];}
	ge_add_frame(gif, 25);

	print_cuda_errors(cudaMalloc(&d_board, size));
	print_cuda_errors(cudaMalloc(&d_new_board, size));
		
	print_cuda_errors(cudaMemcpy(d_board, board, size, cudaMemcpyDefault));
	
	int live;

	for (i = 1; i < frames; i++){ 
		live = 0;
		
		compute_state<<<nB, tPB>>>(d_board, d_new_board, n);
		cudaDeviceSynchronize();
		print_cuda_errors(cudaMemcpy(board, d_new_board, size, cudaMemcpyDefault));

		for (j = 0; j < n * n; j++){
			live += board[j];
			gif->frame[j] = board[j];
		}
		ge_add_frame(gif, 25);
		std::cout << "added frame " << i << ", ";
		std::cout << live << " cells alive" << std::endl;
		// board is empty, but d_new_board points to the old board state
		
		temp = d_board;
		d_board = d_new_board;
		d_new_board = temp;
	}

	ge_close_gif(gif);

	print_cuda_errors(cudaFreeHost(board));
	cudaFree(d_board);
	cudaFree(d_new_board);

	return 0;
}

