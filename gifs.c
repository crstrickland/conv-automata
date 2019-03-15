#include <stdlib.h>
#include <stdio.h>
#include "gifenc/gifenc.h"

void iterate(int *board, int *new_board, int n){
	int i, j;
	int c, r;
	int acc;
	int x, y;

	for(i = 0; i < n; i++){
	for(j = 0; j < n; j++){
		acc = 0;
		for(c = -1; c < 2; c++){
		for(r = -1; r < 2; r++){
			x = i + c < 0 ? c + n : (i + c) % n;
			y = j + r < 0 ? r + n : (j + r) % n;
			acc += board[x * n + y];
		}}
		acc -= board[i * n + j];
		
		if (acc > 0) printf("acc value %d at %d, %d\n", acc, i, j);
		
		if (acc > 3) 		new_board[i * n + j] = 0U;
		else if (acc == 3) 	new_board[i * n + j] = 1U;
		else if (acc > 1) 	new_board[i * n + j] = board[i * n + j];
		else			new_board[i * n + j] = 0U;
	}}
}


int main(){
	
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
		iterate(board, new_board, n);

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

