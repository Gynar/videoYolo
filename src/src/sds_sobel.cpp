/***************************************************************************

*   © Copyright 2013 Xilinx, Inc. All rights reserved.

*   This file contains confidential and proprietary information of Xilinx,
*   Inc. and is protected under U.S. and international copyright and other
*   intellectual property laws.

*   DISCLAIMER
*   This disclaimer is not a license and does not grant any rights to the
*   materials distributed herewith. Except as otherwise provided in a valid
*   license issued to you by Xilinx, and to the maximum extent permitted by
*   applicable law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND WITH
*   ALL FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES AND CONDITIONS,
*   EXPRESS, IMPLIED, OR STATUTORY, INCLUDING BUT NOT LIMITED TO WARRANTIES
*   OF MERCHANTABILITY, NON-INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR
*   PURPOSE; and (2) Xilinx shall not be liable (whether in contract or
*   tort, including negligence, or under any other theory of liability)
*   for any loss or damage of any kind or nature related to, arising under
*   or in connection with these materials, including for any direct, or any
*   indirect, special, incidental, or consequential loss or damage (including
*   loss of data, profits, goodwill, or any type of loss or damage suffered
*   as a result of any action brought by a third party) even if such damage
*   or loss was reasonably foreseeable or Xilinx had been advised of the
*   possibility of the same.

*   CRITICAL APPLICATIONS
*   Xilinx products are not designed or intended to be fail-safe, or for use
*   in any application requiring fail-safe performance, such as life-support
*   or safety devices or systems, Class III medical devices, nuclear facilities,
*   applications related to the deployment of airbags, or any other applications
*   that could lead to death, personal injury, or severe property or environmental
*   damage (individually and collectively, "Critical Applications"). Customer
*   assumes the sole risk and liability of any use of Xilinx products in Critical
*   Applications, subject only to applicable laws and regulations governing
*   limitations on product liability.

*   THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS PART OF THIS FILE AT
*   ALL TIMES.

***************************************************************************/

#define HLS_NO_XIL_FPO_LIB

#include "hls_video.h"
#include "sobel.h"

typedef hls::Window<3, 3, unsigned char>              Y_WINDOW;
typedef hls::LineBuffer<3, MAX_WIDTH, unsigned char>  Y_BUFFER;

// Sobel Computation using a 3x3 neighborhood
unsigned char sobel_operator(Y_WINDOW *window)
{
	short x_weight = 0;
	short y_weight = 0;

	short edge_weight;
	unsigned char edge_val;

	char i;
	char j;

	const short x_op[3][3] = {
			{ 1,  0, -1},
			{ 2,  0, -2},
			{ 1,  0, -1}};

	const short y_op[3][3] = {
			{ 1,  2,  1},
			{ 0,  0,  0},
			{-1, -2, -1}};

	// Compute approximation of the gradients in the X-Y direction
	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			// X direction gradient
			x_weight = x_weight + (window->getval(i,j) * x_op[i][j]);
			// Y direction gradient
			y_weight = y_weight + (window->getval(i,j) * y_op[i][j]);
		}
	}

	edge_weight = ABS(x_weight) + ABS(y_weight);
	if (edge_weight < 255)
		edge_val = (255-(unsigned char)(edge_weight));
	else
		edge_val = 0;

	// Edge thresholding
	if (edge_val > HLS_SOBEL_HIGH_THRESH_VAL)
		edge_val = 255;
	else if (edge_val < HLS_SOBEL_LOW_THRESH_VAL)
		edge_val = 0;

	// Invert
	if (HLS_SOBEL_INVERT_VAL)
		edge_val = 255 - edge_val;

	return edge_val;
}

void sds_sobel(unsigned short *img_in, unsigned short *img_out, int rows, int cols, int stride)
{
	int row;
	int col;

	Y_BUFFER buff_A;
	Y_WINDOW buff_C;

	for (row = 0; row < rows+1; row++) {
#pragma HLS LOOP_TRIPCOUNT MAX = 1081
		for (col = 0; col < stride+1; col++) {
#pragma HLS LOOP_TRIPCOUNT MAX = 2049
#pragma HLS LOOP_FLATTEN OFF
#pragma HLS DEPENDENCE VARIABLE=&buff_A false
#pragma HLS PIPELINE II = 1

			// Temp values are used to reduce the number of memory reads
			unsigned char temp;
			unsigned char tempx;

			// Line Buffer fill
			if (col < stride) {
				buff_A.shift_down(col);
				temp = buff_A.getval(0,col);
			}

			// There is an offset to accomodate the active pixel region
			// There are only MAX_WIDTH and MAX_HEIGHT valid pixels in the image
			if (col < stride && row < rows) {
				tempx = img_in[row*stride+col];
				buff_A.insert_bottom(tempx,col);
			}

			// Shift the processing window to make room for the new column
			buff_C.shift_right();

			// The Sobel processing window only needs to store luminance values
			// rgb2y function computes the luminance from the color pixel
			if (col < stride) {
				buff_C.insert(buff_A.getval(2,col),2,0);
				buff_C.insert(temp,1,0);
				buff_C.insert(tempx,0,0);
			}
			unsigned char edge;

			// The sobel operator only works on the inner part of the image
			// This design assumes there are no edges on the boundary of the image
			if (row <= 1 || col <= 1 || row > (rows-1) || col > (cols-1)) {
				edge = 0;
			} else {
				// Sobel operation on the inner portion of the image
				edge = sobel_operator(&buff_C);
			}

			// The output image is offset from the input to account for the line buffer
			if (row > 0 && col > 0) {
				img_out[(row-1)*stride+(col-1)] = edge | (0x80<<8);
			}
		}
	}
}
