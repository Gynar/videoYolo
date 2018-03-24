/*
 * sds_custom.cpp
 *
 *  Created on: Mar 14, 2018
 *      Author: sdsoc
 */

#define HLS_NO_XIL_FPO_LIB

#include "hls_video.h"
#include "custom.h"
#include <assert.h>

void RGB2YCbCr(pix_rgb* rgb, pix_ycc* ycc){
		//unsigned char R, unsigned char G, unsigned char B, unsigned char* pY, unsigned char* pCb, unsigned char* pCr){

	unsigned char tY = clamp(0.257*rgb->r + 0.504*rgb->g + 0.098*rgb->b + 16);
	unsigned char tCb = clamp(-0.148*rgb->r - 0.291*rgb->g + 0.439*rgb->b + 128);
	unsigned char tCr = clamp(0.439*rgb->r - 0.368*rgb->g - 0.071*rgb->b + 128);

	ycc->y = tY;
	ycc->cb = tCb;
	ycc->cr = tCr;
//	(*pY) = tY;
//	(*pCb) = tCb;
//	(*pCr) = tCr;
}

void YCbCr2RGB(pix_ycc* ycc, pix_rgb* rgb){
	//(unsigned char Y, unsigned char Cb, unsigned char Cr, unsigned char* pR, unsigned char* pG, unsigned char* pB){

	unsigned char tR = clamp(1.164*(ycc->y - 16) + 1.596*(ycc->cr - 128));
	unsigned char tG = clamp(1.164*(ycc->y - 16) - 0.813*(ycc->cr - 128) - 0.391*(ycc->cb - 128));
	unsigned char tB = clamp(1.164*(ycc->y - 16) + 2.018*(ycc->cb - 128));

	rgb->r = tR;
	rgb->g = tG;
	rgb->b = tB;
//	(*pR) = tR;
//	(*pG) = tG;
//	(*pB) = tB;
}

void sds_im2lay(unsigned short *img_in, float *input_layer, int rows, int cols, int stride){

	int row, col;

	pix_rgb rgb1;
	pix_rgb rgb2;
	pix_ycc ycc1;
	pix_ycc ycc2;

	int doCvt;

	for (row = 0; row < rows; row++) {
//#pragma HLS LOOP_TRIPCOUNT MAX = 1080
			for (col = 0; col < stride; col++) {
//#pragma HLS LOOP_TRIPCOUNT MAX = 2048
//#pragma HLS LOOP_FLATTEN OFF
//#pragma HLS PIPELINE II = 1

			if(row < CAM_HEIGHT && col < CAM_WIDTH){

				unsigned short val = img_in[(row)*stride+(col)];

				if((col%2)==0){
					ycc1.y = (unsigned char)(0x00ff & val);
					ycc1.cb = (unsigned char)((0xff00 & val) >> 8);
					ycc2.cb = ycc1.cb;
					doCvt = 0;
				}
				else{
					ycc2.y = (unsigned char)(0x00ff & val);
					ycc2.cr = (unsigned char)((0xff00 & val) >> 8);
					ycc1.cr = ycc2.cr;
					doCvt = 1;
				}

				if(doCvt){
					int r_idx1 = (CAM_WIDTH*CAM_HEIGHT)*0 + (CAM_WIDTH)*(row) + (col-1);
					int g_idx1 = (CAM_WIDTH*CAM_HEIGHT)*1 + (CAM_WIDTH)*(row) + (col-1);
					int b_idx1 = (CAM_WIDTH*CAM_HEIGHT)*2 + (CAM_WIDTH)*(row) + (col-1);

					int r_idx2 = (CAM_WIDTH*CAM_HEIGHT)*0 + (CAM_WIDTH)*(row) + (col);
					int g_idx2 = (CAM_WIDTH*CAM_HEIGHT)*1 + (CAM_WIDTH)*(row) + (col);
					int b_idx2 = (CAM_WIDTH*CAM_HEIGHT)*2 + (CAM_WIDTH)*(row) + (col);

					assert(r_idx1 >= 0 && r_idx1 < CAM_WIDTH*CAM_HEIGHT*3);
					assert(r_idx2 >= 0 && r_idx2 < CAM_WIDTH*CAM_HEIGHT*3);
					assert(g_idx1 >= 0 && g_idx1 < CAM_WIDTH*CAM_HEIGHT*3);
					assert(g_idx2 >= 0 && g_idx2 < CAM_WIDTH*CAM_HEIGHT*3);
					assert(b_idx1 >= 0 && b_idx1 < CAM_WIDTH*CAM_HEIGHT*3);
					assert(b_idx2 >= 0 && b_idx2 < CAM_WIDTH*CAM_HEIGHT*3);

					YCbCr2RGB(&ycc1, &rgb1);
					YCbCr2RGB(&ycc2, &rgb2);

					input_layer[r_idx1] = (float)rgb1.r/255.;
					input_layer[r_idx2] = (float)rgb2.r/255.;
					input_layer[g_idx1] = (float)rgb1.g/255.;
					input_layer[g_idx2] = (float)rgb2.g/255.;
					input_layer[b_idx1] = (float)rgb1.b/255.;
					input_layer[b_idx2] = (float)rgb2.b/255.;
				}
			}
		}
	}
}

void sds_lay2im(float *output_layer, unsigned short *img_out, int rows, int cols, int stride){

	int row, col;
	pix_rgb rgb;
	pix_ycc ycc;

	for (row = 0; row < rows; row++) {
//#pragma HLS LOOP_TRIPCOUNT MAX = 1080
			for (col = 0; col < stride; col++) {
//#pragma HLS LOOP_TRIPCOUNT MAX = 2048
//#pragma HLS LOOP_FLATTEN OFF
//#pragma HLS PIPELINE II = 1

			if(row < CAM_HEIGHT && col < CAM_WIDTH){
				//img_out[(row)*stride+(col)] = (0x80ff);

				rgb.r = clamp(output_layer[(CAM_WIDTH*CAM_HEIGHT)*0 + (CAM_WIDTH)*(row) + (col)]*255);
				rgb.g = clamp(output_layer[(CAM_WIDTH*CAM_HEIGHT)*1 + (CAM_WIDTH)*(row) + (col)]*255);
				rgb.b = clamp(output_layer[(CAM_WIDTH*CAM_HEIGHT)*2 + (CAM_WIDTH)*(row) + (col)]*255);

				RGB2YCbCr(&rgb, &ycc);

				if((col%2)==0)
					img_out[(row)*stride+(col)] = (ycc.cb << 8) | ycc.y;
				else
					img_out[(row)*stride+(col)] = (ycc.cr << 8) | ycc.y;
			}
//			else
//				img_out[(row)*stride+(col)] = (0x8000);
		}
	}
}




