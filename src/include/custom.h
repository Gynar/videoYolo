/*
 * custom.h
 *
 *  Created on: Mar 13, 2018
 *      Author: sdsoc
 */

#ifndef CUSTOM_H_
#define CUSTOM_H_

#ifdef __cplusplus
extern "C" {
#endif

/* */
typedef struct pix_rgb{
	unsigned char r;
	unsigned char g;
	unsigned char b;
}pix_rgb;

typedef struct pix_ycc{
	unsigned char y;
	unsigned char cb;
	unsigned char cr;
}pix_ycc;

/* Maximum image size */
#define MAX_WIDTH  2048 //1920
#define MAX_HEIGHT 1080
#define CAM_WIDTH  1280
#define CAM_HEIGHT 1024

/* Helper macros */
#define clamp(x) ((x) > 255) ? 255 : ((x) < 0) ? 0 : (x)

void sds_im2lay(unsigned short *img_in, float *input_layer, int rows, int cols, int stride);
void sds_lay2im(float *output_layer, unsigned short *img_out, int rows, int cols, int stride);

#ifdef __cplusplus
}
#endif

#endif /* CUSTOM_H_ */
