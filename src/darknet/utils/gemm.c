#include "gemm.h"
#include "utils.h"
#include "cuda.h"
#include "hls_video.h"
#include <stdlib.h>
#include <sds_lib.h>
#include <stdio.h>
#include <math.h>

void gemm_bin(int M, int N, int K, float ALPHA, 
        char  *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            char A_PART = A[i*lda+k];
            if(A_PART){
                for(j = 0; j < N; ++j){
                    C[i*ldc+j] += B[k*ldb+j];
                }
            } else {
                for(j = 0; j < N; ++j){
                    C[i*ldc+j] -= B[k*ldb+j];
                }
            }
        }
    }
}

float *random_matrix(int rows, int cols)
{
    int i;
    float *m = calloc(rows*cols, sizeof(float));
    for(i = 0; i < rows*cols; ++i){
        m[i] = (float)rand()/RAND_MAX;
    }
    return m;
}

void time_random_matrix(int TA, int TB, int m, int k, int n)
{
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    int i;
    clock_t start = clock(), end;
    for(i = 0; i<10; ++i){
        gemm_cpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    }
    end = clock();
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf ms\n",m,k,k,n, TA, TB, (float)(end-start)/CLOCKS_PER_SEC);
    free(a);
    free(b);
    free(c);
}

void gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    gemm_cpu( TA,  TB,  M, N, K, ALPHA,A,lda, B, ldb,BETA,C,ldc);
}

#pragma SDS data access_pattern(A:SEQUENTIAL, B:SEQUENTIAL, C:SEQUENTIAL)
#pragma SDS data copy(A[0:144], B[0:169*144], C[0:169])
int cal_gemm(float *A, float *B, float *C){
    float A_l[144];
    float B_l[144][169];
    float output[169];
    #pragma HLS ARRAY_PARTITION variable=B_l cyclic factor=13 dim=2
    #pragma HLS ARRAY_PARTITION variable=output cyclic factor=13 dim=1
    int i, j, z;
// initialize
    for(i = 0; i < 169; i++){
#pragma HLS pipeline
        output[i] = 0;
    }
    for(i = 0; i < 144; i++){
#pragma HLS pipeline
        A_l[i] = A[i];
    }
    for(i = 0, z = 0; i < 144; i++){
        for(j = 0; j < 169; j++, z++){
#pragma HLS pipeline
//#pragma LOOP_FLATTEN OFF
            B_l[i][j] = B[z];
        }
    }

// calculation
    for(i = 0; i < 144; i++){
        for(j = 0; j < 169; j+=13){
#pragma HLS pipeline
            float *pB_l = B_l[i] + j;
            float *poutput = output + j;
            float A_ll = A_l[i];
            for(z = 0; z < 13; z++){
                poutput[z] += A_ll * pB_l[z];
//            output[j] += A_l[i] * B_l[i][j];
            }
        }
    }

    for(i = 0; i < 169; i++){
#pragma HLS pipeline
        C[i] = output[i];
    }
    return 0;
}

void gemm_nn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc){
    int i,j,k;
//    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[i*lda+k];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_nt(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
//    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
            }
            C[i*ldc+j] += sum;
        }
    }
}

void gemm_tn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
//    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[k*lda+i];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_tt(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
//    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i+k*lda]*B[k+j*ldb];
            }
            C[i*ldc+j] += sum;
        }
    }
}


void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    //printf("cpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);
    int i, j;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            C[i*ldc + j] *= BETA;
        }
    }
    if(!TA && !TB)
        gemm_nn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(TA && !TB)
        gemm_tn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(!TA && TB)
        gemm_nt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else
        gemm_tt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
}

void gemm_fpga(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    int i,j,k,z;
    float *A_buf;
    float *B_buf;
    float *C_buf;
    printf("fpga gemm try...\n");
    A_buf = (float *)sds_alloc(sizeof(float)*144);
    B_buf = (float *)sds_alloc(sizeof(float)*144*169);
    C_buf = (float *)sds_alloc(sizeof(float)*169);
    for(k = 0; k < K; k+= 144){
        for(z = 0; z < N; z += 169){
            memcpy(B_buf, B + k*N+ z*144, sizeof(float) * 144*169);
            for(i = 0; i < M; ++i){
                memcpy(A_buf, A + i*K + k, sizeof(float) * 144);
                // if(N == 169){
                //     memcpy(B_buf, B + k*169, sizeof(float) * 169*144);
                // }else{
                //     for(j = 0; j < 144; j++){
                //         memcpy(B_buf + j*169, B + j*N + z + k*N, sizeof(float) * 169);
                //     }                        
                // }
                cal_gemm(A_buf, B_buf, C_buf);
                for(j = 0; j < 169; j++){
                    C[i*N + z + j] += C_buf[j];
                }
            }
        }
    }
    sds_free(A_buf);
    sds_free(B_buf);
    sds_free(C_buf);
}

#if defined(HLS)
typedef hls::Window<3, 3, float>        window_t;
typedef hls::Window<3, 3, float>        kernel_t;
typedef hls::LineBuffer<3, 416, float>  lineReg_Lt;
typedef hls::LineBuffer<3, 52, float>   lineReg_Mt;
typedef hls::LineBuffer<3, 13, float>   lineReg_St;
#endif

#pragma SDS data access_pattern(conv_in:SEQUENTIAL, we:SEQUENTIAL, conv_out:SEQUENTIAL)
#pragma SDS data copy(conv_in[0:w_im*h_im*c_im], conv_out[0:w_im*h_im], we[0:ksize*ksize*c_im])
void cal_conv_large(
    float* conv_in,
    float* we,
    float* conv_out,
    int w_im,
    int h_im,
    int c_im,
    int ksize
    ){

    int w, h, c, kx, ky, k;

    // Read op.
    // accept upto 32 ch & width 416 im
    #if defined(HLS)
    lineReg_Lt      stream_Reg[32];
    window_t        window[32];
    kernel_t        kernel[32];
    #else
    float stream_Reg[32][3*416];
    float window[32][3*3];
    float kernel[32][3*3];
    #endif

    float* p_we = we;
    for(c = 0; c < c_im; c++){
        for(ky = 0; ky < 3; ky++){
            for(kx = 0; kx < 3; kx++){
                #if defined(HLS)
                kernel[c].insert(*p_we++, ky, kx);
                #else
                kernel[c][ky*ksize + kx] = *p_we++;
                #endif
            }
        }
    }

    // calculation
    for(c = 0; c < c_im; c++){
        for(h = 0; h < h_im + 1; h++){
            for(w = 0; w < w_im + 1; w++){

                float temp;
                float tempx;
                float res = 0;

            #if defined(HLS)
                if(w < w_im){
                    stream_Reg[c].shift_down(w);
                    temp = stream_Reg[c].getval(0,w);
                }

                if(w < w_im && h < h_im){
                    tempx = conv_in[c*w_im*h_im + h*w_im + w];
                    stream_Reg[c].insert_bottom(tempx, w);
                }

                window[c].shift_right();

                if(w < w_im){
                    window[c].insert(stream_Reg[c].getval(2,w),2,0);
                    window[c].insert(temp,1,0);
                    window[c].insert(tempx,0,0);
                }

                // conv
                // check...
                if(w > 1 && h > 1 && w <= (w_im - 1) && h <= (h_im - 1)){
                    for(ky = 0; ky < ksize; ky++){
                        for(kx = 0; kx < ksize; kx++){
                            res += window[c].getval(ky, kx) * kernel[c].getval(ky, kx);
                        }
                    }
                }

            #else
                if(w < w_im){
                    for(k = 3-1; k > 0; k--) {
                        stream_Reg[c][k*416 + w] = stream_Reg[c][(k-1)*416 + w];
                    }
                    temp = stream_Reg[c][0*416 + w];
                }

                if(w < w_im && h < h_im){
                    tempx = conv_in[c*w_im*h_im + h*w_im + w];
                    stream_Reg[c][0*416 + w] = tempx;
                }

                for(ky = 0; ky < 3; ky++){
                    for(kx = 2; kx > 0; kx--){
                        window[c][ky*3 + kx] = window[c][ky*3 + (kx-1)];
                    }
                }

                if(w < w_im){
                    window[c][0*3 + 2] = stream_Reg[c][2*416 + w];
                    window[c][1*3 + 2] = temp;
                    window[c][2*3 + 2] = tempx;
                }

                // conv
                if(w > 1 && h > 1 && w <= (w_im - 1) && h <= (h_im - 1)){
                    for(ky = 0; ky < ksize; ky++){
                        for(kx = 0; kx < ksize; kx++){
                            res += window[c][ky*ksize + kx] * kernel[c][ky*ksize + kx];
                        }
                    }
                }
            #endif

                if(w > 0 && h > 0)
                    conv_out[(h-1)*w_im + (w-1)] += res;
            }
       }
   }
}

#pragma SDS data access_pattern(conv_in:SEQUENTIAL, we:SEQUENTIAL, conv_out:SEQUENTIAL)
#pragma SDS data copy(conv_in[0:w_im*h_im*c_im], conv_out[0:w_im*h_im], we[0:ksize*ksize*c_im])
void cal_conv_medium(
    float* conv_in,
    float* we,
    float* conv_out,
    int w_im,
    int h_im,
    int c_im,
    int ksize
    ){

    int w, h, c, kx, ky, k;

    // Read op.
    // accept upto 512 ch & width 52 im
    
    #if defined(HLS)
    line_Reg_Mt     stream_Reg[512];
    window_t        window[512];
    kernel_t        kernel[512];
    #else
    float stream_Reg[512][3*52];
    float window[512][3*3];
    float kernel[512][3*3];
    #endif

    float* p_we = we;
    for(c = 0; c < c_im; c++){
        for(ky = 0; ky < 3; ky++){
            for(kx = 0; kx < 3; kx++){
                #if defined(HLS)
                kernel[c].insert(*p_we++, ky, kx);
                #else
                kernel[c][ky*ksize + kx] = *p_we++;
                #endif
            }
        }
    }

    // calculation
    for(c = 0; c < c_im; c++){
        for(h = 0; h < h_im + 1; h++){
            for(w = 0; w < w_im + 1; w++){

                float temp;
                float tempx;
                float res = 0;

            #if defined(HLS)
                if(w < w_im){
                    stream_Reg[c].shift_down(w);
                    temp = stream_Reg[c].getval(0,w);
                }

                if(w < w_im && h < h_im){
                    tempx = conv_in[c*w_im*h_im + h*w_im + w];
                    stream_Reg[c].insert_bottom(tempx, w);
                }

                window[c].shift_right();

                if(w < w_im){
                    window[c].insert(stream_Reg[c].getval(2,w),2,0);
                    window[c].insert(temp,1,0);
                    window[c].insert(tempx,0,0);
                }

                // conv
                if(w > 1 && h > 1 && w <= (w_im - 1) && h <= (h_im - 1)){
                    for(ky = 0; ky < ksize; ky++){
                        for(kx = 0; kx < ksize; kx++){
                            res += window[c].getval(ky, kx) * kernel[c].getval(ky, kx);
                        }
                    }
                }

            #else
                if(w < w_im){
                    for(k = 3-1; k > 0; k--) {
                        stream_Reg[c][k*52 + w] = stream_Reg[c][(k-1)*52 + w];
                    }
                    temp = stream_Reg[c][0*52 + w];
                }

                if(w < w_im && h < h_im){
                    tempx = conv_in[c*w_im*h_im + h*w_im + w];
                    stream_Reg[c][0*52 + w] = tempx;
                }

                for(ky = 0; ky < 3; ky++){
                    for(kx = 2; kx > 0; kx--){
                        window[c][ky*3 + kx] = window[c][ky*3 + (kx-1)];
                    }
                }

                if(w < w_im){
                    window[c][0*3 + 2] = stream_Reg[c][2*52 + w];
                    window[c][1*3 + 2] = temp;
                    window[c][2*3 + 2] = tempx;
                }

                // conv
                if(w > 1 && h > 1 && w <= (w_im - 1) && h <= (h_im - 1)){
                    for(ky = 0; ky < ksize; ky++){
                        for(kx = 0; kx < ksize; kx++){
                            res += window[c][ky*ksize + kx] * kernel[c][ky*ksize + kx];
                        }
                    }
                }
            #endif

                if(w > 0 && h > 0)
                    conv_out[(h-1)*w_im + (w-1)] += res;
            }
       }
   }
}

#pragma SDS data access_pattern(conv_in:SEQUENTIAL, we:SEQUENTIAL, conv_out:SEQUENTIAL)
#pragma SDS data copy(conv_in[0:w_im*h_im*c_im], conv_out[0:w_im*h_im], we[0:ksize*ksize*c_im])
void cal_conv_small(
    float* conv_in,
    float* we,
    float* conv_out,
    int w_im,
    int h_im,
    int c_im,
    int ksize
    ){

    int w, h, c, kx, ky, k;

    // Read op.
    // accept upto 1024 ch & width 13 im
    #if defined(HLS)
    line_Reg_St     stream_Reg[1024];
    window_t        window[1024];
    kernel_t        kernel[1024];
    #else
    float stream_Reg[1024][3*13];
    float window[1024][3*3];
    float kernel[1024][3*3];
    #endif

    float* p_we = we;
    for(c = 0; c < c_im; c++){
        for(ky = 0; ky < 3; ky++){
            for(kx = 0; kx < 3; kx++){
                #if defined(HLS)
                kernel[c].insert(*p_we++, ky, kx);
                #else
                kernel[c][ky*ksize + kx] = *p_we++;
                #endif
            }
        }
    }

    // calculation
    for(c = 0; c < c_im; c++){
        for(h = 0; h < h_im + 1; h++){
            for(w = 0; w < w_im + 1; w++){

                float temp;
                float tempx;
                float res = 0;
            #if defined(HLS)
                if(w < w_im){
                    stream_Reg[c].shift_down(w);
                    temp = stream_Reg[c].getval(0,w);
                }

                if(w < w_im && h < h_im){
                    tempx = conv_in[c*w_im*h_im + h*w_im + w];
                    stream_Reg[c].insert_bottom(tempx, w);
                }

                window[c].shift_right();

                if(w < w_im){
                    window[c].insert(stream_Reg[c].getval(2,w),2,0);
                    window[c].insert(temp,1,0);
                    window[c].insert(tempx,0,0);
                }

                // conv
                if(w > 1 && h > 1 && w <= (w_im - 1) && h <= (h_im - 1)){
                    for(ky = 0; ky < ksize; ky++){
                        for(kx = 0; kx < ksize; kx++){
                            res += window[c].getval(ky, kx) * kernel[c].getval(ky, kx);
                        }
                    }
                }
            #else
                if(w < w_im){
                    for(k = 3-1; k > 0; k--) {
                        stream_Reg[c][k*13 + w] = stream_Reg[c][(k-1)*13 + w];
                    }
                    temp = stream_Reg[c][0*13 + w];
                }

                if(w < w_im && h < h_im){
                    tempx = conv_in[c*w_im*h_im + h*w_im + w];
                    stream_Reg[c][0*13 + w] = tempx;
                }

                for(ky = 0; ky < 3; ky++){
                    for(kx = 2; kx > 0; kx--){
                        window[c][ky*3 + kx] = window[c][ky*3 + (kx-1)];
                    }
                }

                if(w < w_im){
                    window[c][0*3 + 2] = stream_Reg[c][2*13 + w];
                    window[c][1*3 + 2] = temp;
                    window[c][2*3 + 2] = tempx;
                }

                // conv
                if(w > 1 && h > 1 && w <= (w_im - 1) && h <= (h_im - 1)){
                    for(ky = 0; ky < ksize; ky++){
                        for(kx = 0; kx < ksize; kx++){
                            res += window[c][ky*ksize + kx] * kernel[c][ky*ksize + kx];
                        }
                    }
                }
            #endif

                if(w > 0 && h > 0)
                    conv_out[(h-1)*w_im + (w-1)] += res;
            }
       }
   }
}

void conv_fpga(
    float* imIn,
    float* imOut,
    float* we,
    int w_im,
    int h_im,
    int c_imIn,
    int c_imOut,
    int ksize){

    int co;

    //memset(imOut, 0, sizeof(float)*w_im*h_im*c_imOut);
    for(co = 0; co < c_imOut; co++){
        if(c_imIn < 64){
            cal_conv_large(imIn, we + co*ksize*ksize*c_imIn, imOut + co*w_im*h_im,
                w_im, h_im, c_imIn, ksize);
        }
        else if(c_imIn < 1024){
            cal_conv_medium(imIn, we + co*ksize*ksize*c_imIn, imOut + co*w_im*h_im,
                w_im, h_im, c_imIn, ksize);
        }
        else{
            cal_conv_small(imIn, we + co*ksize*ksize*c_imIn, imOut + co*w_im*h_im,
                w_im, h_im, c_imIn, ksize);
        }
    }
}

#ifdef GPU

#include <math.h>

void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A_gpu, int lda, 
        float *B_gpu, int ldb,
        float BETA,
        float *C_gpu, int ldc)
{
    cublasHandle_t handle = blas_handle();
    cudaError_t status = cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N), 
            (TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &ALPHA, B_gpu, ldb, A_gpu, lda, &BETA, C_gpu, ldc);
    check_error(status);
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void time_gpu_random_matrix(int TA, int TB, int m, int k, int n)
{
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    int i;
    clock_t start = clock(), end;
    for(i = 0; i<32; ++i){
        gemm_gpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    }
    end = clock();
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf s\n",m,k,k,n, TA, TB, (float)(end-start)/CLOCKS_PER_SEC);
    free(a);
    free(b);
    free(c);
}

void time_gpu(int TA, int TB, int m, int k, int n)
{
    int iter = 10;
    float *a = random_matrix(m,k);
    float *b = random_matrix(k,n);

    int lda = (!TA)?k:m;
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);

    float *a_cl = cuda_make_array(a, m*k);
    float *b_cl = cuda_make_array(b, k*n);
    float *c_cl = cuda_make_array(c, m*n);

    int i;
    clock_t start = clock(), end;
    for(i = 0; i<iter; ++i){
        gemm_gpu(TA,TB,m,n,k,1,a_cl,lda,b_cl,ldb,1,c_cl,n);
        cudaThreadSynchronize();
    }
    double flop = ((double)m)*n*(2.*k + 2.)*iter;
    double gflop = flop/pow(10., 9);
    end = clock();
    double seconds = sec(end-start);
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf s, %lf GFLOPS\n",m,k,k,n, TA, TB, seconds, gflop/seconds);
    cuda_free(a_cl);
    cuda_free(b_cl);
    cuda_free(c_cl);
    free(a);
    free(b);
    free(c);
}


void test_gpu_accuracy(int TA, int TB, int m, int k, int n)
{
    srand(0);
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    float *c_gpu = random_matrix(m,n);
    memset(c, 0, m*n*sizeof(float));
    memset(c_gpu, 0, m*n*sizeof(float));
    int i;
    //pm(m,k,b);
    gemm_gpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c_gpu,n);
    //printf("GPU\n");
    //pm(m, n, c_gpu);

    gemm_cpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    //printf("\n\nCPU\n");
    //pm(m, n, c);
    double sse = 0;
    for(i = 0; i < m*n; ++i) {
        //printf("%f %f\n", c[i], c_gpu[i]);
        sse += pow(c[i]-c_gpu[i], 2);
    }
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %g SSE\n",m,k,k,n, TA, TB, sse/(m*n));
    free(a);
    free(b);
    free(c);
    free(c_gpu);
}

int test_gpu_blas()
{
    /*
       test_gpu_accuracy(0,0,10,576,75); 

       test_gpu_accuracy(0,0,17,10,10); 
       test_gpu_accuracy(1,0,17,10,10); 
       test_gpu_accuracy(0,1,17,10,10); 
       test_gpu_accuracy(1,1,17,10,10); 

       test_gpu_accuracy(0,0,1000,10,100); 
       test_gpu_accuracy(1,0,1000,10,100); 
       test_gpu_accuracy(0,1,1000,10,100); 
       test_gpu_accuracy(1,1,1000,10,100); 

       test_gpu_accuracy(0,0,10,10,10); 

       time_gpu(0,0,64,2916,363); 
       time_gpu(0,0,64,2916,363); 
       time_gpu(0,0,64,2916,363); 
       time_gpu(0,0,192,729,1600); 
       time_gpu(0,0,384,196,1728); 
       time_gpu(0,0,256,196,3456); 
       time_gpu(0,0,256,196,2304); 
       time_gpu(0,0,128,4096,12544); 
       time_gpu(0,0,128,4096,4096); 
     */
    time_gpu(0,0,64,75,12544); 
    time_gpu(0,0,64,75,12544); 
    time_gpu(0,0,64,75,12544); 
    time_gpu(0,0,64,576,12544); 
    time_gpu(0,0,256,2304,784); 
    time_gpu(1,1,2304,256,784); 
    time_gpu(0,0,512,4608,196); 
    time_gpu(1,1,4608,512,196); 

    return 0;
}
#endif

