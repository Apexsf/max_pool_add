#ifndef MAX_POOL_ADD_H
#define MAX_POOL_ADD_H

#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <immintrin.h>
#include <memory>
#include <iostream>


#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include "utils.hpp"


struct Stride {
    size_t stride_B;
    size_t stride_C;
    size_t stride_H;
    size_t stride_W;

};

template<typename T>
T* read_data_from_file(const char * path, T* shape);

template<typename T>
struct Tensor {
    static_assert(std::is_same<T, float>::value 
    || std::is_same<T, double>::value 
    || std::is_same<T, int>::value 
    , "Tensor value types are restricted to double, float or int!");

    /**
     * @brief Default constructor, create a tensor that is not valid / initialized
     * 
     */
    Tensor() { }

    Tensor(size_t B, size_t C, size_t H, size_t W) : 
    B(B), C(C), H(H), W(W) {
         p = (T*)malloc(sizeof(T) * B * C * H * W);
    }

    /**
     * @brief Construct a new Tensor object by deserializing data from file
     * 
     * @param path: path of file from which to deserialize
     */
    Tensor(const char* path) { 
        T shape[4];
        p = read_data_from_file(path, shape);
        B = shape[0];
        C = shape[1];
        H = shape[2];
        W = shape[3];
    }


    Tensor(const Tensor<T>& t) { // copy constructor
        B = t.B;
        C = t.C;
        H = t.H;
        W = t.W;
        size_t size = t.size();
        T* p = (T*) malloc(sizeof(T) * size);
        memcpy(p, t.p, sizeof(T) * size);
    }

    Tensor(Tensor<T> && t) { // move constructor
        B = t.B;
        C = t.C;
        H = t.H;
        W = t.W;
        if (p) delete[] p;
        p = t.p;
        t.p = nullptr;
    }

    ~Tensor() {
        if (p) {
            free(p);
        }
    }

    bool operator == (const Tensor<T>& t) const {
        if (B != t.B) return false;
        if (C != t.C) return false;
        if (H != t.H) return false;
        if (W != t.W) return false;
        size_t size = this->size();
        for (size_t i = 0; i < size; ++i) {
            if (p[i] != t.p[i]) 
                return false;
        }  
        return true;

    }

    size_t size() const{
        return B * C * H * W;
    }

    Stride stride() const {
        return Stride{C*H*W, H*W, W, 1};
    }

    void print_elems() const {
        size_t size = this->size();
        for( size_t i = 0; i < size; ++i) {
            std::cout << p[i] << std::endl;
        }
    }

    bool is_valid() const {
        return p != nullptr;
    }

    size_t B;
    size_t C;
    size_t H;
    size_t W;
    T* p = nullptr;
};


template<typename T> 
void add_array(T* p_a, T*p_b, T* res, size_t size) {
    size_t i;

    #ifdef USE_OMP
    #pragma omp parallel shared(p_a, p_b, res) private(i) num_threads(4)
    {
        #pragma omp for schedule(static)
    #endif
        for(i = 0; i < size; ++i) {
            res[i] = p_a[i] + p_b[i];
        }
    #ifdef USE_OMP
    }
    #endif

}

#ifdef USE_AVX
template <>
void add_array(int* p_a, int*p_b, int* res, size_t size) {
    size_t i ;
    // present no substantial speeding up with both openmp and SIMD compared to using only openmp
    // see possible reasons: https://stackoverflow.com/questions/42895066/no-speedup-using-openmp-simd
    #ifdef USE_OMP
    #pragma omp parallel shared(p_a, p_b, res) private(i) num_threads(4)
    {
        #pragma omp for schedule(static)
    #endif
        for( i = 0; i <= size -8; i += 8){
            __m256i A = _mm256_loadu_si256((const __m256i*) (p_a + i));
            __m256i B = _mm256_loadu_si256((const __m256i*) (p_b + i));
            __m256i C = _mm256_add_epi32(A, B);
            _mm256_storeu_si256((__m256i*)(res + i), C);
        }
    #ifdef USE_OMP   
    }
    #endif

    i = size - 7;
    for(; i < size; ++i) {
        res[i] = p_a[i] + p_b[i];
    }
}

template <>
void add_array(float* p_a, float* p_b, float* res, size_t size) {
    size_t i ;
    // present no substantial speeding up with both openmp and SIMD compared to using only openmp
    // see possible reasons: https://stackoverflow.com/questions/42895066/no-speedup-using-openmp-simd

    #ifdef USE_OMP
    #pragma omp parallel shared(p_a, p_b, res) private(i) num_threads(4)
    {
        #pragma omp for schedule(static)
    #endif
        for( i = 0; i <= size -8; i += 8){
            __m256 A = _mm256_loadu_ps((p_a + i));
            __m256 B = _mm256_loadu_ps((p_b + i));
            __m256 C = _mm256_add_ps(A, B);
            _mm256_storeu_ps((res + i), C);
        }   
    #ifdef USE_OMP
    }
    #endif
    i = size - 7;
    for(; i < size; ++i) {
        res[i] = p_a[i] + p_b[i];
    }
}

template <>
void add_array(double* p_a, double* p_b, double* res, size_t size) {
    size_t i ;
    // present no substantial speeding up with both openmp and SIMD compared to using only openmp
    // see possible reasons: https://stackoverflow.com/questions/42895066/no-speedup-using-openmp-simd

    #ifdef USE_OMP
    #pragma omp parallel shared(p_a, p_b, res) private(i) num_threads(4)
    {
        #pragma omp for schedule(static)
    #endif
        for( i = 0; i <= size -4; i += 4){
            __m256d A = _mm256_loadu_pd((p_a + i));
            __m256d B = _mm256_loadu_pd((p_b + i));
            __m256d C = _mm256_add_pd(A, B);
            _mm256_storeu_pd((res + i), C);
        }   
    #ifdef USE_OMP
    }
    #endif
    i = size - 3;
    for(; i < size; ++i) {
        res[i] = p_a[i] + p_b[i];
    }
}
#endif // USE_AVX


template<typename T>
void padding_2D(T* src, T* dst, size_t H, size_t W) {
    size_t i, j;
    #ifdef USE_OMP
    #pragma omp parallel private(i,j) num_threads(4)
    {
        #pragma omp for schedule(static)
    #endif
        for(i = 0 ; i < H; ++i) {
            for(j = 0; j < W; ++j) {
                dst[(i + 1) * (W + 2) + j + 1] = src[i * W + j];
            }
        }
    #ifdef USE_OMP
    }
    #endif
}

template<typename T>
void max_pool_2D(T* src, T* dst, size_t src_H, size_t src_W, size_t dst_H, size_t dst_W) {
    T max_elem;
    size_t i,j,k_i,k_j;
    #ifdef USE_OMP
    #pragma omp parallel private(i,j,k_i,k_j,max_elem) num_threads(4)
    {
        #pragma omp for schedule(static)
    #endif
        for(i = 0; i < src_H-2;  i += 2) {
            for (j = 0; j < src_W-2; j += 2) {
                max_elem = src[i * src_W + j];
                for(k_i  = 0; k_i < 3; ++k_i) {
                    for(k_j = 0; k_j < 3; ++ k_j) {
                        if (max_elem < src[(i + k_i) * src_W + j + k_j]) {
                            max_elem = src[(i + k_i) * src_W + j + k_j];
                        }
                    }
                }
                dst[(i/2) * dst_W + j/2] = max_elem;
            }
        } 
    #ifdef USE_OMP
    }   
    #endif

}

template<typename T>
Tensor<T> max_pool(Tensor<T>& src) {
    // pre-allocated memory for holding all the following padded 2-D matrix.
    // all the elements are initialized as 0 to save the “0 assignment" in default padding.
    T* padding = (T*) calloc((src.H + 2) * (src.W + 2), sizeof(T)); 
    if (padding == nullptr) {
        abort();
    }

    size_t dst_H =  (src.H + 1) / 2;
    size_t dst_W =  (src.W + 1) / 2 ;
    Tensor<T> dst(src.B, src.C, dst_H, dst_W);

    Stride src_str = src.stride();
    Stride dst_str=  dst.stride();

    size_t b_i, c_i;
    for (b_i = 0 ; b_i < dst.B; ++b_i) {
        for(c_i = 0; c_i < dst.C; ++c_i) {
            padding_2D(src.p + b_i * src_str.stride_B + c_i * src_str.stride_C, padding, src.H, src.W); 
            max_pool_2D(padding, dst.p + b_i * dst_str.stride_B + c_i * dst_str.stride_C, src.H + 2, src.W + 2, dst_H, dst_W);
        }
    }

    free(padding);

    return dst;
}



/**
 * @brief 
 * 
 * @return 
 * -1: not support elem-wise operation. 
 * 0: no broadcast requirement
 * 1: broadcast on the first tensor
 * 2: broadcast on the second tensor
 * 3: broadcast on the both tensors.
 *      
 */
template<typename T> 
int elem_wise_op_size_check(Tensor<T>& a, Tensor<T> & b) { 
    if (a.B != 1 && b.B != 1 && a.B != b.B) return -1;
    if (a.C != 1 && b.C != 1 && a.C != b.C) return -1;
    if (a.H != 1 && b.H != 1 && a.H != b.H) return -1;
    if (a.W != 1 && b.W != 1 && a.W != b.W) return -1;
    
    int broadcast_cnt = 0;
    if (a.B == 1 || a.C == 1 || a.H == 1 || a.W == 1) broadcast_cnt += 1;
    if (b.B == 1 || b.C == 1 || b.H == 1 || b.W == 1) broadcast_cnt += 2;
    return broadcast_cnt;
}


template<typename T>
void expand(Tensor<T>& src, Tensor<T>& dst) { // expand for broadcast,  e.g., 4*1*200*1 -> 4*3*200*400 
    Stride src_str = src.stride();
    Stride dst_str = dst.stride();
    size_t src_index;
    size_t b_i, c_i, h_i, w_i;
    
    // adjust strides of dimensions with 1 to 0 for broadcasting.
    src_str.stride_B *= (src.B == 1 ? 0 : 1);
    src_str.stride_C *= (src.C == 1 ? 0 : 1);
    src_str.stride_H *= (src.H == 1 ? 0 : 1);
    src_str.stride_W *= (src.W == 1 ? 0 : 1);

    #ifdef USE_OMP
    #pragma omp parallel private(b_i,c_i,h_i,w_i, src_index) num_threads(4)
    {
        #pragma omp for schedule(static)
    #endif
        for(b_i = 0; b_i < dst.B; ++b_i) {
            for(c_i = 0; c_i < dst.C; ++c_i) {
                for(h_i = 0; h_i < dst.H; ++h_i) {
                    for(w_i = 0; w_i < dst.W; ++ w_i) {
                        src_index = 0;
                        src_index += b_i * src_str.stride_B;
                        src_index += c_i * src_str.stride_C;
                        src_index += h_i * src_str.stride_H;
                        src_index += w_i * src_str.stride_W;
                        dst.p[b_i * dst_str.stride_B + c_i * dst_str.stride_C + h_i * 
                        dst_str.stride_H + w_i * dst_str.stride_W] = src.p[src_index];
                    }
                }
            }
        }
    #ifdef USE_OMP
    }
    #endif
}


template<typename T>
Tensor<T> add(Tensor<T>& a, Tensor<T>& b) {
    int states = elem_wise_op_size_check(a, b);
    if (states == -1) 
        return Tensor<T>(); // size mismatch, return an invalid tensor indicating elem-wise add cannot be applied to a and b.
    size_t res_B = std::max(a.B, b.B);
    size_t res_C = std::max(a.C, b.C);
    size_t res_H = std::max(a.H, b.H);
    size_t res_W = std::max(a.W, b.W);

    Tensor<T> res(res_B, res_C, res_H, res_W);

    if (states == 0) {  // no broadcast
        add_array(a.p, b.p, res.p, res.size());
    }else if (states == 1) { // broadcast a 
        Tensor<T> broadcast_a(res_B, res_C, res_H, res_W);
        expand(a, broadcast_a);
        add_array(broadcast_a.p, b.p, res.p, res.size());
    } else if (states == 2) { // broadcast b 
        Tensor<T> broadcast_b(res_B, res_C, res_H, res_W);
        expand(b, broadcast_b);
        add_array(a.p, broadcast_b.p, res.p, res.size());
    } else if (states == 3) { // broadcast a and b
        Tensor<T> broadcast_a(res_B, res_C, res_H, res_W);
        expand(a, broadcast_a);
        Tensor<T> broadcast_b(res_B, res_C, res_H, res_W);
        expand(b, broadcast_b);
        add_array(broadcast_a.p, broadcast_b.p, res.p, res.size());
    }
    return res;
}

template<typename T>
Tensor<T> max_pool_add(Tensor<T>& a, Tensor<T>& b) {
    Tensor<T> a_max_pool = max_pool(a);
    Tensor<T> res = add(a_max_pool, b);
    return res;
}

template <typename T>
void helper_fill_sequence(Tensor<T> & tensor) {
    size_t cnt = 0;
    size_t tensor_len = tensor.size();
    for(size_t i = 0; i < tensor_len; ++i) {
        tensor.p[i] = cnt;
        cnt += 1;
    }
}

template<typename T>
T* read_data_from_file(const char * path, T* shape) {
    std::ifstream input;
    input.open(path, std::ios::in | std::ios::binary);
    input.read((char*)shape, 4 * sizeof(T));
    size_t size =(size_t)shape[0] * (size_t)shape[1] * (size_t)shape[2] *(size_t) shape[3]; // cast to size_t to avoid overflow
    T * arr = (T *)malloc(size * sizeof(T));
    input.read((char*)arr, size * sizeof(T));
    input.close();
    return arr;
}


#endif // MAX_POOL_ADD_H