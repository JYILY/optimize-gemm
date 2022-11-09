//
// Created by 14110 on 2022/11/9.
//

#ifndef MMO_MULTIPLICATION_H
#define MMO_MULTIPLICATION_H

#include <thread>

#define index(i, j) ((j)*dim+(i))

void row_loop(const float *A, const float *B, float *C, int dim) {
    int i, j, p;
    for (j = 0; j < dim; ++j) {
        for (i = 0; i < dim; ++i) {
            for (p = 0; p < dim; ++p) {
                C[index(i, j)] += A[index(i, p)] * B[index(p, j)];
            }
        }
    }
}

void col_loop(const float *A, const float *B, float *C, int dim) {
    int i, j, p;
    for (i = 0; i < dim; ++i) {
        for (j = 0; j < dim; ++j) {
            for (p = 0; p < dim; ++p) {
                C[index(i, j)] += A[index(i, p)] * B[index(p, j)];
            }
        }
    }
}

void cache_11(const float *A, const float *B, float *C, int dim) {
    int i, j, p, u;
    float reg;
    for (j = 0; j < dim; ++j) {
        for (i = 0; i < dim; ++i) {
            reg = 0;
            for (p = 0; p < dim; ++p) {
                reg += A[index(i, p)] * B[index(p, j)];
            }
            C[index(i, j)] = reg;
        }
    }
}

void cache_12(const float *A, const float *B, float *C, int dim) {
    int i, j, p, u;
    register float reg;
    for (j = 0; j < dim; ++j) {
        for (i = 0; i < dim; ++i) {
            reg = 0;
            for (p = 0; p < dim; ++p) {
                reg += A[index(i, p)] * B[index(p, j)];
            }
            C[index(i, j)] = reg;
        }
    }
}

void cache_21(const float *A, const float *B, float *C, int dim) {
    int i, j, p, u;
    float reg[4];
    for (j = 0; j < dim; j += 4) {
        for (i = 0; i < dim; ++i) {
            for (u = 0; u < 4; ++u) reg[u] = 0;
            for (p = 0; p < dim; ++p) {
                for (u = 0; u < 4; ++u)
                    reg[u] += A[index(i, p)] * B[index(p, j + u)];
            }
            for (u = 0; u < 4; ++u) C[index(i, j + u)] = reg[u];
        }
    }
}

void cache_22(const float *A, const float *B, float *C, int dim) {
    int i, j, p, u;
    register float reg[4];
    for (j = 0; j < dim; j += 4) {
        for (i = 0; i < dim; ++i) {
            for (u = 0; u < 4; ++u) reg[u] = 0;
            for (p = 0; p < dim; ++p) {
                for (u = 0; u < 4; ++u)
                    reg[u] += A[index(i, p)] * B[index(p, j + u)];
            }
            for (u = 0; u < 4; ++u) C[index(i, j + u)] = reg[u];
        }
    }
}

void cache_31(const float *A, const float *B, float *C, int dim) {
    int i, j, p, u;
    float c_reg;
    float *b_ptr, *bb_ptr;
    b_ptr = const_cast<float *>(B);

    for (j = 0; j < dim; ++j, b_ptr += dim) {
        for (i = 0; i < dim; ++i) {
            bb_ptr = b_ptr;
            c_reg = 0;
            for (p = 0; p < dim; ++p) {
                c_reg += A[index(i, p)] * (*bb_ptr);
                ++bb_ptr;
            }
            C[index(i, j)] = c_reg;
        }
    }
}

void cache_32(const float *A, const float *B, float *C, int dim) {
    int i, j, p, u;
    float c_reg0, c_reg1, c_reg2, c_reg3, a_reg;
    float *b0_ptr, *b1_ptr, *b2_ptr, *b3_ptr, *bb0_ptr, *bb1_ptr, *bb2_ptr, *bb3_ptr;
    auto *b_ptr = const_cast<float *>(B);

    for (j = 0; j < dim; j += 4) {

        b0_ptr = b_ptr;
        b_ptr += dim;
        b1_ptr = b_ptr;
        b_ptr += dim;
        b2_ptr = b_ptr;
        b_ptr += dim;
        b3_ptr = b_ptr;
        b_ptr += dim;

        for (i = 0; i < dim; ++i) {

            bb0_ptr = b0_ptr;
            bb1_ptr = b1_ptr;
            bb2_ptr = b2_ptr;
            bb3_ptr = b3_ptr;

            c_reg0 = c_reg1 = c_reg2 = c_reg3 = 0;

            for (p = 0; p < dim; ++p) {
                a_reg = A[index(i, p)];

                c_reg0 += a_reg * *bb0_ptr++;
                c_reg1 += a_reg * *bb1_ptr++;
                c_reg2 += a_reg * *bb2_ptr++;
                c_reg3 += a_reg * *bb3_ptr++;
            }
            C[index(i, j)] = c_reg0;
            C[index(i, j + 1)] = c_reg1;
            C[index(i, j + 2)] = c_reg2;
            C[index(i, j + 3)] = c_reg3;
        }
    }
}

void cache_33(const float *A, const float *B, float *C, int dim) {
    int i, j, p, u;
    float c_regs[4];
    float a_reg;
    float *b_ptr_arr[4];
    float *bb_ptr_arr[4];
    auto *b_ptr = const_cast<float *>(B);

    for (j = 0; j < dim; j += 4) {

        for (u = 0; u < 4; ++u, b_ptr += dim) {
            b_ptr_arr[u] = b_ptr;
        }

        for (i = 0; i < dim; ++i) {
            for (u = 0; u < 4; ++u) {
                bb_ptr_arr[u] = b_ptr_arr[u];
                c_regs[u] = 0;
            }
            for (p = 0; p < dim; ++p) {
                a_reg = A[index(i, p)];
                for (u = 0; u < 4; ++u) {
                    c_regs[u] += a_reg * *bb_ptr_arr[u]++;
                }

            }
            for (u = 0; u < 4; ++u) {
                C[index(i, j + u)] += c_regs[u];
            }
        }
    }
}

void cache_41(const float *A, const float *B, float *C, int dim) {
    int i, j, p, u;
    float reg00, reg01, reg02, reg03;
    float reg10, reg11, reg12, reg13;
    float reg20, reg21, reg22, reg23;
    float reg30, reg31, reg32, reg33;

    float a0_reg, a1_reg, a2_reg, a3_reg;

    float *b0_ptr, *b1_ptr, *b2_ptr, *b3_ptr, *bb0_ptr, *bb1_ptr, *bb2_ptr, *bb3_ptr;
    auto *b_ptr = const_cast<float *>(B);
    auto *a_ptr = const_cast<float *>(A);
    float *aa_ptr, *c_ptr, *cc_ptr;
    for (j = 0; j < dim; j += 4) {

        b0_ptr = b_ptr;
        b_ptr += dim;
        b1_ptr = b_ptr;
        b_ptr += dim;
        b2_ptr = b_ptr;
        b_ptr += dim;
        b3_ptr = b_ptr;
        b_ptr += dim;

        for (i = 0; i < dim; i += 4) {

            bb0_ptr = b0_ptr;
            bb1_ptr = b1_ptr;
            bb2_ptr = b2_ptr;
            bb3_ptr = b3_ptr;

            reg00 = reg01 = reg02 = reg03 = 0;
            reg10 = reg11 = reg12 = reg13 = 0;
            reg20 = reg21 = reg22 = reg23 = 0;
            reg30 = reg31 = reg32 = reg33 = 0;

            for (p = 0; p < dim; ++p, aa_ptr += dim) {
                aa_ptr = a_ptr + p * dim + i;
                a0_reg = *aa_ptr++;
                a1_reg = *aa_ptr++;
                a2_reg = *aa_ptr++;
                a3_reg = *aa_ptr++;

                reg00 += a0_reg * *bb0_ptr;
                reg10 += a1_reg * *bb0_ptr;
                reg20 += a2_reg * *bb0_ptr;
                reg30 += a3_reg * *bb0_ptr;
                ++bb0_ptr;

                reg01 += a0_reg * *bb1_ptr;
                reg11 += a1_reg * *bb1_ptr;
                reg21 += a2_reg * *bb1_ptr;
                reg31 += a3_reg * *bb1_ptr;
                ++bb1_ptr;

                reg02 += a0_reg * *bb2_ptr;
                reg12 += a1_reg * *bb2_ptr;
                reg22 += a2_reg * *bb2_ptr;
                reg32 += a3_reg * *bb2_ptr;
                ++bb2_ptr;

                reg03 += a0_reg * *bb3_ptr;
                reg13 += a1_reg * *bb3_ptr;
                reg23 += a2_reg * *bb3_ptr;
                reg33 += a3_reg * *bb3_ptr;
                ++bb3_ptr;
            }
            c_ptr = &C[index(i, j)];
            cc_ptr = c_ptr;
            *cc_ptr++ = reg00;
            *cc_ptr++ = reg10;
            *cc_ptr++ = reg20;
            *cc_ptr = reg30;

            c_ptr += dim;
            cc_ptr = c_ptr;
            *cc_ptr++ = reg01;
            *cc_ptr++ = reg11;
            *cc_ptr++ = reg21;
            *cc_ptr = reg31;

            c_ptr += dim;
            cc_ptr = c_ptr;
            *cc_ptr++ = reg02;
            *cc_ptr++ = reg12;
            *cc_ptr++ = reg22;
            *cc_ptr = reg32;

            c_ptr += dim;
            cc_ptr = c_ptr;
            *cc_ptr++ = reg03;
            *cc_ptr++ = reg13;
            *cc_ptr++ = reg23;
            *cc_ptr = reg33;

        }
    }
}

void simd(const float *A, const float *B, float *C, int dim) {
    int i, j, p, u;

    float *c0_ptr, *c1_ptr, *c2_ptr, *c3_ptr;
    float *b0_ptr, *b1_ptr, *b2_ptr, *b3_ptr;
    auto *packedA = new float[4 * dim];
    float *p_ptr, *a_ptr, *b_ptr;
    float *aa_ptr, *bb_ptr, *cc_ptr;
    float *t_ptr;

    __m128 cv0, cv1, cv2, cv3;
    __m128 va0, va1, va2, va3;
    __m128 vb0, vb1, vb2, vb3;
    __m128 vc0, vc1, vc2, vc3;
    // data : A[mxk]  B[kxn]
    aa_ptr = const_cast<float *>(A);
    bb_ptr = const_cast<float *>(B);
    cc_ptr = C;

    for (i = 0; i < dim; i += 4) {
        p_ptr = packedA;
        for (p = 0; p < dim; ++p) {
            a_ptr = aa_ptr + p * dim + i;
            for (u = 0; u < 4; ++u) {
                *p_ptr++ = *a_ptr++;
            }
        }

        b_ptr = bb_ptr;
        for (j = 0; j < dim; j += 4) {
            p_ptr = packedA;
            t_ptr = bb_ptr + j * dim;
            b0_ptr = t_ptr;
            t_ptr += dim;
            b1_ptr = t_ptr;
            t_ptr += dim;
            b2_ptr = t_ptr;
            t_ptr += dim;
            b3_ptr = t_ptr;

            cv0 = _mm_setzero_ps();
            cv1 = _mm_setzero_ps();
            cv2 = _mm_setzero_ps();
            cv3 = _mm_setzero_ps();

            for (p = 0; p < dim; p += 4) {
                // mul : A[i:i+4,p:p+4] B[p:p+4,j:j+4]
                va0 = _mm_load_ps(p_ptr);
                p_ptr += 4;
                va1 = _mm_load_ps(p_ptr);
                p_ptr += 4;
                va2 = _mm_load_ps(p_ptr);
                p_ptr += 4;
                va3 = _mm_load_ps(p_ptr);
                p_ptr += 4;

                cv0 += va0 * *b0_ptr++;
                cv0 += va1 * *b0_ptr++;
                cv0 += va2 * *b0_ptr++;
                cv0 += va3 * *b0_ptr++;

                cv1 += va0 * *b1_ptr++;
                cv1 += va1 * *b1_ptr++;
                cv1 += va2 * *b1_ptr++;
                cv1 += va3 * *b1_ptr++;

                cv2 += va0 * *b2_ptr++;
                cv2 += va1 * *b2_ptr++;
                cv2 += va2 * *b2_ptr++;
                cv2 += va3 * *b2_ptr++;

                cv3 += va0 * *b3_ptr++;
                cv3 += va1 * *b3_ptr++;
                cv3 += va2 * *b3_ptr++;
                cv3 += va3 * *b3_ptr++;
            }

            cc_ptr = &C[index(i, j)];
            _mm_store_ps(cc_ptr, cv0);
            cc_ptr += dim;
            _mm_store_ps(cc_ptr, cv1);
            cc_ptr += dim;
            _mm_store_ps(cc_ptr, cv2);
            cc_ptr += dim;
            _mm_store_ps(cc_ptr, cv3);
            cc_ptr += dim;
        }
    }
    delete[] packedA;
}

void mul_thread(float *a_ptr, float *b_ptr, float *c_ptr, int dim) {
    int i, j, p, u;
    float *aa_ptr, *bb_ptr, *cc_ptr, *p_ptr;
    __m128 va0, va1, va2, va3;
    __m128 vb, vc;
    auto *packedA = new float[4 * dim];
    int num_col = dim / 4;

    for (i = 0; i < dim; i += 4) {
        p_ptr = packedA;
        for (p = 0; p < dim; ++p) {
            aa_ptr = a_ptr + p * dim + i;
            for (u = 0; u < 4; ++u) {
                *p_ptr++ = *aa_ptr++;
            }
        }
        bb_ptr = b_ptr;
        cc_ptr = c_ptr;
        for (j = 0; j < num_col; ++j) {
            vc = _mm_setzero_ps();
            aa_ptr = packedA;
            for (p = 0; p < dim; p += 4) {
                va0 = _mm_load_ps(aa_ptr);
                aa_ptr += 4;
                va1 = _mm_load_ps(aa_ptr);
                aa_ptr += 4;
                va2 = _mm_load_ps(aa_ptr);
                aa_ptr += 4;
                va3 = _mm_load_ps(aa_ptr);
                aa_ptr += 4;

                vc += va0 * *bb_ptr++;
                vc += va1 * *bb_ptr++;
                vc += va2 * *bb_ptr++;
                vc += va3 * *bb_ptr++;
            }
            _mm_store_ps(cc_ptr, vc);
            cc_ptr += dim;
        }
        c_ptr += 4;
    }
    delete[] packedA;
}

void thread1(const float *A, const float *B, float *C, int dim) {
    int i, j, p, u, num_col;
    pthread_t tid_list[4];
    auto *a_ptr = const_cast<float *>(A);
    auto *b_ptr = const_cast<float *>(B);
    auto *c_ptr = C;
    thread th1(mul_thread, a_ptr, b_ptr, c_ptr, dim);
    num_col = dim / 4;
    c_ptr += num_col * dim;
    b_ptr += num_col * dim;
    thread th2(mul_thread, a_ptr, b_ptr, c_ptr, dim);
    c_ptr += num_col * dim;
    b_ptr += num_col * dim;
    thread th3(mul_thread, a_ptr, b_ptr, c_ptr, dim);
    c_ptr += num_col * dim;
    b_ptr += num_col * dim;
    thread th4(mul_thread, a_ptr, b_ptr, c_ptr, dim);

    th1.join();
    th2.join();
    th3.join();
    th4.join();
}

#endif //MMO_MULTIPLICATION_H
