//
// Created by 14110 on 2022/11/9.
//

#ifndef MMO_UTILS_H
#define MMO_UTILS_H

#include <random>
#include <algorithm>
#include <iostream>
#include <windows.h>
#include <fstream>

using namespace std;

const int MIN_DIM = 100;
const int MAX_DIM = 2000;
const int STEP = 100;

typedef void (*mul_fun)(const float *, const float *, float *, int);

float create_random_number() {
    uniform_real_distribution<float> u(-100, 100);
    default_random_engine e{random_device{}()};
    return u(e);
}

float *create_random_matrix(int dim) {
    auto *mat = new float[dim * dim];
    for (int i = 0; i < dim * dim; ++i) mat[i] = create_random_number();
    return mat;
}

float *create_zero_matrix(int dim) {
    auto *mat = new float[dim * dim];
    for (int i = 0; i < dim * dim; ++i) mat[i] = 0;
    return mat;
}

void check(mul_fun fun1, mul_fun fun2, int dim) {
    const float EPS = 0.1;

    float *matA = create_random_matrix(dim);
    float *matB = create_random_matrix(dim);
    float *matC1 = create_zero_matrix(dim);
    float *matC2 = create_zero_matrix(dim);

    fun1(matA, matB, matC1, dim);
    fun2(matA, matB, matC2, dim);

    bool ok = true;
    for (int i = 0; i < dim * dim; ++i) {
        if (abs(matC1[i] - matC2[i]) > EPS) {
            cerr << matC1[i] << " " << matC2[i] << " is not equal! Function is error!" << endl;
            ok = false;
            break;
        }
    }
    if (ok) cout << "AC" << endl;

    delete[] matA, matB, matC1, matC2;
}

void test(mul_fun fun, int test_times, const string &log_name) {
    LARGE_INTEGER st, et, freq;
    ofstream logger("../logs/" + log_name + ".log");
    if (!logger) {
        cerr << "Cant create " + log_name + ".log" << endl;
        exit(-1);
    } else {
        cout << "Start testing. Save in : " << log_name + ".log" << endl;
    }
    for (int dim = MIN_DIM; dim <= MAX_DIM; dim += STEP) {
        double total_time = 0;
        cout << "Testing dim :" << dim << endl;
        QueryPerformanceFrequency(&freq);
        for (int i = 0; i < test_times; ++i) {
            float *matA = create_random_matrix(dim);
            float *matB = create_random_matrix(dim);
            float *matC = create_zero_matrix(dim);

            QueryPerformanceCounter(&st);
            fun(matA, matB, matC, dim);
            QueryPerformanceCounter(&et);

            total_time += (double) (et.QuadPart - st.QuadPart) / (double) freq.QuadPart;
            delete[] matA, matB, matC;
        }
        logger << dim << ' ' << test_times << ' ' << total_time << '\n';
    }
    logger.close();
}


#endif //MMO_UTILS_H
