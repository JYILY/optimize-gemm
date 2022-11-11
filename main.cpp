//
// Created by 14110 on 2022/11/9.
//
#include<iostream>
#include "utils.h"
#include "Multiplication.h"

using namespace std;

const int TEST_TIMES = 10;
const int CHECK_DIM = 100;


void test_row_col() {
    check(row_loop, col_loop, CHECK_DIM);
    test(row_loop, TEST_TIMES, "row_loop");
    test(col_loop, TEST_TIMES, "col_loop");
}

void test_cache1() {
    check(row_loop, cache_11, CHECK_DIM);
    check(row_loop, cache_12, CHECK_DIM);
    test(row_loop, TEST_TIMES, "row_loop");
    test(cache_11, TEST_TIMES, "cache_11");
    test(cache_12, TEST_TIMES, "cache_12");
}

void test_cache2() {
    check(row_loop, cache_11, CHECK_DIM);
    check(row_loop, cache_12, CHECK_DIM);
    check(row_loop, cache_21, CHECK_DIM);
    check(row_loop, cache_22, CHECK_DIM);
    test(row_loop, TEST_TIMES, "row_loop");
    test(cache_11, TEST_TIMES, "cache_11");
    test(cache_12, TEST_TIMES, "cache_12");
    test(cache_21, TEST_TIMES, "cache_21");
    test(cache_22, TEST_TIMES, "cache_22");
}

void test_cache3() {
    check(row_loop, cache_11, CHECK_DIM);
    check(row_loop, cache_12, CHECK_DIM);
    check(row_loop, cache_21, CHECK_DIM);
    check(row_loop, cache_22, CHECK_DIM);
    check(row_loop, cache_31, CHECK_DIM);
    check(row_loop, cache_32, CHECK_DIM);
    test(row_loop, TEST_TIMES, "row_loop");
    test(cache_11, TEST_TIMES, "cache_11");
    test(cache_12, TEST_TIMES, "cache_12");
    test(cache_21, TEST_TIMES, "cache_21");
    test(cache_22, TEST_TIMES, "cache_22");
    test(cache_31, TEST_TIMES, "cache_31");
    test(cache_32, TEST_TIMES, "cache_32");
    test(cache_33, TEST_TIMES, "cache_33");
}

void test_cache4() {

    check(row_loop, cache_12, CHECK_DIM);
    check(row_loop, cache_22, CHECK_DIM);
    check(row_loop, cache_32, CHECK_DIM);
    check(row_loop, cache_41, CHECK_DIM);

    test(row_loop, TEST_TIMES, "row_loop");
    test(cache_12, TEST_TIMES, "cache_12");
    test(cache_22, TEST_TIMES, "cache_22");
    test(cache_32, TEST_TIMES, "cache_32");
    test(cache_41, TEST_TIMES, "cache_41");
}

void test_simd() {
    check(row_loop, cache_12, CHECK_DIM);
    check(row_loop, cache_22, CHECK_DIM);
    check(row_loop, cache_32, CHECK_DIM);
    check(row_loop, cache_41, CHECK_DIM);
    check(row_loop, simd, CHECK_DIM);

    test(row_loop, TEST_TIMES, "row_loop");
    test(cache_12, TEST_TIMES, "cache_12");
    test(cache_22, TEST_TIMES, "cache_22");
    test(cache_32, TEST_TIMES, "cache_32");
    test(cache_41, TEST_TIMES, "cache_41");
    test(simd, TEST_TIMES, "simd");
}

void test_thread() {
    check(row_loop, cache_12, CHECK_DIM);
    check(row_loop, cache_22, CHECK_DIM);
    check(row_loop, cache_32, CHECK_DIM);
    check(row_loop, cache_41, CHECK_DIM);
    check(row_loop, simd, CHECK_DIM);
    check(row_loop, thread1, CHECK_DIM);

    test(row_loop, TEST_TIMES, "row_loop");
    test(cache_12, TEST_TIMES, "cache_12");
    test(cache_22, TEST_TIMES, "cache_22");
    test(cache_32, TEST_TIMES, "cache_32");
    test(cache_41, TEST_TIMES, "cache_41");
    test(simd, TEST_TIMES, "simd");
    test(thread1, TEST_TIMES, "thread1");
}

void test_all() {
    test(row_loop, TEST_TIMES, "row_loop");
    test(col_loop, TEST_TIMES, "col_loop");
    test(cache_11, TEST_TIMES, "cache_11");
    test(cache_12, TEST_TIMES, "cache_12");
    test(cache_21, TEST_TIMES, "cache_21");
    test(cache_22, TEST_TIMES, "cache_22");
    test(cache_31, TEST_TIMES, "cache_31");
    test(cache_32, TEST_TIMES, "cache_32");
    test(cache_33, TEST_TIMES, "cache_33");
    test(cache_41, TEST_TIMES, "cache_41");
    test(simd, TEST_TIMES, "simd");
    test(thread1, TEST_TIMES, "thread1");
}

int main() {
    test_all();
    return 0;
}