#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include <stdbool.h>

#include "sorting.h"

/*
   odd-even sort -- sequential, parallel --
*/

void sequential_oddeven_sort(uint64_t *T, const uint64_t size) {
    bool sorted;
    do {
        sorted = true;
        // Even phase: compare indices 0, 2, 4, …, size-2
        for (uint64_t i = 0; i < size - 1; i += 2) {
            if (T[i] > T[i + 1]) {
                uint64_t temp = T[i];
                T[i] = T[i + 1];
                T[i + 1] = temp;
                sorted = false;
            }
        }
        // Odd phase: compare indices 1, 3, 5, …, size-2
        for (uint64_t i = 1; i < size - 1; i += 2) {
            if (T[i] > T[i + 1]) {
                uint64_t temp = T[i];
                T[i] = T[i + 1];
                T[i + 1] = temp;
                sorted = false;
            }
        }
    } while (!sorted);
}

void parallel_oddeven_sort(uint64_t *T, const uint64_t size) {
    bool sorted;
    do {
        sorted = true;
        // Even phase: process even-indexed pairs in parallel.
        #pragma omp parallel for schedule(static) reduction(&& : sorted)
        for (int i = 0; i < (int)size - 1; i += 2) {
            if (T[i] > T[i + 1]) {
                uint64_t temp = T[i];
                T[i] = T[i + 1];
                T[i + 1] = temp;
                sorted = false;
            }
        }
        // Odd phase: process odd-indexed pairs in parallel.
        #pragma omp parallel for schedule(static) reduction(&& : sorted)
        for (int i = 1; i < (int)size - 1; i += 2) {
            if (T[i] > T[i + 1]) {
                uint64_t temp = T[i];
                T[i] = T[i + 1];
                T[i + 1] = temp;
                sorted = false;
            }
        }
    } while (!sorted);
}

int main(int argc, char **argv) {
    // Init cpu_stats to measure CPU cycles and elapsed time
    struct cpu_stats *stats = cpu_stats_init();

    unsigned int exp;

    /* the program takes one parameter N which is the size of the array to
       be sorted. The array will have size 2^N */
    if (argc != 2) {
        fprintf(stderr, "Usage: odd-even.run N \n");
        exit(-1);
    }

    uint64_t N = 1 << (atoi(argv[1]));
    /* the array to be sorted */
    uint64_t *X = (uint64_t *)malloc(N * sizeof(uint64_t));

    printf("--> Sorting an array of size %lu\n", N);
#ifdef RINIT
    printf("--> The array is initialized randomly\n");
#endif

    for (exp = 0; exp < NB_EXPERIMENTS; exp++) {
#ifdef RINIT
        init_array_random(X, N);
#else
        init_array_sequence(X, N);
#endif

        cpu_stats_begin(stats);

        sequential_oddeven_sort(X, N);

        experiments[exp] = cpu_stats_end(stats);

        /* verifying that X is properly sorted */
#ifdef RINIT
        if (!is_sorted(X, N)) {
            print_array(X, N);
            fprintf(stderr, "ERROR: the sequential sorting of the array failed\n");
            exit(-1);
        }
#else
        if (!is_sorted_sequence(X, N)) {
            print_array(X, N);
            fprintf(stderr, "ERROR: the sequential sorting of the array failed\n");
            exit(-1);
        }
#endif
    }

    println_cpu_stats_report("odd-even serial", average_report(experiments, NB_EXPERIMENTS));

    for (exp = 0; exp < NB_EXPERIMENTS; exp++) {
#ifdef RINIT
        init_array_random(X, N);
#else
        init_array_sequence(X, N);
#endif

        cpu_stats_begin(stats);

        parallel_oddeven_sort(X, N);

        experiments[exp] = cpu_stats_end(stats);

        /* verifying that X is properly sorted */
#ifdef RINIT
        if (!is_sorted(X, N)) {
            print_array(X, N);
            fprintf(stderr, "ERROR: the parallel sorting of the array failed\n");
            exit(-1);
        }
#else
        if (!is_sorted_sequence(X, N)) {
            print_array(X, N);
            fprintf(stderr, "ERROR: the parallel sorting of the array failed\n");
            exit(-1);
        }
#endif
    }

    println_cpu_stats_report("odd-even parallel", average_report(experiments, NB_EXPERIMENTS));

    /* print_array (X, N) ; */

    /* before terminating, we run one extra test of the algorithm */
    uint64_t *Y = (uint64_t *)malloc(N * sizeof(uint64_t));
    uint64_t *Z = (uint64_t *)malloc(N * sizeof(uint64_t));

#ifdef RINIT
    init_array_random(Y, N);
#else
    init_array_sequence(Y, N);
#endif

    memcpy(Z, Y, N * sizeof(uint64_t));

    sequential_oddeven_sort(Y, N);
    parallel_oddeven_sort(Z, N);

    if (!are_vector_equals(Y, Z, N)) {
        fprintf(stderr, "ERROR: sorting with the sequential and the parallel algorithm does not give the same result\n");
        exit(-1);
    }

    free(X);
    free(Y);
    free(Z);
}
