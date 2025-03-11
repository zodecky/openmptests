#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "sorting.h"

/*
   bubble sort -- sequential, parallel --
*/


/// Performs a bubble pass on a subarray
/// MARK: Helper
///
/// - Parameters:
///   - T: subarray to be bubble-passed
///   - size: subarray size
bool bubble(uint64_t *T, uint64_t size) {
    bool sorted = true;
    for (uint64_t i = 0; i < size - 1; i++) {
        if (T[i] > T[i + 1]) {
            uint64_t temp = T[i];
            T[i] = T[i + 1];
            T[i + 1] = temp;
            sorted = false;
        }
    }
    return sorted;
}


/// Bubble sort sequential
/// - Parameters:
///   - T: array to be sorted
///   - size: array size
void sequential_bubble_sort(uint64_t *T, const uint64_t size) {
    bool sorted;
    do {
        sorted = bubble(T, size);
    } while (!sorted);
    return;
}

/// Parallel bubble sort
/// - Parameters:
///   - T: subarray to be bubble-passed
///   - size: subarray size
///
/// After testing a dynamic for schedule and also a static for schedule, even though dynamic made more sense logically,
/// because some chunks may be easier to compute than others, in all of the cases, static was always faster, probably because the gains
/// of using dynamic aren't large enough to surpass its cost.
///
/// After that, a more optimized implementation was made, so we reuse the threads and use a barrier, so all the threads wait, instead of killing them and starting new ones.
///
///
void parallel_bubble_sort(uint64_t *T, const uint64_t size) {
    int num_threads = omp_get_max_threads();
    if (size < num_threads * 2) {
        num_threads = size / 2;
        if (num_threads < 1)
            num_threads = 1;
    }
    int chunk_size = size / num_threads;
    
    bool sorted;
    do {
        bool global_sorted = true;
        
#pragma omp parallel default(none) shared(T, size, chunk_size, num_threads) reduction(&&:global_sorted)
        {
            bool local_sorted = true;
            int t = omp_get_thread_num();
            int start = t * chunk_size;
            int end = (t == num_threads - 1) ? size : (start + chunk_size);
            
            // Only run bubble if there's more than one element.
            if (end - start > 1) {
                local_sorted = bubble(&T[start], end - start);
            }
            
            #pragma omp barrier
            
            // Check border swap with the next chunk if not the last thread.
            if (t < num_threads - 1) {
                int border = (t + 1) * chunk_size - 1;
                if (T[border] > T[border + 1]) {
                    uint64_t temp = T[border];
                    T[border] = T[border + 1];
                    T[border + 1] = temp;
                    local_sorted = false;
                }
            }
            
            // The reduction will combine each thread's local_sorted.
            global_sorted = local_sorted;
        }
        
        sorted = global_sorted;
    } while (!sorted);
}

int main(int argc, char **argv) {
    // Init cpu_stats to measure CPU cycles and elapsed time
    struct cpu_stats *stats = cpu_stats_init();

    unsigned int exp;

    /* the program takes one parameter N which is the size of the array to
       be sorted. The array will have size 2^N */
    if (argc != 2) {
        fprintf(stderr, "Usage: bubble.run N \n");
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

        sequential_bubble_sort(X, N);

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

    println_cpu_stats_report("bubble serial", average_report(experiments, NB_EXPERIMENTS));

    for (exp = 0; exp < NB_EXPERIMENTS; exp++) {
#ifdef RINIT
        init_array_random(X, N);
#else
        init_array_sequence(X, N);
#endif

        cpu_stats_begin(stats);

        parallel_bubble_sort(X, N);

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

    println_cpu_stats_report("bubble parallel", average_report(experiments, NB_EXPERIMENTS));

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

    sequential_bubble_sort(Y, N);
    parallel_bubble_sort(Z, N);

    if (!are_vector_equals(Y, Z, N)) {
        fprintf(stderr, "ERROR: sorting with the sequential and the parallel algorithm does not give the same result\n");
        exit(-1);
    }

    free(X);
    free(Y);
    free(Z);
}
