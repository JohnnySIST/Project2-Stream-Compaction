#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <iostream>

#define NUM_PER_BLOCK 1024

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        int nextPow2(int n) {
            int ret = 1;
			while (ret < n) ret <<= 1;
            return ret;
        }

        __global__ void scan(int n, int *idata) {
			__shared__ int temp[NUM_PER_BLOCK];
            n = NUM_PER_BLOCK;
            int index = threadIdx.x + (blockIdx.x * blockDim.x);

            if (index < n) {
                temp[index] = idata[index];
            }

            int offset = 1;
            for (int d = n>>1; d > 0; d >>= 1) {
                if (index < d) {
                    int ai = offset * (2 * index + 2) - 1;
                    int bi = offset * (2 * index + 1) - 1;
                    if (ai >= n || bi >= n) continue;
                    temp[ai] += temp[bi];
                }
                offset <<= 1;
                __syncthreads();
            }

            // clear the last element
			if (index == 0) {
                temp[n - 1] = 0;
            }
            __syncthreads();

            offset = n >> 1;
            for (int d = 1; d < n; d <<= 1) {
                if (index < d) {
                    int ai = offset * (2 * index + 2) - 1;
                    int bi = offset * (2 * index + 1) - 1;
					if (ai >= n || bi >= n) continue;
                    int t = temp[bi];
                    temp[bi] = temp[ai];
                    temp[ai] += t;
                }
				offset >>= 1;
                __syncthreads();
            }

            if (index < n) {
                idata[index] = temp[index];
			}
        }

        __global__ void upsweep(int n, int* data, int d) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
			int offset = (n >> 1) / d;
            if (index < d) {
                int ai = offset * (2 * index + 2) - 1;
                int bi = offset * (2 * index + 1) - 1;
                data[ai] += data[bi];
            }
        }

        __global__ void downsweep(int n, int* data, int d) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            int offset = (n >> 1) / d;
            if (index < d) {
                int ai = offset * (2 * index + 2) - 1;
                int bi = offset * (2 * index + 1) - 1;
                int t = data[bi];
                data[bi] = data[ai];
                data[ai] += t;
            }
        }

        void scanOnDevice(int N, int* d_idata) {
            int blockSize = 256;
            int numBlocks = (N + blockSize - 1) / blockSize;
            for (int d = N >> 1; d > 0; d >>= 1) {
                upsweep << <(d + blockSize - 1) / blockSize, blockSize >> > (N, d_idata, d);
            }
            //cudaMemcpy(odata, d_idata, n * sizeof(int), cudaMemcpyDeviceToHost);
            //for (int i = 0; i < n; i++) {
            //	std::cout << odata[i] << " ";
            //}
            //std::cout << std::endl;
            cudaMemset(d_idata + N - 1, 0, sizeof(int));
            for (int d = 1; d < N; d <<= 1) {
                downsweep << <(d + blockSize - 1) / blockSize, blockSize >> > (N, d_idata, d);
            }
            //cudaMemcpy(odata, d_idata, n * sizeof(int), cudaMemcpyDeviceToHost);
            //for (int i = 0; i < n; i++) {
            //    std::cout << odata[i] << " ";
            //}
            //std::cout << std::endl;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			int *d_idata;
			int N = nextPow2(n);
			cudaMalloc((void**)&d_idata, N * sizeof(int));
			cudaMemset(d_idata, 0, N * sizeof(int));
			cudaMemcpy(d_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            timer().startGpuTimer();
			scanOnDevice(N, d_idata);
            timer().endGpuTimer();
			cudaMemcpy(odata, d_idata, n * sizeof(int), cudaMemcpyDeviceToHost);
			cudaFree(d_idata);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            int* d_bools;
			int* d_indices;
			int* d_idata;
			int* d_odata;
			int N = nextPow2(n);
			cudaMalloc((void**)&d_bools, n * sizeof(int));
			cudaMalloc((void**)&d_indices, N * sizeof(int));
			cudaMalloc((void**)&d_idata, n * sizeof(int));
			cudaMalloc((void**)&d_odata, n * sizeof(int));
            cudaMemcpy(d_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            timer().startGpuTimer();
			int blockSize = 256;
			StreamCompaction::Common::kernMapToBoolean << <(n + blockSize - 1) / blockSize, blockSize >> > (n, d_bools, d_idata);
			cudaMemcpy(d_indices, d_bools, n * sizeof(int), cudaMemcpyDeviceToDevice);
			scanOnDevice(N, d_indices);
            StreamCompaction::Common::kernScatter << < (n + blockSize - 1) / blockSize, blockSize >> > (n, d_odata, d_idata, d_bools, d_indices);
            timer().endGpuTimer();
            int sum;
			cudaMemcpy(&sum, d_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
			if (idata[n - 1] != 0) sum++;
			cudaMemcpy(odata, d_odata, sum * sizeof(int), cudaMemcpyDeviceToHost);
			cudaFree(d_bools);
			cudaFree(d_indices);
			cudaFree(d_idata);
			cudaFree(d_odata);
            return sum;
        }
    }
}
