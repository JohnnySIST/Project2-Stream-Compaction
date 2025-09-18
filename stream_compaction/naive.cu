#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        __global__ void scan(int n, int *odata, const int *idata, int offset) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) return;
            if (index >= offset) {
                odata[index] = idata[index - offset] + idata[index];
            } else {
                odata[index] = idata[index];
            }
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
			int* d_idata, * d_odata;
			cudaMalloc((void**)&d_idata, n * sizeof(int));
			cudaMalloc((void**)&d_odata, n * sizeof(int));
			cudaMemcpy(d_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            timer().startGpuTimer();
            int blockSize = 256;
            int numBlocks = (n + blockSize - 1) / blockSize;
            for (int offset = 1; offset < n; offset *= 2) {
                scan << <numBlocks, blockSize >> > (n, d_odata, d_idata, offset);
                std::swap(d_odata, d_idata);
			}
            cudaMemcpy(odata, d_idata, n * sizeof(int), cudaMemcpyDeviceToHost);
            for (int i = n - 1; i > 0; i--) {
                odata[i] = odata[i - 1];
            }
			odata[0] = 0;
            timer().endGpuTimer();
            cudaFree(d_idata);
            cudaFree(d_odata);
        }
    }
}
