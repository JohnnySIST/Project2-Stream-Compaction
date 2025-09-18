#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
			odata[0] = 0;
            for (int i = 1; i < n; i++) {
                odata[i] = odata[i - 1] + idata[i - 1];
			}
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
			int i = 0, j = 0;
			while (i < n) {
				if (idata[i] != 0) {
					odata[j++] = idata[i];
				}
                i++;
			}
            timer().endCpuTimer();
            return j;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            int* b = new int[n], * sum = new int[n];
            timer().startCpuTimer();
			for (int i = 0; i < n; i++) {
				b[i] = (idata[i] != 0) ? 1 : 0;
			}
            sum[0] = 0;
            for (int i = 1; i < n; i++) {
                sum[i] = sum[i - 1] + b[i - 1];
            }
			int count = (n > 0) ? sum[n - 1] + b[n - 1] : 0;
			for (int i = 0; i < n; i++) {
				if (b[i] == 1) {
					odata[sum[i]] = idata[i];
				}
			}
            timer().endCpuTimer();
            delete[] b;
            delete[] sum;
            return count;
        }
    }
}
