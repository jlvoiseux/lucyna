#include "lyUtil.h"

#include <stdio.h>

void printDeviceInfo(void)
{
	int			deviceCount;
	cudaError_t error = cudaGetDeviceCount(&deviceCount);
	if (error != cudaSuccess)
	{
		printf("CUDA Error: Failed to get device count: %s\n", cudaGetErrorString(error));
		return;
	}

	printf("CUDA Devices Found: %d\n\n", deviceCount);

	for (int i = 0; i < deviceCount; i++)
	{
		cudaDeviceProp props;
		error = cudaGetDeviceProperties(&props, i);
		if (error != cudaSuccess)
		{
			printf("CUDA Error: Failed to get device properties: %s\n", cudaGetErrorString(error));
			continue;
		}

		printf("Device %d: %s\n", i, props.name);
		printf("  Compute Capability: %d.%d\n", props.major, props.minor);
		printf("  Total Global Memory: %.2f GB\n", props.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
		printf("  Memory Clock Rate: %.0f MHz\n", props.memoryClockRate * 1e-3f);
		printf("  Memory Bus Width: %d bits\n", props.memoryBusWidth);
		printf("  L2 Cache Size: %d bytes\n\n", props.l2CacheSize);
	}
}