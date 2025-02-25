#include "lyOpenCL.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char* readKernelSource(const char* filename)
{
	FILE* file = fopen(filename, "r");
	if (!file)
	{
		fprintf(stderr, "Failed to open kernel file: %s\n", filename);
		return NULL;
	}

	fseek(file, 0, SEEK_END);
	size_t size = ftell(file);
	rewind(file);

	char* source = (char*)malloc(size + 1);
	if (!source)
	{
		fclose(file);
		return NULL;
	}

	size_t read = fread(source, 1, size, file);
	fclose(file);

	source[read] = '\0';
	return source;
}

void lyOpenCLInit(lyOpenCLContext** ppContext)
{
	lyOpenCLContext* pContext = (lyOpenCLContext*)malloc(sizeof(lyOpenCLContext));
	memset(pContext, 0, sizeof(lyOpenCLContext));

	cl_int	status;
	cl_uint numPlatforms;
	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	if (status != CL_SUCCESS || numPlatforms == 0)
	{
		fprintf(stderr, "Failed to find any OpenCL platforms\n");
		free(pContext);
		return;
	}

	cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * numPlatforms);
	status					  = clGetPlatformIDs(numPlatforms, platforms, NULL);
	if (status != CL_SUCCESS)
	{
		fprintf(stderr, "Failed to get OpenCL platforms\n");
		free(platforms);
		free(pContext);
		return;
	}

	// Just pick the first platform for now
	pContext->platform = platforms[0];

	cl_uint numDevices;
	status = clGetDeviceIDs(pContext->platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);

	// If no GPU devices found, try CPU
	if (status != CL_SUCCESS || numDevices == 0)
	{
		status = clGetDeviceIDs(pContext->platform, CL_DEVICE_TYPE_CPU, 0, NULL, &numDevices);
		if (status != CL_SUCCESS || numDevices == 0)
		{
			fprintf(stderr, "Failed to find any OpenCL devices\n");
			free(platforms);
			free(pContext);
			return;
		}
	}

	cl_device_id* devices = (cl_device_id*)malloc(sizeof(cl_device_id) * numDevices);
	status				  = clGetDeviceIDs(pContext->platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);

	// If no GPU devices found, try CPU
	if (status != CL_SUCCESS)
	{
		status = clGetDeviceIDs(pContext->platform, CL_DEVICE_TYPE_CPU, numDevices, devices, NULL);
		if (status != CL_SUCCESS)
		{
			fprintf(stderr, "Failed to get OpenCL devices\n");
			free(platforms);
			free(devices);
			free(pContext);
			return;
		}
	}

	// Just pick the first device for now
	pContext->device = devices[0];

	pContext->context = clCreateContext(NULL, 1, &pContext->device, NULL, NULL, &status);
	if (status != CL_SUCCESS)
	{
		fprintf(stderr, "Failed to create OpenCL context\n");
		free(platforms);
		free(devices);
		free(pContext);
		return;
	}

	pContext->queue = clCreateCommandQueue(pContext->context, pContext->device, 0, &status);
	if (status != CL_SUCCESS)
	{
		fprintf(stderr, "Failed to create OpenCL command queue\n");
		clReleaseContext(pContext->context);
		free(platforms);
		free(devices);
		free(pContext);
		return;
	}

	pContext->initialized = true;

	lyOpenCLInitKernels(pContext);

	free(platforms);
	free(devices);

	*ppContext = pContext;
}

void lyOpenCLDestroy(lyOpenCLContext* pContext)
{
	if (!pContext)
	{
		return;
	}

	lyOpenCLReleaseKernels(pContext);

	if (pContext->queue)
	{
		clReleaseCommandQueue(pContext->queue);
	}

	if (pContext->context)
	{
		clReleaseContext(pContext->context);
	}

	free(pContext);
}

void lyOpenCLPrintDeviceInfo(lyOpenCLContext* pContext)
{
	if (!pContext || !pContext->initialized)
	{
		printf("OpenCL context not initialized\n");
		return;
	}

	char	 name[1024];
	char	 vendor[1024];
	cl_uint	 computeUnits;
	cl_ulong globalMemSize;
	cl_ulong localMemSize;

	clGetDeviceInfo(pContext->device, CL_DEVICE_NAME, sizeof(name), name, NULL);
	clGetDeviceInfo(pContext->device, CL_DEVICE_VENDOR, sizeof(vendor), vendor, NULL);
	clGetDeviceInfo(pContext->device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(computeUnits), &computeUnits, NULL);
	clGetDeviceInfo(pContext->device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(globalMemSize), &globalMemSize, NULL);
	clGetDeviceInfo(pContext->device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(localMemSize), &localMemSize, NULL);

	printf("OpenCL Device: %s\n", name);
	printf("  Vendor: %s\n", vendor);
	printf("  Compute Units: %u\n", computeUnits);
	printf("  Global Memory: %.2f GB\n", globalMemSize / (1024.0 * 1024.0 * 1024.0));
	printf("  Local Memory: %.2f KB\n", localMemSize / 1024.0);
}

cl_program lyOpenCLBuildProgramFromFile(lyOpenCLContext* pContext, const char* filename)
{
	char* source = readKernelSource(filename);
	if (!source)
	{
		return NULL;
	}

	cl_int	   status;
	cl_program program = clCreateProgramWithSource(pContext->context, 1, (const char**)&source, NULL, &status);
	free(source);

	if (status != CL_SUCCESS)
	{
		fprintf(stderr, "Failed to create OpenCL program: %d\n", status);
		return NULL;
	}

	status = clBuildProgram(program, 1, &pContext->device, NULL, NULL, NULL);
	if (status != CL_SUCCESS)
	{
		char log[16384];
		clGetProgramBuildInfo(program, pContext->device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
		fprintf(stderr, "OpenCL program build error: %s\n", log);
		clReleaseProgram(program);
		return NULL;
	}

	return program;
}

void lyOpenCLInitKernels(lyOpenCLContext* pContext)
{
	if (!pContext || !pContext->initialized)
	{
		return;
	}

	pContext->kernels.tensorMathProgram = lyOpenCLBuildProgramFromFile(pContext, "../src/kernels.cl");
	if (!pContext->kernels.tensorMathProgram)
	{
		fprintf(stderr, "Failed to build tensor math program\n");
		return;
	}

	cl_int status;
	pContext->kernels.matMulKernel = clCreateKernel(pContext->kernels.tensorMathProgram, "matMulKernel", &status);
	if (status != CL_SUCCESS)
	{
		fprintf(stderr, "Failed to create matMul kernel: %d\n", status);
		clReleaseProgram(pContext->kernels.tensorMathProgram);
		return;
	}

	pContext->kernels.scaleAndAddKernel = clCreateKernel(pContext->kernels.tensorMathProgram, "scaleAndAddKernel", &status);
	if (status != CL_SUCCESS)
	{
		fprintf(stderr, "Failed to create scaleAndAdd kernel: %d\n", status);
		clReleaseKernel(pContext->kernels.matMulKernel);
		clReleaseProgram(pContext->kernels.tensorMathProgram);
		return;
	}

	pContext->kernels.initialized = true;
}

void lyOpenCLReleaseKernels(lyOpenCLContext* pContext)
{
	if (!pContext || !pContext->kernels.initialized)
	{
		return;
	}

	if (pContext->kernels.matMulKernel)
	{
		clReleaseKernel(pContext->kernels.matMulKernel);
	}

	if (pContext->kernels.scaleAndAddKernel)
	{
		clReleaseKernel(pContext->kernels.scaleAndAddKernel);
	}

	if (pContext->kernels.tensorMathProgram)
	{
		clReleaseProgram(pContext->kernels.tensorMathProgram);
	}

	pContext->kernels.initialized = false;
}