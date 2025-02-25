#pragma once

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <stdbool.h>
#include <stddef.h>

typedef struct lyOpenCLKernels
{
	cl_program tensorMathProgram;
	cl_kernel  matMulKernel;
	cl_kernel  scaleAndAddKernel;
	bool	   initialized;
} lyOpenCLKernels;

typedef struct lyOpenCLContext
{
	cl_platform_id	 platform;
	cl_device_id	 device;
	cl_context		 context;
	cl_command_queue queue;
	lyOpenCLKernels	 kernels;
	bool			 initialized;
} lyOpenCLContext;

void	   lyOpenCLInit(lyOpenCLContext** ppContext);
void	   lyOpenCLDestroy(lyOpenCLContext* pContext);
void	   lyOpenCLPrintDeviceInfo(lyOpenCLContext* pContext);
void	   lyOpenCLInitKernels(lyOpenCLContext* pContext);
void	   lyOpenCLReleaseKernels(lyOpenCLContext* pContext);
cl_program lyOpenCLBuildProgramFromFile(lyOpenCLContext* pContext, const char* filename);