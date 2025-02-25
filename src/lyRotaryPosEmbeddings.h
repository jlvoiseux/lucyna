#include "lyTensor.h"

#include <lyOpenCL.h>
#include <stdint.h>

void lyRopePrecomputeFreqsCis(lyTensorDouble** ppOut, int32_t dim, int32_t end, double theta, lyOpenCLContext* pOpenCLContext);
void lyRopeApplyEmbeddings(lyTensor** ppXQOut, lyTensor** ppXKOut, lyTensor* pXQ, lyTensor* pXK, lyTensorDouble* pFreqsCis);