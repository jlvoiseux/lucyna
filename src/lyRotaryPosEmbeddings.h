#include "lyTensor.h"

#include <stdint.h>

void lyRopePrecomputeFreqsCis(lyTensor** ppOut, int32_t dim, int32_t end, float theta);
void lyRopeApplyEmbeddings(lyTensor** ppXQOut, lyTensor** ppXKOut, lyTensor* pXQ, lyTensor* pXK, lyTensor* pFreqsCis);