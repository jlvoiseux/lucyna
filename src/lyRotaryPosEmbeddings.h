#include "lyTensor.h"

#include <stdint.h>

bool precomputeFreqsCis(lyTensor** ppOut, int32_t dim, int32_t end, float theta);
bool lyApplyRotaryEmbedding(lyTensor** ppXQOut, lyTensor** ppXKOut, const lyTensor* pXQ, const lyTensor* pXK, const lyTensor* pFreqsCis);