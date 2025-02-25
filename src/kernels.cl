__kernel void matMulKernel(__global const ushort* A, __global const ushort* B, __global float* C,
                          const int m, const int n, const int k) {
    int row = get_global_id(1);
    int col = get_global_id(0);

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            uint aVal_bits = ((uint)A[row * k + i]) << 16;
            uint bVal_bits = ((uint)B[i * n + col]) << 16;
            float aVal = as_float(aVal_bits);
            float bVal = as_float(bVal_bits);
            sum += aVal * bVal;
        }
        C[row * n + col] = sum;
    }
}

__kernel void scaleAndAddKernel(__global const ushort* A, __global const ushort* B,
                               __global ushort* C, __global const int* aStrides,
                               __global const int* bStrides, __global const int* aShape,
                               const int aRank, const int bRank, const float alpha,
                               const float beta, const int totalElements) {
    int idx = get_global_id(0);
    if (idx >= totalElements)
        return;

    uint aVal_bits = ((uint)A[idx]) << 16;
    float aVal = as_float(aVal_bits) * alpha;

    if (B != NULL) {
        int bIdx = 0;
        int temp = idx;

        for (int j = 0; j < bRank; j++) {
            int aRankOffset = aRank - bRank + j;
            int dimIdx = (temp / aStrides[aRankOffset]) % aShape[aRankOffset];
            bIdx += dimIdx * bStrides[j];
            temp %= aStrides[aRankOffset];
        }

        uint bVal_bits = ((uint)B[bIdx]) << 16;
        float bVal = as_float(bVal_bits);

        float result = aVal + bVal * beta;
        uint result_bits = as_uint(result);
        C[idx] = (ushort)(result_bits >> 16);
    } else {
        uint result_bits = as_uint(aVal);
        C[idx] = (ushort)(result_bits >> 16);
    }
}