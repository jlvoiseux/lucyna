#include "lyUtil.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

unsigned char* lyBase64Decode(const char* input, size_t inLen, size_t* outLen)
{
	static unsigned char	   decode_table[256] = {0};
	static const unsigned char base64_table[65]	 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

	static bool initialized = false;
	if (!initialized)
	{
		memset(decode_table, 0xFF, sizeof(decode_table));
		for (int i = 0; i < 64; i++)
		{
			decode_table[base64_table[i]] = i;
		}
		initialized = true;
	}

	size_t pad = 0;
	if (inLen > 0)
	{
		if (input[inLen - 1] == '=')
			pad++;
		if (inLen > 1 && input[inLen - 2] == '=')
			pad++;
	}

	*outLen				  = (inLen * 3) / 4 - pad;
	unsigned char* output = (unsigned char*)malloc(*outLen + 1);
	if (!output)
		return NULL;

	size_t	 i = 0, j = 0;
	uint32_t accum = 0;
	int		 bits  = 0;

	for (i = 0; i < inLen; i++)
	{
		if (input[i] == '=')
			break;

		unsigned char val = decode_table[(unsigned char)input[i]];
		if (val == 0xFF)
			continue;

		accum = (accum << 6) | val;
		bits += 6;

		if (bits >= 8)
		{
			bits -= 8;
			if (j < *outLen)
			{
				output[j++] = (accum >> bits) & 0xFF;
			}
		}
	}

	output[*outLen] = '\0';
	return output;
}