#pragma once

#include "lyTokenizer.h"

#include <stdint.h>

bool lyLoadTokenizer(lyTokenizer** ppTokenizer, const char* modelDir);
void lyDestroyTokenizer(lyTokenizer* pTokenizer);
