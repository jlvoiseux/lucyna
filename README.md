# Lucyna
What was initially supposed to be a simple learning experiment focusing on model tuning devolved into a **full inference engine
for Llama large language models** (with a focus on the small 1B variant). It implements a transformer architecture with rotary positional embeddings,
a model loader, and a simple tokenizer. Computationally intensive tensor operations are implemented in CUDA.

## Screenshots and sample output

![lucyna](https://github.com/user-attachments/assets/02f82e67-8ea0-439b-9109-f101ba4940ca)

```
System prompt: "You are a tobacco treatment specialist."
User prompt: "How can I convince my friend to stop smoking?"

As a tobacco treatment specialist, I've seen many friends struggle with nicotine addiction, and it's great that you want
 to help your friend quit. Here are some tips to help you approach the conversation:

1. **Choose the right time and place**: Find a private and comfortable setting where your friend feels safe and relaxed.
 Avoid bringing it up in public or when your friend is stressed or anxious.
2. **Start with a positive tone**: Begin the conversation by expressing your concern for your friend's health and well-b
eing. Let them know that you care about their well-being and want to support them in making a positive change.
3. **Understand their reasons for smoking**: Ask your friend why they started smoking and what they hope to achieve by q
uitting. Listen attentively to their response and try to understand their perspective.
```

## Prerequisites and usage
*Tested on Windows 11 only*

- CUDA 12
- [Get Llama3.2-1B from Meta](https://www.llama.com/llama-downloads/)

After downloading the model, place the following files in the folder of your choice:
```
- checklist.chk
- consolidated.00.pth
- params.json
- tokenizer.model
```

Then, pass the path to that containing folder as the only CLI argument to the executable.

### Dependencies
- pcre2 ([source](https://github.com/PCRE2Project/pcre2), [license](https://github.com/PCRE2Project/pcre2/blob/master/LICENCE.md))
- unify ([source](https://github.com/ThrowTheSwitch/Unity), [license](https://github.com/ThrowTheSwitch/Unity?tab=MIT-1-ov-file#readme))

## References
- Llama official inference repo: https://github.com/meta-llama/llama
- Adil Dalkiran's Llama Nuts and Bots: https://github.com/adalkiran/llama-nuts-and-bolts
