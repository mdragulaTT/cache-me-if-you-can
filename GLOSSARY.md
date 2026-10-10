# Glossary

Every term the notebooks and talks use, in the order you meet them.

| Term | Meaning |
|---|---|
| **Token** | A piece of text, usually a word or part of a word. Models read and write text as tokens. |
| **Kernel** | A small program that runs on the chip's cores and performs one operation (a matrix multiplication, a softmax, ...). |
| **Warm-up** | Running the models once before you need them, so TT-NN compiles their kernels in advance. A kernel is compiled the first time an operation runs with a new tensor shape and kept on disk afterwards. |
| **Prefill** | The phase in which the model reads the whole prompt in one pass. |
| **Decode** | The phase in which the model writes the answer one token at a time; each token needs a full pass through the model. |
| **Time to first token (TTFT)** | Time from sending the prompt to receiving the first token of the answer. Mostly prefill. |
| **Tokens per second per user** | How fast the answer arrives for one user once it has started. |
| **DRAM** | The card's main memory (12 GB on an n150). Weights and the KV cache live here. |
| **SRAM** | The small, fast memory inside each core, where the arithmetic happens. |
| **Memory bandwidth** | How many bytes per second DRAM can deliver to the cores (about 288 GB/s on an n150). |
| **Memory-bound** | Limited by how fast data arrives from memory, not by arithmetic. Decode at small batch sizes. |
| **Compute-bound** | Limited by the arithmetic. Prefill of a long prompt. |
| **Trace** | Recording the list of commands the host sends for one decode step, then replaying the recording on the card for every later token, so the host does not send them again. |
| **Dispatch** | The host sending commands to the card. With trace on, this cost mostly disappears. |
| **Fused operation** | One operation that does the work of several (for example splitting q, k and v into heads), so fewer commands are sent. |
| **Tile** | A 32 x 32 block of numbers, the unit in which the hardware moves and multiplies data. |
| **Query, key, value (Q, K, V)** | The three vectors attention computes for every token: what it looks for, what it can be found by, and what it passes on. |
| **Causal mask** | The part of attention that stops a token from looking at later tokens. |
| **KV cache** | The stored keys and values of all earlier tokens, so they are not recomputed for every new token. Queries are not stored because each is used only once. |
| **Relative error** | The size of the difference between a result and a reference, divided by the size of the reference. Used to check our outputs against PyTorch; bfloat16 rounding gives a few percent. |
| **Pre-allocated KV cache** | The production form of the cache: allocated once at full size and written in place at the current position, so its shape never changes. Trace needs this. |
| **User** | Anything holding a conversation with the model at the same time: a person in a chat window, an agent calling the model in a loop, a script processing documents. |
| **Agent** | A program that calls a model repeatedly to complete a task. |
| **Batch, batch size** | A group of users processed together; one decode step produces one token for each of them. |
| **Throughput** | Tokens per second summed over all users. |
| **bfloat16 (bf16)** | A 16-bit number format: 1 sign bit, 8 exponent bits, 7 mantissa bits. |
| **bfloat8_b (bfp8)** | Block floating point: blocks of 16 numbers share one 8-bit exponent; each number keeps a sign and 7 mantissa bits. About 8.5 bits per number. |
| **bfloat4_b (bfp4)** | The same with 3 mantissa bits, about 4.5 bits per number. |
| **Precision preset** | tt_transformers' choice of number formats per group of weights: *accuracy* or *performance*. |
| **Budget** | The service promise to every user: the first token within a set time, then at least a set number of tokens per second. |
| **Scheduler** | The part of an inference server that decides which requests share a batch and in which order batches run. |
| **Tensor parallelism** | Splitting each layer of a model across several chips. |
| **Disaggregated serving** | Running prefill and decode on different chips and moving the KV cache between them. |
| **Continuous batching** | Rebuilding batches at every decode step as requests arrive and finish. |
| **Paged KV cache** | Storing the KV cache in fixed-size pages, like virtual memory. |
| **Speculative decoding** | A small draft model guesses several tokens; the large model checks them all in one pass. |
