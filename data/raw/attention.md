# Attention

Attention is a mechanism that allows a model to focus on relevant parts of an input sequence. Instead of treating all tokens equally, attention assigns different weights to different tokens according to their relevance. This is one of the core ideas behind Transformer models.

In scaled dot-product attention, each token is projected into three vectors: query, key, and value. The query represents what the current token is looking for. The key represents what each token offers. The value contains the information that will be aggregated. Attention scores are computed by comparing queries with keys.

The standard scaled dot-product attention formula computes the dot product between queries and keys, divides by the square root of the key dimension, applies softmax, and uses the result to weight the values. The scaling factor prevents dot products from becoming too large when the vector dimension increases.

Multi-head attention runs several attention operations in parallel. Each head can focus on different relationships in the sequence. One head may focus on local syntax, another may focus on long-range dependency, and another may capture repeated patterns. The outputs of all heads are concatenated and projected back to the model dimension.

Causal attention is used in decoder-only language models. It prevents a token from attending to future tokens. During training, the model predicts the next token, so it must not see the answer in advance. A causal mask sets attention scores for future positions to negative infinity before softmax.

Attention is different from a simple fixed-size context window. The context window defines how many tokens are available, while attention determines which tokens inside that window are more important. In a Transformer block, attention is usually combined with residual connections, layer normalization, and a feed-forward network.
