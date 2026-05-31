# Attention

Attention is a mechanism that allows a model to focus on relevant parts of the input sequence.

In Transformer models, scaled dot-product attention computes similarity scores between queries and keys. These scores are normalized with softmax and used to weight the values.

Causal attention prevents a decoder-only language model from seeing future tokens during training.
