# Implementation Notes & Learnings

## Verifying Tensor Shapes in PyTorch
When writing shape assertions for tests, calling `.shape` returns a `torch.Size` object. This behaves exactly like a standard Python tuple and can be compared directly:

```python
assert weight.shape == (32, 4, 20, 20), f"Weight shape mismatch. Expected (32, 4, 20, 20), got {weight.shape}"
```

## Input Data Types for Attention Layers
A common misunderstanding is assuming that transformer inputs should be discrete integer tokens (e.g., using `torch.randint` for word IDs).

**Correction**: While a language model initially takes discrete vocabulary token IDs, these integers pass through an `nn.Embedding` layer. The embedding layer acts as a lookup table, converting integer IDs into dense continuous floating-point vectors (e.g., a 128-dimensional vector). Our isolated attention mechanism receives inputs *after* this embedding step, meaning it expects continuous floats (`torch.randn`).

## Why Linear Layers Require Floats
The underlying `nn.Linear` modules inside the Multi-Head Attention block fundamentally reject integer inputs because neural networks learn via backpropagation:
- Calculating gradients (derivatives) to find the slope of the loss function requires a continuous, smooth mathematical space.
- Integer inputs create step functions ("staircases"), which lack smooth derivatives.
- Floating-point representations allow the network to calculate fractional gradients and make microscopic weight adjustments.
