# Multi-Head Attention Complexity Comparison

## Experiment Setup
- **Batch Size**: 32
- **Sequence Length** ($n$): 1024
- **Embedding Dimension** ($d_{model}$): 512
- **Data Type**: Synthetic continuous standard normal distribution (`torch.randn`)
- **Hardware**: Google Colab T4 GPU
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: Mean Squared Error (MSE)

## Results

| Heads | Training Loss | Validation Loss | Training Time / Step (s) | Peak Memory (MB) |
|-------|---------------|-----------------|--------------------------|------------------|
| 1     | 0.9992        | 1.0000          | 0.107                    | 1070.27          |
| 4     | 0.9992        | 1.0000          | 0.130                    | 2606.27          |
| 8     | 0.9992        | 1.0000          | 0.168                    | 4654.27          |

## Analysis

### 1. Memory Usage (The $\mathcal{O}(n^2)$ Bottleneck)

The data shows a massive spike in VRAM allocation, jumping from 1.07 GB (1 head) to 4.65 GB (8 heads). This isolates the primary hardware bottleneck of the Transformer architecture: the attention weights matrix.

- The shape of the attention matrix is `(batch_size, num_heads, seq_len, seq_len)`.
- At **1 head**, the matrix contains $32 \times 1 \times 1024 \times 1024 = 33,554,432$ elements.
- At **8 heads**, the matrix contains $32 \times 8 \times 1024 \times 1024 = 268,435,456$ elements.

In 32-bit floating-point precision (4 bytes per float), the raw 8-head attention maps consume over 1 GB of memory. However, the recorded peak memory is 4.65 GB. This inflation is caused by the training process:
- **Forward Pass Caching**: PyTorch must keep the intermediate forward pass tensors in memory to compute gradients during the backward pass.
- **Adam Optimizer**: Adam maintains two exponentially decaying moving averages for every parameter, effectively tripling the memory required for the model weights and gradients.

### 2. Execution Time (Compute-Bound vs. Memory-Bound)

Theoretically, the total number of mathematical operations (FLOPs) is identical across all three configurations. A single head mapping 512 dimensions requires the same number of multiplications as 8 parallel heads mapping 64 dimensions each.

Despite this, the execution time increased from 0.107s to 0.168s. This indicates the 8-head configuration is presumably **memory-bound**, not compute-bound. The GPU cores can perform the matrix multiplications faster, but latency is added waiting for the massive attention grids to be read/written between the physical GPU VRAM and the compute cores.
