# Multi-Head Attention Complexity & Scaling Analysis

## Overview
This note summarizes the two main costs of multi-head attention: quadratic memory growth in sequence length and linear work per attention score with respect to embedding dimension.

## 1. Memory Analysis
The bottleneck is the attention matrix, which compares every token with every other token in the sequence. For a sequence of length $n$, the attention map has shape $n \times n$.

For the reference batch below:
- Batch size = 32
- Sequence length = 20

The attention weights tensor has shape:
- 1 head: `(32, 1, 20, 20)` with $12,800$ stored values
- 8 heads: `(32, 8, 20, 20)` with $102,400$ stored values

This shows why increasing the number of heads raises memory pressure: the attention map grows with both sequence length and head count.

## 2. Parameter Counting
For $d_{model} = 128$ and 4 heads, each projection layer contains:
- Weight matrix: $128 \times 128 = 16,384$ parameters
- Bias vector: $128$ parameters
- Total per layer: $16,512$ parameters

The full block uses four learned projections: $W_Q$, $W_K$, $W_V$, and $W_O$.

Total parameters:
$$16,512 \times 4 = 66,048$$

## 3. Computational Complexity
The attention score matrix is formed by multiplying $Q$ by $K^T$. That produces $n^2$ output cells.

Each cell is a dot product across $d$ dimensions, so the cost per cell is proportional to $d$.

Therefore, the total work is:
$$\mathcal{O}(n^2 d)$$

## Interpretation
The quadratic term comes from comparing every token to every other token. The linear $d$ term comes from the size of each token representation used in the dot product.