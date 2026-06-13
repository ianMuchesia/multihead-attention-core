# Multi-Head Attention Core

A PyTorch implementation and mathematical analysis of the Multi-Head Attention mechanism, the foundational component of Transformer neural networks.

## Overview

This repository explores the internal mechanics, memory bottlenecks, and computational complexity of multi-head attention. It includes a custom PyTorch module, rigorous shape-checking tests, and theoretical derivations, providing a clear bridge between theory and implementation.

## Project Structure

- **`src/`**: Contains the core `MultiHeadAttention` PyTorch implementation.
- **`tests/`**: Unit tests verifying correct tensor transformations and layer projection shapes.
- **`experiments/`**: Scripts measuring parameter counts and memory scaling requirements.
- **`math-notes/`**: Explanations detailing memory analysis, parameter counting, and Big-O computational complexity ($\mathcal{O}(n^2d)$).

## Status

**Work in Progress (WIP)**: The core implementation for query/key/value projection mappings and head splitting is established. Attention scaling and masked dot-product operations are currently being developed.

## Setup & Testing

```bash
git clone https://github.com/ianMuchesia/multihead-attention-core.git
cd multihead-attention-core
pytest tests/
```
