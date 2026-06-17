# Multi-Head Attention Core

PyTorch implementation and analysis of the Multi-Head Attention mechanism used in Transformer models.

## Highlights

- Core attention module with query, key, value, and output projections.
- Shape-focused tests for validating tensor transformations.
- Experiment scripts and generated artifacts for memory, speed, and attention-map analysis.
- Professional math notes covering memory scaling, parameter counts, and $\mathcal{O}(n^2d)$ complexity.

## Project Structure

- `src/` - Core model code and experiment runners.
- `tests/` - Unit tests for tensor shapes and attention behavior.
- `experiments/` - Training outputs, checkpoints, plots, and experiment summaries.
- `math-notes/` - Technical notes and complexity analysis.
- `notebooks/` - Exploratory analysis notebooks and visual inspection outputs.

## Current State

This repository is in an active research stage. The attention block is implemented, the experiment pipeline is producing comparison data, and the repository includes saved plots, checkpoints, and notebook-based analysis for the tested head configurations.

## Run Tests

```bash
pytest tests/
```
