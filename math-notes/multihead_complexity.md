# Multi-Head Attention Complexity & Scaling Analysis

## 1. Memory Analysis (Attention Matrix Scaling)
**The Bottleneck:** The attention mechanism calculates how much every word relates to every other word in the sequence. This creates an $n \times n$ attention matrix (where $n$ is the sequence length).

**Proof (Batch = 32, Seq = 20):**
- **1 Head:** The weights tensor shape is `(32, 1, 20, 20)`. Total elements stored in memory = $12,800$.
- **8 Heads:** The weights tensor shape is `(32, 8, 20, 20)`. Total elements stored in memory = $102,400$.

**Takeaway:** Adding heads massively multiplies the RAM required to store attention maps. This is the primary reason why scaling context windows in Large Language Models (LLMs) is highly memory-intensive.

## 2. Parameter Counting ($d_{model} = 128$, heads = 4)
**Single Linear Layer ($W_Q$):**
- **Weight matrix**: 128 input neurons connected to 128 output neurons = $16,384$ connections.
- **Bias vector**: 1 adjustable unit added to each of the 128 outputs = $128$.
- **Total per layer**: $16,384 + 128 = 16,512$ parameters.

**Multi-Head Block Total:**
- The architecture features exactly four projection layers: $W_Q, W_K, W_V,$ and $W_O$.
- **Total parameters**: $16,512 \times 4 = 66,048$.

## 3. Computational Complexity: $\mathcal{O}(n^2d)$
- **The Output Grid ($n^2$):** Multiplying the Query matrix $Q$ and the transposed Key matrix $K^T$ results in an $n \times n$ matrix. There are $n^2$ total cells to compute.
- **The Work Per Cell ($d$):** To calculate a single cell, we perform a dot product between a row from $Q$ and a column from $K^T$. Since the sequence representations use embedding dimension $d$, taking the dot product requires $d$ individual multiplications.
- **Total Computational Work:** $(\text{Total cells}) \times (\text{Multiplications per cell}) = n^2 \times d$.
