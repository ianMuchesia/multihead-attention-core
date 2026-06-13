import torch
import time


from src.multihead_attention import MultiHeadAttention



batch_size = 32
seq_len = 20
vocab_size = 1000


#fake input sentences
X = torch.randint(0,vocab_size,(batch_size,seq_len))

#fake target labels to calculate the loss against
Y = torch.randint(0,vocab_size,(batch_size,seq_len))