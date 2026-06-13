import torch
from src.multihead_attention import MultiHeadAttention



def verify_output_shapes():
    attention_head = MultiHeadAttention(128,4)
    
    
    x = torch.randn(32,20,128)
    
    
    output , weight = attention_head.forward(x,x,x,None)
    
    
    assert output.shape == (32,20,128), f"output shape mismatch. Expected (32,20,128), got {output.shape}"
    
    assert weight.shape == (32, 4, 20, 20), f"Weight shape mismatch. Expected (32, 4, 20, 20), got {weight.shape}"
    
    
    
if __name__ == "__main__":
    verify_output_shapes()
    print("All shape tests passed successfully!")
    
    
    
    
    
    