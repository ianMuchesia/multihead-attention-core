import torch
import torch.nn as nn
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


from src.multihead_attention import MultiHeadAttention


#1. Setup
batch_size = 1
seq_len = 10
vocab_size = 1000
d_model = 128
heads_to_test = [1,4,8]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#fake input sentences
X = torch.randn(batch_size, seq_len, d_model).to(device)



for num_heads in heads_to_test:
    model = MultiHeadAttention(d_model,num_heads).to(device)
    
    
    # Step 2: Load the state dictionary from the .pth file
    weights_path = f'experiments/trained_{num_heads}head.pth'
    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))

    # Step 3: Load the weights into your instantiated model class
    model.load_state_dict(state_dict)

    
    
    model.eval()
    
    
    _,weights = model(X,X,X)
    
    
    newest_token_weights = weights[-1,:,:,:]
    
    matrix_to_plot = newest_token_weights.detach().cpu().numpy()
    
    
    for i in range(num_heads):
        
        head_matrix = matrix_to_plot[i,:,:]
        
        plt.figure(figsize=(14, 8))

        # 2. Configure the heatmap
        sns.heatmap(head_matrix, 
                    cmap='viridis', 
                    # 3. Use an accurate label for attention weights (0.0 to 1.0)
                    cbar_kws={'label': f'Attention Weight (Probability)- head {i} of {num_heads} heads'})

        # 4. Apply RIGOROUS, accurate professional labels
        plt.title("Attention Alignment Heatmap (Multiple Heads)", fontsize=16, pad=20)

        # The generated tokens (rows) are Queries asking the source tokens (columns)
        plt.ylabel("Query Tokens (Looking for Context)", fontsize=12, labelpad=10) 
        plt.xlabel("Key Tokens (Context Provided)", fontsize=12, labelpad=20)

        # Save the final figure
        filepath = Path(f'./experiments/attention_weights head {i+1} of {num_heads} heads.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')

        plt.show()
        print(f"Heatmap successfully saved to: {filepath}")

    
    
X = torch.randn(batch_size, seq_len, 512).to(device)
    
    
for num_heads in heads_to_test:
    model = MultiHeadAttention(512,num_heads).to(device)
    
    
    # Step 2: Load the state dictionary from the .pth file
    weights_path = f'experiments/gpu_trained_{num_heads}head.pth'
    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))

    # Step 3: Load the weights into your instantiated model class
    model.load_state_dict(state_dict)

    
    
    model.eval()
    
    
    _,weights = model(X,X,X)
    
    
    newest_token_weights = weights[-1,:,:,:]
    
    matrix_to_plot = newest_token_weights.detach().cpu().numpy()
    
    
    for i in range(num_heads):
        
        head_matrix = matrix_to_plot[i,:,:]
        
        plt.figure(figsize=(14, 8))

        # 2. Configure the heatmap
        sns.heatmap(head_matrix, 
                    cmap='viridis', 
                    # 3. Use an accurate label for attention weights (0.0 to 1.0)
                    cbar_kws={'label': f'GPU Attention Weight (Probability)- head {i+1} of {num_heads} heads'})

        # 4. Apply RIGOROUS, accurate professional labels
        plt.title("GPU Attention Alignment Heatmap (Multiple Heads)", fontsize=16, pad=20)

        # The generated tokens (rows) are Queries asking the source tokens (columns)
        plt.ylabel("Query Tokens (Looking for Context)", fontsize=12, labelpad=10) 
        plt.xlabel("Key Tokens (Context Provided)", fontsize=12, labelpad=20)

        # Save the final figure
        filepath = Path(f'./experiments/gpu_attention_weights head {i+1} of {num_heads} heads.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')

        plt.show()
        print(f"Heatmap successfully saved to: {filepath}")

    
    
   
   