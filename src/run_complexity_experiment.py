import torch
import torch.nn as nn
import time
import json


from src.multihead_attention import MultiHeadAttention


#1. Setup
batch_size = 32
seq_len = 20
vocab_size = 1000
d_model = 128
heads_to_test = [1,4,8]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#fake input sentences
X = torch.randn(batch_size, seq_len, d_model).to(device)

X_val = torch.randn(batch_size,seq_len,d_model).to(device)
#fake target labels to calculate the loss against
Y = torch.randn(batch_size, seq_len, d_model).to(device)

# Loss Function
criterion = nn.MSELoss()


history = []




# Experiment Loop
for num_heads in heads_to_test:
    model = MultiHeadAttention(d_model,num_heads).to(device)
    #Optimizer
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
   
    
    total_time = 0
    running_loss = 0
    correct = 0
    total_steps = 0
    average_training_time = 0
    tolerance = 0.1
    is_close = 0
    total_elements = 0
  
    
    
    
    # --- 1. RESET MEMORY HERE ---
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model.train()
    for i in range(10):
        #start timer safely
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()
        
        optimizer.zero_grad()
        
        # Forward Pass
        output,weights = model(X,X,X)
        
        print(f"This is the shape of decoder weights: {weights.shape}")

        
        loss = criterion(output,Y)
        
        
        #Backward pass
        loss.backward()
        
        
        optimizer.step()
        
        
        
        
        
        #stop timer safely
        if device.type == "cuda":
            torch.cuda.synchronize()
        end_time = time.time()
        
        total_time += (end_time - start_time)
        
        running_loss += loss.item()
        
        total_steps += 1
        is_close = torch.abs(output - Y) <= tolerance
        correct += is_close.sum().item()
        
        total_elements += Y.numel()
        
        
   
            
        
    #Printing stats per epoch
    train_loss = running_loss/total_steps
    
    average_training_time = total_time /total_steps
    
    training_accuracy =  100 * correct /total_elements
    
    peak_memory_mb = 0
    if device.type == "cuda":
        peak_memory_bytes = torch.cuda.max_memory_allocated(device)
        peak_memory_mb = peak_memory_bytes / (1024 * 1024)
    
    
    model.eval()
    
    val_loss = 0
    total_time = 0
    total_elements = 0
    is_close = 0
    total_steps = 0
    running_loss =0
    
  
    
    
    with torch.no_grad():
        for i in range(10):
            #start timer safely
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.time()
            
            
            # Forward Pass
            output,_ = model(X_val,X_val,X_val)
            
            loss = criterion(output,Y)
                              
            
            #stop timer safely
            if device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.time()
            
            total_time += (end_time - start_time)
            
            val_loss += loss.item()
            
            total_steps += 1
            is_close = torch.abs(output - Y) <= tolerance
            correct += is_close.sum().item()
            
            total_elements += Y.numel()
            
           
                
                
            
            
    val_loss = val_loss/total_steps
    
    average_validation_time = total_time /total_steps
    
    validation_accuracy =  100 * correct /total_elements

    
    metrics = {
        "Num Head": num_heads,
        "train_loss": train_loss,
        "training_acc": training_accuracy,
        "training_time":average_training_time,
        "val_loss":val_loss,
        "val_time": average_validation_time,
        "val_accuracy": validation_accuracy,
        "training_memory_mb":peak_memory_mb
    }
    
    
    history.append(metrics)
    
    torch.save(model.state_dict(),f"./experiments/trained_{num_heads}head.pth")
    
    
    
    
# with open(f"./../experiments/complexity_comparion.json","w") as f:
#     json.dump(history,f,indent=4)
    
    
    
            
            
        
    
    
    
    
    
        
        
    
    
    
    
    
    
    
        