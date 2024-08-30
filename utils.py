from tqdm.notebook import tqdm
import torch

def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

def train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0
    model.train() 
    
    for (x,y) in tqdm(iterator, desc='Training', leave=False): 
        x = x.to(device) 
        y = y.to(device) 
        
        optimizer.zero_grad() 
        y_pred= model(x) 
        
        loss = criterion(y_pred, y) 
        acc = calculate_accuracy(y_pred, y) 
        
        loss.backward() 
        optimizer.step() 
        
        epoch_loss += loss.item() 
        epoch_acc += acc.item() 
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for (x,y) in tqdm(iterator, desc='Evaluating', leave=False):
            x = x.to(device)
            y = y.to(device)
            
            y_pred = model(x)
            
            loss = criterion(y_pred, y)
            acc = calculate_accuracy(y_pred, y)
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs