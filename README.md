# DEEP-LEARNING-WORKSHOP-1
## Binary Classification with Neural Networks on the Census Income Dataset
## NAME: GANESH D
## REGISTER NUMBER:212223240035
# PROGRAM:

```
import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
%matplotlib inline

df = pd.read_csv('income.csv')
```


```
print(len(df))
df.head()
```
<img width="1114" height="296" alt="image" src="https://github.com/user-attachments/assets/778bcbde-01b9-49b5-bfb8-9ea179beb709" />



```
df['label'].value_counts()
```

<img width="782" height="122" alt="image" src="https://github.com/user-attachments/assets/a1a57f6e-318e-4090-8526-995c1e684465" />

```
df.columns
```
<img width="863" height="84" alt="image" src="https://github.com/user-attachments/assets/4a2008a0-663a-410c-965b-2f34956bd206" />

```
cat_cols = ['sex', 'education', 'marital-status', 'workclass', 'occupation']
cont_cols = ['age', 'hours-per-week']
y_col = ['label']
print(f'cat_cols  has {len(cat_cols)} columns')
print(f'cont_cols has {len(cont_cols)} columns')
print(f'y_col     has {len(y_col)} column')
```
<img width="794" height="93" alt="image" src="https://github.com/user-attachments/assets/61e5b46d-4a76-4c33-b7a0-3c9f9bb481c7" />

```
for col in cat_cols:
    df[col] = df[col].astype('category')
cat_szs = [len(df[col].cat.categories) for col in cat_cols]
emb_szs = [(size, min(50, (size+1)//2)) for size in cat_szs]
print(emb_szs)
```
<img width="829" height="43" alt="image" src="https://github.com/user-attachments/assets/a3ce5f16-5112-4f69-b919-80f0f5a4fb73" />

```
cats = np.stack([df[col].cat.codes.values for col in cat_cols], axis=1)
cats[:5]
cats = torch.tensor(cats, dtype=torch.int64)
cats
```
<img width="768" height="187" alt="image" src="https://github.com/user-attachments/assets/a468041c-de4e-42aa-a1bb-f9d56e27438f" />

```
conts = np.stack([df[col].values for col in cont_cols], axis=1)
conts[:5]
conts = torch.tensor(conts, dtype=torch.float32)
conts
```
<img width="816" height="166" alt="image" src="https://github.com/user-attachments/assets/3efa4717-1874-4927-aea9-cfe9dde85a46" />

```
y = torch.tensor(df[y_col].values, dtype=torch.int64).flatten()
b = 30000  # total records
t = 5000   # test size

cat_train = cats[:b-t]
con_train = conts[:b-t]
y_train = y[:b-t]

cat_test = cats[b-t:]
con_test = conts[b-t:]
y_test = y[b-t:]

torch.manual_seed(33)

```
<img width="644" height="42" alt="image" src="https://github.com/user-attachments/assets/39dca12c-05b0-443f-9d21-19d220c9c6f9" />

```
class TabularModel(nn.Module):

    def __init__(self, emb_szs, n_cont, out_sz, layers, p=0.5):
        # Call the parent __init__
        super().__init__()
        
        # Set up the embedding, dropout, and batch normalization layer attributes
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni,nf in emb_szs])
        self.emb_drop = nn.Dropout(p)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        
        # Assign a variable to hold a list of layers
        layerlist = []
        
        # Assign a variable to store the number of embedding and continuous layers
        n_emb = sum((nf for ni,nf in emb_szs))
        n_in = n_emb + n_cont
        
        # Iterate through the passed-in "layers" parameter (ie, [200,100]) to build a list of layers
        for i in layers:
            layerlist.append(nn.Linear(n_in,i)) 
            layerlist.append(nn.ReLU(inplace=True))
            layerlist.append(nn.BatchNorm1d(i))
            layerlist.append(nn.Dropout(p))
            n_in = i
        layerlist.append(nn.Linear(layers[-1],out_sz))
        
        # Convert the list of layers into an attribute
        self.layers = nn.Sequential(*layerlist)
    
    def forward(self, x_cat, x_cont):
        # Extract embedding values from the incoming categorical data
        embeddings = []
        for i,e in enumerate(self.embeds):
            embeddings.append(e(x_cat[:,i]))
        x = torch.cat(embeddings, 1)
        # Perform an initial dropout on the embeddings
        x = self.emb_drop(x)
        
        # Normalize the incoming continuous data
        x_cont = self.bn_cont(x_cont)
        x = torch.cat([x, x_cont], 1)
        
        # Set up model layers
        x = self.layers(x)
        return x
```

```
model = TabularModel(emb_szs, n_cont=len(cont_cols), out_sz=2, layers=[50], p=0.4)
model 
```
<img width="1112" height="426" alt="image" src="https://github.com/user-attachments/assets/e8646fb4-f675-4713-b2a2-34336d747ab9" />

```
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

```

```
import time
start_time = time.time()

epochs = 300
losses = []

for i in range(epochs):
    i+=1
    y_pred = model(cat_train, con_train)
    loss = criterion(y_pred, y_train)
    losses.append(loss)
    
    # a neat trick to save screen space:
    if i%25 == 1:
        print(f'epoch: {i:3}  loss: {loss.item():10.8f}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'epoch: {i:3}  loss: {loss.item():10.8f}') # print the last line
print(f'\nDuration: {time.time() - start_time:.0f} seconds') # print the time elapsed

```
<img width="748" height="312" alt="image" src="https://github.com/user-attachments/assets/bec63bad-86da-462e-965e-3c90da010984" />

```
plt.plot([loss.item() for loss in losses])
plt.xlabel("Epoch")
plt.ylabel("Cross Entropy Loss")
plt.title("Training Loss")
plt.show()

```
<img width="1066" height="621" alt="image" src="https://github.com/user-attachments/assets/bbfa9f79-030d-439d-9601-595f1ed37bd1" />

```
with torch.no_grad():
    y_val = model(cat_test, con_test)
    loss = criterion(y_val, y_test)
print(f'CE Loss: {loss:.8f}')

```

<img width="599" height="53" alt="image" src="https://github.com/user-attachments/assets/84dade80-5042-467a-95f7-accb879daf02" />

```
correct = 0
for i in range(len(y_test)):
    if y_val[i].argmax().item() == y_test[i].item():
        correct += 1

accuracy = correct / len(y_test) * 100
print(f'{correct} out of {len(y_test)} = {accuracy:.2f}% correct')
```

<img width="492" height="42" alt="image" src="https://github.com/user-attachments/assets/524f274f-d5f9-4c1f-bc72-d3a5a7e2c914" />
