import torch 
import torch.nn.functional as F 
import matplotlib.pyplot as plt 
words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s, i in stoi.items()}
print(itos)

#build the dataset 
block_size = 3 # context length: how many characters do we take to predict the next one? 
X, Y = [], []
for w in words[:5]:
    print(w)
    context = [0] * block_size
    for ch in w + '.':
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        print(''.join(itos[i] for i in context), '--->', itos[ix])
        context = context[1:] + [ix] # crop and append

X = torch.tensor(X)
Y = torch.tensor(Y)

# embedding look up table 
'''
C = torch.randn((27, 2))
F.one_hot(torch.tensor(5), num_classes=27).float() @ C 
emb = C[X]
W1 = torch.randn((6, 100))
b1 = torch.randn(100)
torch.cat([emb[:, 0, : ], emb[:, 1, : ], emb[:, 2, : ]], 1).shape
torch.cat(torch.unbind(emb, 1), 1).shape
a = torch.arange(18)
a.view(2, 9)
'''

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27, 2), generator=g)
W1 = torch.randn((6, 300), generator=g)
b1 = torch.randn(300, generator=g)
W2 = torch.randn((300, 27), generator=g)
b2 = torch.randn(27, generator=g)
paramaters = [C, W1, b1, W2, b2]

for p in paramaters:
    p.requires_grad = True

lre = torch.linspace(-3, 0, 1000)
lrs = 10**lre
lossi = []
lri = []

for i in range(30000):
    #minibatch construct 
    ix = torch.randint(0, X.shape[0], (32,))

    #forward pass 
    emb = C[X[ix]]
    h = torch.tanh(emb.view(-1, 6) @ W1 + b1) 
    logits = h @ W2 + b2 
    #counts = logits.exp()
    #prob = counts / counts.sum(1, keepdims=True)
    #loss = -prob[torch.arange(32), Y].log().mean()
    loss = F.cross_entropy(logits, Y[ix])
    

    # backward pass 
    for p in paramaters:
        p.grad = None
    loss.backward()

    #update 
    #lr = lrs[i]
    lr = 0.1
    for p in paramaters:
        p.data += -lr * p.grad 

    # track stats 
    #lri.append(lr)
    #lossi.append(loss.item())

print(loss.item())

#plt.plot(lri, lossi)
#plt.show()

#torch.randint(0, 5, (32,))

# sample from the model
g = torch.Generator().manual_seed(2147483647 + 10)

with torch.no_grad():                          # no gradients needed for sampling
    for _ in range(20):
        out = []
        context = [0] * block_size             # start with '...'
        while True:
            emb = C[torch.tensor([context])]   # (1, block_size, 2)
            h = torch.tanh(emb.view(1, -1) @ W1 + b1)
            logits = h @ W2 + b2
            probs = F.softmax(logits, dim=1)   # turn logits into probabilities
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            context = context[1:] + [ix]       # slide the window
            out.append(ix)
            if ix == 0:                        # '.' means end of name
                break
        print(''.join(itos[i] for i in out))