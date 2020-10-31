import torch
from model import SARNN
import torch.nn.functional as F
import matplotlib.pyplot as plt

# --- create some parameters for a synthetic dataset ---
ninp = 3  # Number of variables in the data
nclasses = 2 # Number of classes to predict (set 1 for regression) -- binary classification for this example
nhid = 50 # Hidden dimension of the RNN
batch_size = 8 # mini-batch size
ntimesteps = 10
nepoch = 50 # How many epochs to train for

# --- defining data and model ---
x = torch.rand((ntimesteps, batch_size, ninp)) # A simple synthetic time series with 5 timesteps and 3 variables
labels = torch.randint(nclasses, (batch_size,))
M = SARNN(ninp, nhid, nclasses, nepoch, lam=1e-03, anneal=True) # Initializing the model

# --- inference ---
# Now we can use m for inference
logits, update_decisions = M(x, epoch=0) # During training, use the current epoch as input
_, predictions = torch.max(torch.softmax(logits, 1), 1)

# --- computing loss and gradients ---
# Computing the loss is quite simple:
loss = M.computeLoss(F.cross_entropy, logits, labels)
loss.backward() # Compute all gradients

# Visual inspection of update decisions
i = 0 # Choose which element of the batch to visualize

# Generate the sort of plot shown in the paper Figure 3
fig, ax = plt.subplots(figsize=(16, 8))
ax.imshow(update_decisions[i].T, cmap="Greys")
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xlabel("Time")
ax.set_ylabel("Hidden Dimension")
plt.show()
