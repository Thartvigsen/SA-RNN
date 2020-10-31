import torch
from torch import nn
import numpy as np

def hardSigma(a, x):
    x = torch.div(torch.add(torch.mul(x, a), 1), 2.0)
    x = torch.clamp(x, min=0, max=1)
    return x

class Binarize(torch.autograd.Function):
    """Custom rounding of PyTorch tensors
    Rounding with straight-through gradient estimation. 
    """
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class SARNN(nn.Module):
    """
    Selective-Activation RNN (SA-RNN), as proposed in the paper "Learning to
    Selectively Update State Neurons in Recurrent Networks", published at
    CIKM, 2020.

    The key idea is that at every step in a sequence, we compute a new
    hidden state with an RNN, but use a Controller network that predicts
    which dimensions of the hidden state to update. Therefore, some
    dimensions persist over long periods of time with no updates. This
    allows for an adaptive approach to sequential representation learning
    where the model can alter the complexity of its representations
    according to the task discretely.

    Usage of the forward method
    ---------------------------

    Parameters
    ----------
    X : torch.tensor of shape (timesteps x instances x variables)
        This is the time series input. As implemented, this method requires
        equal-length time series.
    epoch : boolean
        Integer value indicating the current epoch. Used for slope
        annealing.  

    Returns
    -------
    logits : torch.tensor of shape (instances x number of output dimenisions)
        This matrix contains the predictions made by the network for each
        input instance.
    update_decisions: torch.tensor of shape (instance x timesteps x number of hidden dimensions)
        This tensor records the update decisions made for each timestep of
        the RNN for each instance. 1 indicates "UPDATE", 0 indicates "SKIP".
    """
    def __init__(self, ninp, nhid, nout, nepoch, lam=1e-03, anneal=True):
        super(SARNN, self).__init__()

        # --- model parameters ---
        self.ninp = ninp
        self.nhid = nhid
        self.nout = nout
        self.nepoch = nepoch

        # --- hyperparameters ---
        self.LAMBDA = lam
        self.anneal = anneal
        combined_dim = ninp + nhid

        # --- Recurrence Function ---
        self.r = nn.Linear(combined_dim, self.nhid)
        self.z = nn.Linear(combined_dim, self.nhid)
        self.h = nn.Linear(combined_dim, self.nhid)

        # --- coordinator mappings ---
        self.fc1 = torch.nn.Linear(combined_dim, self.nhid)
        new_weights = torch.cat((self.fc1.weight[:, :self.nhid]*torch.eye(self.nhid),
                                 self.fc1.weight[:, -self.ninp:]), 1)
        self.fc1.weight = torch.nn.Parameter(new_weights)

        # --- output mappings ---
        self.out = nn.Linear(self.nhid, self.nout)

        # --- Slope Annealing (setting from HM-RNN) ---
        if anneal:
            slopes = []
            for i in range(self.nepoch):
                slopes.append(min(5, 1 + 0.04*i))
            self._slopes = torch.tensor(slopes)

    def GRUCell(self, x, hidden):
        combined = torch.cat((x, hidden), dim=1)
        z = torch.sigmoid(self.z(combined))
        r = torch.sigmoid(self.r(combined))
        h = torch.tanh(self.h(torch.cat((x, torch.mul(hidden, r)), dim=1)))
        hidden = torch.mul((1-z), h) + torch.mul(z, hidden)
        return hidden

    def modifyGradients(self):
        new_grads = torch.cat((self.fc1.weight.grad[:, :self.nhid]*torch.eye(self.nhid), self.fc1.weight.grad[:, -self.ninp:]), 1)
        self.fc1.weight.grad = torch.nn.Parameter(new_grads)

    def coordinate(self, state_prev, state_new, x_t, u_prev, epoch):
        x = torch.cat((state_prev, x_t), dim=1)

        if self.anneal:
            alpha = self._slopes[epoch]
        else:
            alpha = 0.1
        update_probability = hardSigma(alpha, self.fc1(x))

        # --- Decide which neurons to update ---
        binarize = Binarize.apply
        update_decision = binarize(update_probability)

        h_new = update_decision*state_new
        h = (1-update_decision)*state_prev + h_new
        return h, update_probability, update_decision

    def forward(self, x, epoch):
        T, B, V = x.shape # Assume input is of shape timesteps x batch x variables
        hidden = torch.zeros((B, self.nhid))
        update_decisions = []
        update_probs = torch.zeros((1, B, self.nhid))
        for i in range(T):
            x_t = x[i]
            h_tilde = self.GRUCell(x_t, hidden)
            hidden, update_probs, update_decision = self.coordinate(hidden, h_tilde, x_t, update_probs, epoch)
            update_decisions.append(update_decision)

        self.update_decisions = torch.stack(update_decisions).transpose(0, 1) # B x T x nhid
        logits = self.out(hidden)
        return logits, self.update_decisions.detach().numpy()

    def computeLoss(self, loss_fn, logits, labels):
        # loss_fn may be torch.nn.functional.cross_entropy for Classification
        task_loss = loss_fn(logits, labels)
        budget_loss = self.update_decisions.sum(1).mean() # Sum 
        return task_loss + self.LAMBDA*budget_loss
