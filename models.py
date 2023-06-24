import random, math
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'


class MolecularVAE(nn.Module):
    def __init__(self):
        super(MolecularVAE, self).__init__()

        # The input filter dim should be 35
        #  corresponds to the size of CHARSET
        self.conv1d1 = nn.Conv1d(36, 9, kernel_size=9) # I added a char which changed the input dimension
        self.conv1d2 = nn.Conv1d(9, 9, kernel_size=9)
        self.conv1d3 = nn.Conv1d(9, 10, kernel_size=11)

        self.bn1 = nn.BatchNorm1d(9, affine=False)
        self.bn2 = nn.BatchNorm1d(10, affine=False)
        self.bn3 = nn.BatchNorm1d(196, affine=False)
        self.bn4 = nn.BatchNorm1d(240, affine=False)
        self.bn5 = nn.BatchNorm1d(66, affine=False)
        self.bn6 = nn.BatchNorm1d(65, affine=False)

        self.dropout1 = nn.Dropout(p=0.0828)
        self.dropout2 = nn.Dropout(p=0.1569)

        self.fc0 = nn.Linear(940, 196)
        self.fc11 = nn.Linear(196, 196)
        self.fc12 = nn.Linear(196, 196)

        self.fc2 = nn.Linear(196, 196)

        self.fc3 = nn.Linear(196, 240)

        self.gru = nn.GRU(240, 488, 3, batch_first=True)

        # self.gru = nn.GRU(196, 488, 3, batch_first=True)
        self.fc4 = nn.Linear(488, 36)

        self.fc5 = nn.Linear(196, 67)
        self.fc6 = nn.Linear(67, 66)
        self.fc7 = nn.Linear(66, 65)
        self.fc8 = nn.Linear(65, 1)


        # self.prop_1 = nn.Linear(196, 1)

    def encode(self, x):
        # conv layers
        h = F.tanh(self.conv1d1(x))
        h = self.bn1(self.dropout1(h))
        h = F.tanh(self.conv1d2(h))
        h = self.bn1(self.dropout1(h))
        h = F.tanh(self.conv1d3(h))
        h = self.bn2(self.dropout1(h))

        # flatten
        h = h.view(h.size(0), -1)

        # mid layers
        h = self.fc0(h)
        h = self.bn3(h)

        z_mean = self.fc11(h)
        z_std = self.fc12(h)

        return z_mean, z_std

    def reparametrize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            # eps = 1e-2 * torch.randn_like(std)
            eps = torch.randn_like(std) * 0.5
            w = eps.mul(std).add_(mu)
            return w
        else:
            return mu

    def predict_prop(self, z):
        z = F.tanh(self.fc5(z))
        z = self.dropout2(z)
        z = F.tanh(self.fc6(z))
        z = self.bn5(self.dropout2(z))
        z = F.tanh(self.fc7(z))
        z = self.bn6(self.dropout2(z))
        z = self.fc8(z)
        return z

    def decode(self, z):
        z = F.tanh(self.fc2(z))
        z = self.bn3(self.dropout1(z))

        z = F.tanh(self.fc3(z))
        z = self.bn4(self.dropout1(z))

        z = z.view(z.size(0), 1, z.size(-1)).repeat(1, 120, 1)
        
        out, h = self.gru(z)

        out_reshape = out.contiguous().view(-1, out.size(-1))
        # y0 = F.softmax(self.ggvG)
        # out, h = self.gru(z)
        y0 = F.softmax(self.fc4(out_reshape), dim=1)
        y = y0.contiguous().view(out.size(0), -1, y0.size(-1))
        return y
        # return y

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        pred_y = self.predict_prop(mu).reshape(-1)
        return self.decode(z), mu, logvar, pred_y
