import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

class OPDMolecularVAE(nn.Module):
    def __init__(self):
        super(OPDMolecularVAE, self).__init__()

        # The input filter dim should be 35
        #  corresponds to the size of CHARSET
        # self.conv1d1 = nn.Conv1d(120, 9, kernel_size=9)
        self.conv1d1 = nn.Conv1d(50, 9, kernel_size=9)  
        self.conv1d2 = nn.Conv1d(9, 9, kernel_size=9)
        self.conv1d3 = nn.Conv1d(9, 10, kernel_size=11)
        # self.conv1d4 = nn.Conv1d(10, 10, kernel_size=11)
        # self.conv1d5 = nn.Conv1d(10, 10, kernel_size=11)
 
        self.fc0 = nn.Linear(1190, 435)
        # self.fc0 = nn.Linear(90, 435)
        self.fc11 = nn.Linear(435, 292)
        self.fc12 = nn.Linear(435, 292)

        self.fc2 = nn.Linear(292, 292)
        self.gru = nn.GRU(292, 501, 3, batch_first=True)
        self.fc3 = nn.Linear(501, 50)

        self.prop_1 = nn.Linear(292, 64)
        self.prop_2 = nn.Linear(64, 32)
        self.prop_3 = nn.Linear(32, 1)

    def encode(self, x):
        h = F.relu(self.conv1d1(x))
        h = F.relu(self.conv1d2(h))
        h = F.relu(self.conv1d3(h))
        # h = F.relu(self.conv1d4(h))
        # h = F.relu(self.conv1d5(h))
        h = h.view(h.size(0), -1)
        h = F.selu(self.fc0(h))
        return self.fc11(h), self.fc12(h)

    def reparametrize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = 1e-2 * torch.randn_like(std)
            w = eps.mul(std).add_(mu)
            return w
        else:
            return mu

    def predict_prop(self, z):
        p = F.relu(self.prop_1(z))
        p = F.relu(self.prop_2(p))
        return self.prop_3(p)
    
    def decode(self, z):
        z = F.selu(self.fc2(z))
        z = z.view(z.size(0), 1, z.size(-1)).repeat(1, 145, 1)
        out, h = self.gru(z)
        out_reshape = out.contiguous().view(-1, out.size(-1))
        y0 = F.softmax(self.fc3(out_reshape), dim=1)
        y = y0.contiguous().view(out.size(0), -1, y0.size(-1))
        return y

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        pred_y = self.predict_prop(z).reshape(-1)
        return self.decode(z), mu, logvar, pred_y


class MolecularVAE(nn.Module):
    def __init__(self):
        super(MolecularVAE, self).__init__()

        # The input filter dim should be 35
        #  corresponds to the size of CHARSET
        # self.conv1d1 = nn.Conv1d(120, 9, kernel_size=9)
        self.conv1d1 = nn.Conv1d(35, 9, kernel_size=9)  
        self.conv1d2 = nn.Conv1d(9, 9, kernel_size=9)
        self.conv1d3 = nn.Conv1d(9, 10, kernel_size=11)
        # self.conv1d4 = nn.Conv1d(10, 10, kernel_size=11)
        # self.conv1d5 = nn.Conv1d(10, 10, kernel_size=11)
 
        self.fc0 = nn.Linear(940, 435)
        # self.fc0 = nn.Linear(90, 435)
        self.fc11 = nn.Linear(435, 196)
        self.fc12 = nn.Linear(435, 196)

        self.fc2 = nn.Linear(196, 196)
        self.gru = nn.GRU(196, 501, 3, batch_first=True)
        self.fc3 = nn.Linear(501, 35)

        # self.prop_1 = nn.Linear(196, 64)
        # self.prop_2 = nn.Linear(64, 32)
        # self.prop_3 = nn.Linear(32, 1)
        self.prop_1 = nn.Linear(196, 1)

    def encode(self, x):
        h = F.relu(self.conv1d1(x))
        h = F.relu(self.conv1d2(h))
        h = F.relu(self.conv1d3(h))
        # h = F.relu(self.conv1d4(h))
        # h = F.relu(self.conv1d5(h))
        h = h.view(h.size(0), -1)
        h = F.selu(self.fc0(h))
        return self.fc11(h), self.fc12(h)

    def reparametrize(self, mu, logvar):
        import pdb; pdb.set_trace()
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = 1e-2 * torch.randn_like(std)
            w = eps.mul(std).add_(mu)
            return w
        else:
            return mu

    def predict_prop(self, z):
        # p = F.relu(self.prop_1(z))
        # p = F.relu(self.prop_2(p))
        # return self.prop_3(p)
        return self.prop_1(z)
    
    def decode(self, z):
        z = F.selu(self.fc2(z))
        z = z.view(z.size(0), 1, z.size(-1)).repeat(1, 120, 1)
        out, h = self.gru(z)
        out_reshape = out.contiguous().view(-1, out.size(-1))
        y0 = F.softmax(self.fc3(out_reshape), dim=1)
        y = y0.contiguous().view(out.size(0), -1, y0.size(-1))
        return y

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        pred_y = self.predict_prop(z).reshape(-1)
        return self.decode(z), mu, logvar, pred_y
