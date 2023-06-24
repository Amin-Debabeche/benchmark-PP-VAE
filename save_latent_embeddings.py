import random
import os
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
import torch.nn.functional as F
from models import MolecularVAE, OPDMolecularVAE, OPDAllMolecularVAE, TKIMolecularVAE
from dataset import ZINCDataset
from torch.utils.tensorboard import SummaryWriter
from featurizer import OneHotFeaturizer

# training property loss stagnates at 0.6 if kl div is set to 1 at the start
# So try step-wise schedule with warmup time
batch_size = 500
epochs = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    batch_size = 100
    epochs = 100
    # num_samples = 3000

    ground_truth = False
    num_samples = 500000

    beta = 4.235e-5
    # beta_initial = 0
    # warmup_iters = 1000
    # beta_final = 0

    prop_weight = 0.75
    lr = 0.0013
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # task = 'tki'
    # prop = 'fda'
    task = 'zinc'
    prop = 'logP'
    # task = 'opd'
    # prop = 'gap'

    X_train = np.load('./data/{}_smiles_train.npz'.format(task)
                      )['arr'].astype(np.float32)
    y_train = np.load('./data/{}_{}_train.npz'.format(task,
                      prop))['arr'].astype(np.float32)
    # y_train = (y_train - y_train.mean()) / y_train.std()

    X_val = np.load('./data/{}_smiles_val.npz'.format(task)
                    )['arr'].astype(np.float32)
    y_val = np.load('./data/{}_{}_val.npz'.format(task, prop)
                    )['arr'].astype(np.float32)
    # y_val = (y_val - y_val.mean()) / y_val.std()

    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size)

    torch.manual_seed(42)

    if task == 'opd':
        model = OPDMolecularVAE().to(device)
    elif task == 'zinc':
        model = MolecularVAE().to(device)
    elif task == 'tki':
        model = TKIMolecularVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    TB_LOG_PATH = 'runs_kl_{}_lr_{}_prop_{}'.format(beta, lr, prop_weight)
    weights_path = 'weights_kl_{}_lr_{}_prop_{}'.format(beta, lr, prop_weight)

    model.load_state_dict(torch.load(
        'tki_weights_kl_0.5_lr_0.000456_prop_0.5_recon_0.5_smaller_latent_normalized_y_linear_MLP_patent_bidirectional/vae-100-0.3924936354160309.pth'))
    model.eval()

    set_all_seeds(9999)
    for batch_idx, data in enumerate(train_loader):
        inputs, labels = data
        # inputs = inputs.to(device)
        inputs = inputs.transpose(1, 2).to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar, pred_y_batch = model(inputs)

        embeddings_batch = mu.cpu().detach().numpy()

        properties_batch_ground_truth = labels.cpu().detach().numpy()
        properties_batch_predicted = pred_y_batch.cpu().detach().numpy()

        with open(os.path.join('{}_embeddings'.format(task), 'embeddings_{}.npy'.format(batch_idx)), 'wb') as f:
            np.save(f, embeddings_batch)

        # if ground_truth:
        with open(os.path.join('{}_ground_truth_properties'.format(task), 'properties_{}.npy'.format(batch_idx)), 'wb') as f:
            np.save(f, properties_batch_ground_truth)
        # else:
        with open(os.path.join('{}_properties'.format(task), 'properties_{}.npy'.format(batch_idx)), 'wb') as f:
            np.save(f, properties_batch_predicted)

        if batch_idx == int(num_samples / batch_size):
            break
