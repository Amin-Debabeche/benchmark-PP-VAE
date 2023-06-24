import argparse
import random
import os
import math
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
import torch.nn.functional as F

from models import MolecularVAE
from torch.utils.tensorboard import SummaryWriter
from featurizer import OneHotFeaturizer

import wandb
wandb.init(project="vae_dummy")

# training property loss stagnates at 0.6 if kl div is set to 1 at the start
# So try step-wise schedule with warmup time

# batch_size = 100
# epochs = 100
os.environ["CUDA_VISIBLE_DEVICES"]="0,2"
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def loss_function(recon_x, x, mu, logvar, pred_y, y):
    CE = torch.mean(-torch.sum(x * torch.log(recon_x + 1e-7), axis=-1))
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())  # sum
    prop_loss = F.mse_loss(pred_y, y)  # mean
    return CE, KLD, prop_loss


def train(epoch, num_iters):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        inputs, labels = data
        inputs = inputs.transpose(1, 2).to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar, pred_y_batch = model(inputs)

        bce, kld, prop_loss = loss_function(
            recon_batch, inputs.transpose(1, 2), mu, logvar, pred_y_batch, labels)

        loss = (recon_weight * bce + beta * kld + prop_weight * prop_loss)

        loss.backward()

        # Calculate and print the norm of gradients for encoder
        encoder_norm = 0
        for name, param in model.named_parameters():
            if param.grad is not None and 'conv_1d' in name:
                param_norm = param.grad.data.norm(2)
                encoder_norm += param_norm.item() ** 2
        encoder_norm = encoder_norm ** 0.5
        # print(f"Iteration: {num_iters}\tEncoder Gradient norm: {encoder_norm:.4f}")

        # Calculate and print the norm of gradients for decoder
        decoder_norm = 0
        for name, param in model.named_parameters():
            if param.grad is not None and ('gru' in name):
                param_norm = param.grad.data.norm(2)
                decoder_norm += param_norm.item() ** 2
        decoder_norm = decoder_norm ** 0.5
        # print(f"Iteration: {num_iters}\tDecoder Gradient norm: {decoder_norm:.4f}")

        optimizer.step()
        num_iters += 1
        train_loss += loss

        wandb.log({"Train Loss": loss}) #Logging w&b
        wandb.log({"Train Prop Loss": (prop_loss) ** 0.5}) #Logging w&b
        
        writer.add_scalar('train loss', loss, num_iters)
        writer.add_scalar('train bce', bce, num_iters)
        writer.add_scalar('train kld', kld, num_iters)
        writer.add_scalar('train prop loss', (prop_loss) ** 0.5, num_iters)
        writer.add_scalar('train beta', beta, num_iters)
        writer.add_scalar('encoder grad norm', encoder_norm, num_iters)
        writer.add_scalar('decoder grad norm', decoder_norm, num_iters)

        if num_iters % 100 == 0:
            print(f'train: {epoch} / {batch_idx}\t{(loss):.4f}')
            val_loss = val(epoch, num_iters)

            oh = OneHotFeaturizer()

            cpu_x = inputs.transpose(1, 2)[0].cpu().detach().numpy()
            recon_x = recon_batch[0].cpu().detach().numpy()
            smiles_x = oh.decode_smiles_from_index(np.argmax(cpu_x, axis=1))
            smiles_recon = oh.decode_smiles_from_index(
                np.argmax(recon_x, axis=1))
            print('train smiles input: ', smiles_x)
            print('train smiles recon: ', smiles_recon)

    return train_loss / len(train_loader), num_iters


def val(epoch, num_iters):
    # model.eval()
    with torch.no_grad():
        val_loss = 0
        val_bce = 0
        val_kld = 0
        val_prop_loss = 0
        chosen_idx = random.randint(0, len(val_loader))
        for batch_idx, data in enumerate(val_loader):
            inputs, labels = data
            inputs = inputs.transpose(1, 2).to(device)
            labels = labels.to(device)
            recon_batch, mu, logvar, pred_y_batch = model(inputs)

            if batch_idx == chosen_idx:
                oh = OneHotFeaturizer()
                cpu_x = inputs.transpose(1, 2)[0].cpu().detach().numpy()
                recon_x = recon_batch[0].cpu().detach().numpy()
                smiles_x = oh.decode_smiles_from_index(
                    np.argmax(cpu_x, axis=1))
                smiles_recon = oh.decode_smiles_from_index(
                    np.argmax(recon_x, axis=1))
                print('val smiles input: ', smiles_x)
                print('val smiles recon: ', smiles_recon)

            bce, kld, prop_loss = loss_function(
                recon_batch, inputs.transpose(1, 2), mu, logvar, pred_y_batch, labels)

            val_bce += bce
            val_kld += kld

            val_prop_loss += prop_loss
            val_loss += (recon_weight * bce + beta *
                         kld + prop_weight * prop_loss)

        writer.add_scalar('val bce', val_bce / len(val_loader), num_iters)
        writer.add_scalar('val kld', val_kld / len(val_loader), num_iters)
        writer.add_scalar('val prop loss', (val_prop_loss /
                          len(val_loader)) ** 0.5, num_iters)
        writer.add_scalar('val loss', (val_loss) / len(val_loader), num_iters)
        writer.add_scalar('val beta', beta, num_iters)
        
        wandb.log({"Valid Loss": val_loss / len(val_loader)}) # Logging w&b
        wandb.log({"Valid Prop Loss": (val_prop_loss /
                          len(val_loader)) ** 0.5}) #Logging w&b
        
        print(f'val: {epoch} \t{(val_loss / len(val_loader)):.4f}')


if __name__ == '__main__':
    task = 'zinc'
    prop = 'logP'

    batch_size = 100
    epochs = 5000

    recon_weight = 0.5
    beta = 0.5
    prop_weight = 0.5
    lr = 4.56e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    X_train = np.load(
        f'./data/{task}_smiles_train.npz')['arr'].astype(np.float32)
    y_train = np.load(
        f'./data/{task}_{prop}_train.npz')['arr'].astype(np.float32)
    y_train = (y_train - y_train.mean()) / y_train.std()

    X_val = np.load(f'./data/{task}_smiles_val.npz'
                    )['arr'].astype(np.float32)
    y_val = np.load(f'./data/{task}_{prop}_val.npz')['arr'].astype(np.float32)
    y_val = (y_val - y_val.mean()) / y_val.std()

    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size)

    torch.manual_seed(42)

    model = MolecularVAE()

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    TB_LOG_PATH = '{}_runs_kl_{}_lr_{}_prop_{}_recon_{}_smaller_latent_normalized_y_linear_MLP_patent_bidirectional'.format(
        task, beta, lr, prop_weight, recon_weight)
    weights_path = '{}_weights_kl_{}_lr_{}_prop_{}_recon_{}_smaller_latent_normalized_y_linear_MLP_patent_bidirectional'.format(
        task, beta, lr, prop_weight, recon_weight)

    if not os.path.isdir(TB_LOG_PATH):
        os.makedirs(TB_LOG_PATH)
    if not os.path.isdir(weights_path):
        os.makedirs(weights_path)

    writer = SummaryWriter(TB_LOG_PATH)
    set_all_seeds(9999)
    num_iters = 0
    for epoch in range(1, epochs + 1):
        train_loss, num_iters = train(epoch, num_iters)
        if epoch % 1 == 0:
            torch.save(model.state_dict(),
                       './{}_weights_kl_{}_lr_{}_prop_{}_recon_{}_smaller_latent_normalized_y_linear_MLP_patent_bidirectional/vae-{:03d}-{}.pth'.format(task, beta, lr, prop_weight, recon_weight, epoch, train_loss))
            
    print('===== Finished =====')
