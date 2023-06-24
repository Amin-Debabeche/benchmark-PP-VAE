import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE


plt.rc('text')
plt.rc('font', family='sans-serif')

plt.rcParams['figure.figsize'] = [7, 7]
fig, ax = plt.subplots(1)

embeddings_list = []
properties_list = []

# task = 'opd'
task = 'zinc'
# task = 'tki'
standardize = False

ground_truth = False

print('Embeddings loading...')

for i in range(0, 1124):
    array = np.load(os.path.join('{}_embeddings'.format(task),
                    'embeddings_{}.npy'.format(i)), allow_pickle=True)
    ls = array.tolist()
    embeddings_list += ls
    print(i)

for i in range(0, 1124):
    if ground_truth:
        array = np.load(os.path.join('{}_ground_truth_properties'.format(
            task), 'properties_{}.npy'.format(i)), allow_pickle=True)
    else:
        array = np.load(os.path.join('{}_properties'.format(
            task), 'properties_{}.npy'.format(i)), allow_pickle=True)
    ls = array.tolist()
    properties_list += ls
    print(i)

properties_list = np.array(properties_list)
if standardize:
    properties_list = (properties_list - properties_list.mean()
                       ) / properties_list.std()


pca = PCA(n_components=2)

embeddings = pca.fit_transform(embeddings_list)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)

print(sum(pca.explained_variance_ratio_))

with open('{}_PCA_embeddings.npy'.format(task), 'wb') as f:
    np.save(f, embeddings)

print("Saved PCA embeddings")


plot = plt.scatter(embeddings[:, 0], embeddings[:, 1],
                   c=properties_list, cmap='viridis', s=3)
cbar = plt.colorbar(plot)
for t in cbar.ax.get_yticklabels():
    t.set_fontsize(25)

if ground_truth:
    if task == 'zinc':
        cbar.ax.set_ylabel('Ground truth logP', fontsize=25)
    elif task == 'tki':
        cbar.ax.set_ylabel('Ground truth Erlotinib similarity', fontsize=25)
    elif task == 'opd':
        cbar.ax.set_ylabel('Ground truth $E_{rep}$'.replace(
            'rep', 'GAP'), fontsize=25)
else:
    if task == 'zinc':
        cbar.ax.set_ylabel('MLP predicted logP', fontsize=25)
    elif task == 'tki':
        cbar.ax.set_ylabel('MLP predicted Erlotinib similarity', fontsize=25)
    elif task == 'opd':
        cbar.ax.set_ylabel(
            'MLP predicted $E_{rep}$ / eV'.replace('rep', 'GAP'), fontsize=25)

ax.spines['bottom'].set_linewidth('3')
ax.spines['top'].set_linewidth('3')
ax.spines['left'].set_linewidth('3')
ax.spines['right'].set_linewidth('3')

ax.tick_params(axis='y', length=6, width=3,
               labelsize=25, pad=10, direction='in')
ax.tick_params(axis='x', length=6, width=3,
               labelsize=25, pad=10, direction='in')

ax.set_xlabel('Principal Component 1', fontsize=30)
ax.set_ylabel('Principal Component 2', fontsize=30)

plt.tight_layout()

if ground_truth:
    plt.savefig('Ground_truth_cvae_{}_train_latent.png'.format(task), dpi=400)
    plt.savefig('Ground_truth_cvae_{}_train_latent.pdf'.format(task))
    plt.savefig('Ground_truth_cvae_{}_train_latent.svg'.format(task))
else:
    plt.savefig('cvae_{}_train_latent.png'.format(task), dpi=400)
    plt.savefig('cvae_{}_train_latent.pdf'.format(task))
    plt.savefig('cvae_{}_train_latent.svg'.format(task))

plt.close()
