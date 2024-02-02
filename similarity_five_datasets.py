import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def l2_normalize(x, dim=None, epsilon=1e-12):
    square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
    x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
    return x * x_inv_norm
        
num_classes = 50
embeddings = torch.zeros((num_classes, 768))
similarity = np.zeros((num_classes, num_classes))

for c in range(num_classes):
    embeddings[c] = torch.load(f"./init_prompts/5-datasets/class/prompt_{c}.pt")
    embeddings[c] = l2_normalize(embeddings[c], dim=0)

for i in range(num_classes):
    for j in range(num_classes):
        sim = torch.dot(embeddings[i], embeddings[j]).detach().numpy()
        similarity[i, j] = sim

plt.figure(figsize=(10, 10))
hm = sns.heatmap(data=similarity, cmap="coolwarm", annot=False, square=True, vmin = 0, vmax = 1)
plt.xlabel("Classes")
plt.ylabel("Classes")
plt.yticks(hm.get_yticks(), size = 8)
plt.xticks(hm.get_xticks(), size = 8)
# cbar = hm.collections[0].colorbar
# cbar.ax.tick_params(labelsize=80)
plt.savefig(f'plots/similarity/five_datasets_classes_similarity.png')
plt.clf()
# plt.show()


num_tasks = 5
embeddings = torch.zeros((num_tasks, 768))
similarity = np.zeros((num_tasks, num_tasks))

for t in range(num_tasks):
    embeddings[t] = torch.load(f"./init_prompts/5-datasets/task/prompt_{t}.pt")
    embeddings[t] = l2_normalize(embeddings[t], dim=0)

for i in range(num_tasks):
    for j in range(num_tasks):
        sim = torch.dot(embeddings[i], embeddings[j]).detach().numpy()
        similarity[i, j] = sim

plt.figure(figsize=(10, 10))
hm = sns.heatmap(data=similarity, cmap="coolwarm", annot=True, square=True, vmin = 0, vmax = 1)
plt.xlabel("Tasks")
plt.ylabel("Tasks")
plt.yticks(hm.get_yticks(), size = 10)
plt.xticks(hm.get_xticks(), size = 10)
# cbar = hm.collections[0].colorbar
# cbar.ax.tick_params(labelsize=80)
plt.savefig(f'plots/similarity/five_datasets_tasks_similarity.png')
plt.clf()