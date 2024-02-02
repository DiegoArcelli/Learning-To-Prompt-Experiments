import torch
import sys
sys.path.append("./../")
import torch
from torchvision import transforms
# from avalanche.models.vit import create_model
import torchvision
from tqdm import tqdm
import argparse
from timm.models import create_model
import models

parser = argparse.ArgumentParser(prog='Compute key counts', description='Compute the key-class and key-task counts')
parser.add_argument('--dataset', default="train", type=str,  choices=["train", "test"])
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = 100
num_tasks = 10
num_keys = 50
key_class_counts = {k: {c: 0 for c in range(num_classes)} for k in range(num_keys)}
key_task_counts = {k: {t: 0 for t in range(num_tasks)} for k in range(num_keys)}

original_model = create_model(
    "vit_base_patch16_224",
    pretrained=True,
    num_classes=100,
    drop_rate=0.0,
    drop_path_rate=0.0,
    drop_block_rate=None,
)

original_model = original_model.to(device)


model = create_model(
        "vit_base_patch16_224",
        pretrained=True,
        num_classes=100,
        drop_rate=0.0,
        drop_path_rate=0.0,
        drop_block_rate=None,
        prompt_length=5,
        embedding_key="cls",
        prompt_init="uniform",
        prompt_pool=True,
        prompt_key=True,
        pool_size=num_keys,
        top_k=5,
        batchwise_prompt=True,
        prompt_key_init="uniform",
        head_type="prompt",
        use_prompt_mask=True,
    )
# model = torch.load("l2p.pt")
model = model.to(device)
checkpoint = torch.load("./output_experiment/checkpoint/task10_checkpoint.pth")
model.load_state_dict(checkpoint["model"])



scale = (0.05, 1.0)
ratio = (3. / 4., 4. / 3.)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=scale, ratio=ratio),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
])

size = int((256 / 224) * 224)
eval_transform = transforms.Compose([
    transforms.Resize(size, interpolation=3),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])


batch_size = 16
num_classes = 100


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.dataset == "test":
    dataset = torchvision.datasets.CIFAR100(root='./local_datasets', train=False, download=True, transform=train_transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
elif args.dataset == "train":
    dataset = torchvision.datasets.CIFAR100(root='./local_datasets', train=True, download=True, transform=eval_transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)


with tqdm(total=len(data_loader)) as pbar:
    for batch_idx, (inputs, labels) in enumerate(data_loader, 0):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            cls_features = original_model(inputs)["pre_logits"]

        res = model(
            x=inputs,
            task_id=-1,
            cls_features=cls_features,
            train=False,
        )

         # iterate over each prediction
        batch_size = labels.shape[0]
        for i in range(batch_size):
            label = labels[i].item() # get the true class of the i-th element of the batch
            # iterate over each prompt selected prompt for the i-th element of the batch
            for j in range(5): 
                key_id = res["prompt_idx"][i][j].item() # get the j-th selected prompt of the i-th batch element
                # increase the key-class counter
                task = label // 10
                key_class_counts[key_id][label] += 1
                key_task_counts[key_id][task] += 1

        pbar.update(1)

        

print(f"key_class_counts_{args.dataset}_repo = {key_class_counts}")
print(f"key_task_counts_{args.dataset}_repo = {key_task_counts}")
