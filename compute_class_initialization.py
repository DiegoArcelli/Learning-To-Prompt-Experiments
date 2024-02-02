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
from datasets import build_continual_dataloader, get_dataset
from torchvision.datasets import CIFAR100, CIFAR10, SVHN
from torch.utils.data import DataLoader
from continual_datasets.continual_datasets import NotMNIST, FashionMNIST, MNIST_RGB
import copy


def get_class_data(data, targets, task):
    task_data = [d for d, t in zip(data, targets) if t == c]
    task_targets = [t for _, t in zip(data, targets) if t == c]
    return task_data, task_targets

def build_transform(is_train, input_size):
    resize_im = args.input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    
    return transforms.Compose(t)

parser = argparse.ArgumentParser(prog='Compute key counts', description='Compute the key-class and key-task counts')
parser.add_argument('--split', default="test", type=str,  choices=["train", "test"])
parser.add_argument('--dataset', default="5-datasets", type=str,  choices=["5-datasets", "cifar100"])
args = parser.parse_args()

args.input_size = 224
args.num_tasks = 5
args.task_inc = False
args.distributed = False
args.batch_size = 16
args.pin_mem = True
args.num_workers = 4
args.data_path = "./local_datasets"
args.train_mask = False
args.shuffle = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.dataset == "5-datasets":
    num_classes = 50
    num_tasks = 5
    num_keys = 20
    length = 10
    top_k = 4
elif args.dataset == "cifar100":
    num_classes = 100
    num_tasks = 10
    num_keys = 50
    length = 5
    top_k = 5

original_model = create_model(
    "vit_base_patch16_224",
    pretrained=True,
    num_classes=num_classes,
    drop_rate=0.0,
    drop_path_rate=0.0,
    drop_block_rate=None,
)

original_model = original_model.to(device)


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

if args.dataset == "cifar100":
    # if args.split == "test":
    #     cifar100 = CIFAR100(root='./local_datasets', train=False, download=True, transform=eval_transform)
    # elif args.split == "train":
    cifar100 = CIFAR100(root='./local_datasets', train=True, download=True, transform=eval_transform)
    datasets = []
    for c in range(100):
        data, targets = get_class_data(cifar100.data, cifar100.targets, c)
        dataset = copy.deepcopy(cifar100)
        dataset.data = data
        dataset.targets = targets
        datasets.append(dataset)
elif args.dataset == "5-datasets":
    transform_train = build_transform(False, 224)
    cifar10 = CIFAR10(args.data_path, train=True, download=True, transform=transform_train)
    mnist = MNIST_RGB(args.data_path, train=True, download=True, transform=transform_train)
    not_mnist = NotMNIST(args.data_path, train=True, download=True, transform=transform_train)
    fashion_mnist = FashionMNIST(args.data_path, train=True, download=True, transform=transform_train)
    svhn = SVHN(args.data_path, split='train', download=True, transform=transform_train)
    tasks = [svhn, mnist, cifar10, not_mnist, fashion_mnist]
    add = 0
    for t in range(len(tasks)):
        if hasattr(tasks[t], "targets"):
            if type(tasks[t].targets) != list:
                tasks[t].targets = tasks[t].targets.tolist()
            tasks[t].targets = [t + add for t in tasks[t].targets]
        else:
            if type(tasks[t].labels) != list:
                tasks[t].labels = tasks[t].labels.tolist()
            tasks[t].labels = [t + add for t in tasks[t].labels]

        add += 10

    datasets = []
    for task in tasks:
        if hasattr(task, "targets"):
            for c in set(task.targets):
                data, targets = get_class_data(task.data, task.targets, c)
                dataset = copy.deepcopy(task)
                dataset.data = data
                dataset.targets = targets
                datasets.append(dataset)
        else:
            for c in set(task.labels):
                data, labels = get_class_data(task.data, task.labels, c)
                dataset = copy.deepcopy(task)
                dataset.data = data
                dataset.labels = labels
                datasets.append(dataset)
        
    # datasets = [mnist, cifar10, fashion_mnist]
    # data_loader, _ = build_continual_dataloader(args)
    # data_loader = data_loader[0]["train" if args.split == "train" else "val"]

tot, corr = 0, 0
for dataset_id, dataset in enumerate(datasets):
    data_loader = DataLoader(
        dataset, sampler=None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )
    avg = None
    count = 0
    with tqdm(total=len(data_loader)) as pbar:
        for batch_idx, (inputs, labels) in enumerate(data_loader, 0):
            inputs = inputs.to(device)
            for i in range(inputs.shape[0]):
                # print(inputs[i].shape)

                with torch.no_grad():
                    cls_features = original_model(inputs[i].unsqueeze(0))["pre_logits"]

                if avg is None:
                    avg = cls_features[0]
                else:
                    avg = (count*avg + cls_features[0])/(count+1)

                count += 1

            pbar.update(1)

    torch.save(avg, f"./init_prompts/{args.dataset}/class/prompt_{dataset_id}.pt")

