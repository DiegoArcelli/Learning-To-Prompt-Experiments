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
from datasets import build_continual_dataloader
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(prog='Compute key counts', description='Compute the key-class and key-task counts')
parser.add_argument('--split', default="test", type=str,  choices=["train", "test"])
parser.add_argument('--dataset', default="5-datasets", type=str,  choices=["5-datasets", "cifar100"])
parser.add_argument('--batchwise', action="store_true")
parser.add_argument('--ref_task', default=0, type=int)
# parser.add_argument('--model', default="transfer", type=str)
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
args.class_incremental = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batchwise = args.batchwise
ref_task = args.ref_task
print(batchwise)

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

key_class_counts = {k: {c: 0 for c in range(num_classes)} for k in range(num_keys)}
key_task_counts = {k: {t: 0 for t in range(num_tasks)} for k in range(num_keys)}

original_model = create_model(
    "vit_base_patch16_224",
    pretrained=True,
    num_classes=num_classes,
    drop_rate=0.0,
    drop_path_rate=0.0,
    drop_block_rate=None,
)

original_model = original_model.to(device)


model = create_model(
        "vit_base_patch16_224",
        pretrained=True,
        num_classes=num_classes,
        drop_rate=0.0,
        drop_path_rate=0.0,
        drop_block_rate=None,
        prompt_length=length,
        embedding_key="cls",
        prompt_init="uniform",
        prompt_pool=True,
        prompt_key=True,
        pool_size=num_keys,
        top_k=top_k,
        batchwise_prompt=batchwise,
        prompt_key_init="uniform",
        head_type="prompt",
        use_prompt_mask=True,
    )
# model = torch.load("l2p.pt")

# if args.model == "transfer":
#     pass
# elif args.model == "no_transfer":
#     pass

model = model.to(device)
if args.dataset == "5-datasets":
    checkpoint = torch.load("./output_5_datasets_no_transfer_no_task_id/checkpoint/task5_checkpoint.pth")
elif args.dataset == "cifar100":
    # checkpoint = torch.load("./output_experiment/checkpoint/task10_checkpoint.pth")
    checkpoint = torch.load("./output_task_wise_transfer_scale/checkpoint/task10_checkpoint.pth")
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

if args.dataset == "cifar100":
    if args.split == "test":
        dataset = CIFAR100(root='./local_datasets', train=False, download=True, transform=eval_transform)
    elif args.split == "train":
        dataset = CIFAR100(root='./local_datasets', train=True, download=True, transform=eval_transform)
    targets = dataset.targets
    data = dataset.data
    sorted_data_targets = sorted(list(zip(data, targets)), key=lambda x: x[1])
    new_data = [d for d, t in sorted_data_targets]
    new_targets = [t for d, t in sorted_data_targets]
    new_data = new_data[ref_task*1000:(ref_task+1)*1000]
    new_targets = new_targets[ref_task*1000:(ref_task+1)*1000]
    dataset.data = new_data
    dataset.targets = new_targets
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
elif args.dataset == "5-datasets":
    data_loader, _ = build_continual_dataloader(args)
    data_loader = data_loader[0]["train" if args.split == "train" else "val"]

tot, corr = 0, 0

# add = 30

if args.dataset == "cifar100":
    add = 0

# add = 30

with tqdm(total=len(data_loader)) as pbar:
    for batch_idx, (inputs, labels) in enumerate(data_loader, 0):
        inputs = inputs.to(device)
        labels = labels.to(device)

        labels += add
        with torch.no_grad():
            cls_features = original_model(inputs)["pre_logits"]
        for i in range(len(labels)):
            label = labels[i].item() # get the true class of the i-th element of the batch
        
        res = model(
            x=inputs,
            task_id=-1,
            cls_features=cls_features,
            train=False,
        )

        # pred = res["logits"].argmax(dim=1)
        # print(labels)
        # print(pred)
        # print("\n\n")

        # print(labels)
        # iterate over each prediction
        # batch_size = labels.shape[0]

        # # print(labels)
        task_idx = res["prompt_idx"] // top_k
        pred, _ = torch.mode(task_idx, dim=1)
        # print(pred)
        tasks = labels // 10

        # print(pred)
        # print(tasks)

        # for i in range(batch_size):
        #     label = labels[i].item() # get the true class of the i-th element of the batch
        #     print(res["prompt_idx"][i], task_idx[i], pred[i].item(), tasks[i].item(), label)
        # corr += (labels == pred).sum().item()
        corr += (tasks == pred).sum().item()
        # print(corr/batch_size)
        tot += len(pred)
        pbar.update(1)

print(f"Accuracy {corr/tot}")

        

# print(f"key_class_counts_{args.dataset}_repo = {key_class_counts}")
# print(f"key_task_counts_{args.dataset}_repo = {key_task_counts}")
