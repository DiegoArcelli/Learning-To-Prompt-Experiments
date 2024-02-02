import torch
import models

model = torch.load("l2p_cifar100_repo.pt")
checkpoint = torch.load("./output_experiment/checkpoint/task10_checkpoint.pth")
checkpoint["counts"] = {
    "key_class_counts": model.key_class_counts,
    "key_task_counts": model.key_task_counts
}
torch.save(checkpoint, "l2p_cifar100_task_wise.pth")
