import torch
import torchvision
import torchvision.transforms as transforms

# Define a transformation to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the CIFAR-100 training dataset
train_dataset = torchvision.datasets.CIFAR100(root='./local_datasets', train=False, download=True, transform=transform)


targets = train_dataset.targets
data = train_dataset.data

sorted_data_targets = sorted(list(zip(data, targets)), key=lambda x: x[1])
new_data = [d for d, t in sorted_data_targets]
new_targets = [t for d, t in sorted_data_targets]
print(new_targets)

# # Create a DataLoader to iterate through the dataset
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=2)

# # Print the classes in CIFAR-100
# classes = train_dataset.classes
# print("Original classes:")
# print(classes)

# # Sort the dataset with respect to classes
# sorted_dataset = sorted(train_dataset, key=lambda x: x[1])

# for img, label in sorted_dataset:
#     print(label, img.shape)