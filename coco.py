import torchvision.datasets
import torchvision.transforms
import torch


def load_dataset(path):
    data_path = path
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor()
    ])
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=8,
        num_workers=0,
        shuffle=False
    )
    return train_loader


'''
need to resize
'''

if __name__ == "__main__":
    dataloader = load_dataset("")
    for batch_idx, (image, target) in enumerate(dataloader):
        image = image.cuda()
