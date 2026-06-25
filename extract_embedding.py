import torch
from torchvision import transforms
from torchvision import models
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn.functional as F

def get_transform():
    return transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])

def extract():
    stl10_test = datasets.STL10(root=r'/data', split='test', download=True, transform=get_transform())

    print("====Extract embedding====")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataloader = DataLoader(stl10_test, batch_size=64, shuffle=False)

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    modules = list(model.children())[:-1]
    model = torch.nn.Sequential(*modules, torch.nn.AdaptiveAvgPool2d((1, 1)))
    model.eval().to(device)

    feature_list = []
    for imgs, _ in dataloader:
        imgs = imgs.to(device)
        with torch.no_grad():
            embs = model(imgs)
            embs = torch.flatten(embs, start_dim=1)
            embs = F.normalize(embs, p=2, dim=1)
            feature_list.append(embs)

    feature_tensor = torch.cat(feature_list, dim=0)

    torch.save(feature_tensor, 'image_features.pt')
    print("Đã lưu Tensor thành công!")

if __name__ == "__main__":
    extract()