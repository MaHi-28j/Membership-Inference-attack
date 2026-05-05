import sys
import torch
import numpy as np
import pandas as pd
import requests
import torch.nn.functional as F

from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.models import resnet18
import torchvision.transforms as transforms
from tqdm import tqdm

# config
try:
    BASE = Path(__file__).parent
except NameError:
    BASE = Path(".")

PUB_PATH   = BASE / "pub.pt"
PRIV_PATH  = BASE / "priv.pt"
MODEL_PATH = BASE / "model.pt"
REF_MODEL_PATH = BASE / "ref_model.pt"
OUTPUT_CSV = BASE / "submission.csv"

BASE_URL = "http://34.63.153.158"
API_KEY  = "05c66a14c43988572b6ba4f2e502460c"
TASK_ID  = "01-mia"

RETRAIN_REF = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

MEAN = [0.7406, 0.5331, 0.7059]
STD  = [0.1491, 0.1864, 0.1301]


# dataset classes
class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids        = []
        self.imgs       = []
        self.labels     = []
        self.transform  = transform

    def __getitem__(self, index):
        id_   = self.ids[index]
        img   = self.imgs[index]
        label = self.labels[index]
        if self.transform:
            img = self.transform(img)
        return id_, img, label

    def __len__(self):
        return len(self.ids)


class MembershipDataset(TaskDataset):
    def __init__(self, transform=None):
        super().__init__(transform)
        self.membership = []

    def __getitem__(self, index):
        id_, img, label = super().__getitem__(index)
        return id_, img, label, self.membership[index]


# load data
print("Data Loading:")
pub_ds  = torch.load(PUB_PATH,  weights_only=False)
priv_ds = torch.load(PRIV_PATH, weights_only=False)

train_transform = transforms.Compose([
    transforms.Resize((36, 36)),
    transforms.RandomCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.Normalize(mean=MEAN, std=STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Normalize(mean=MEAN, std=STD),
])

pub_ds.transform  = eval_transform
priv_ds.transform = eval_transform


# split public dataset
print("Dataset Split:")
non_member_idx = [i for i, m in enumerate(pub_ds.membership) if int(m) == 0]
member_idx     = [i for i, m in enumerate(pub_ds.membership) if int(m) == 1]
print(f"Members: {len(member_idx)}")
print(f"Non-members: {len(non_member_idx)}")


# model architecture
def make_model():
    m = resnet18(weights=None)
    m.conv1  = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    m.maxpool = torch.nn.Identity()
    m.fc      = torch.nn.Linear(512, 9)
    return m


# train reference model
def train_reference_model(pub_ds, non_member_idx, epochs=30, lr=0.05, batch_size=128):
    print("Reference Model Training:")

    class RefDataset(Dataset):
        def __init__(self, base_ds, indices, transform):
            self.base    = base_ds
            self.indices = indices
            self.transform = transform

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, i):
            idx = self.indices[i]
            raw_img = self.base.imgs[idx]
            label   = self.base.labels[idx]
            img = self.transform(raw_img)
            return img, label

    ref_dataset = RefDataset(pub_ds, non_member_idx, train_transform)
    loader = DataLoader(ref_dataset, batch_size=batch_size, shuffle=True,
                        num_workers=2, pin_memory=True)

    ref_model = make_model().to(device)

    optimizer = torch.optim.SGD(ref_model.parameters(), lr=lr,
                                momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ref_model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct    = 0
        total      = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = ref_model(imgs)
            loss   = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(imgs)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += len(imgs)
        scheduler.step()
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/total:.4f} - Accuracy: {correct/total:.3f}")

    ref_model.eval()
    return ref_model


# reference model load or train
if RETRAIN_REF or not REF_MODEL_PATH.exists():
    ref_model = train_reference_model(pub_ds, non_member_idx)
    torch.save(ref_model.state_dict(), REF_MODEL_PATH)
    print("Reference Model Saved")
else:
    print("Reference Model Loading")
    ref_model = make_model()
    ref_model.load_state_dict(torch.load(REF_MODEL_PATH, map_location=device))
    ref_model = ref_model.to(device)
    ref_model.eval()


# load target model
print("Target Model Loading:")
target_model = make_model()
target_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
target_model = target_model.to(device)
target_model.eval()


# compute scores
def compute_scores(dataset):
    ids_out    = []
    labels_out = []
    member_out = []
    scores_out = []

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Scoring"):
            batch = dataset[idx]
            if len(batch) == 4:
                id_, img, label, member = batch
            else:
                id_, img, label = batch
                member = None

            img_t   = img.unsqueeze(0).to(device)
            label_t = torch.tensor([label]).to(device)

            loss_target = F.cross_entropy(target_model(img_t), label_t).item()
            loss_ref    = F.cross_entropy(ref_model(img_t),    label_t).item()

            score = loss_ref - loss_target

            ids_out.append(id_)
            labels_out.append(label)
            member_out.append(member)
            scores_out.append(score)

    return {
        "ids":        ids_out,
        "labels":     torch.tensor(labels_out),
        "membership": member_out,
        "scores":     torch.tensor(scores_out, dtype=torch.float32),
    }


# scoring
print("Score Computation:")
priv_result = compute_scores(priv_ds)


# save submission
print("Saving Submission:")
scores = priv_result["scores"]
scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)

df = pd.DataFrame({
    "id":    [str(i) for i in priv_result["ids"]],
    "score": scores_norm.numpy(),
})
df.to_csv(OUTPUT_CSV, index=False)
print(f"Submission Saved: {OUTPUT_CSV}")
