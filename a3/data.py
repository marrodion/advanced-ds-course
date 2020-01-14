import torch
import tqdm
from PIL import Image
import pandas as pd
from pathlib import Path

class SignsDataset(torch.utils.data.Dataset):
    def __init__(self, img_root, ann_root, transforms):
        self.img_root = img_root        
        self.transforms = transforms
        self.usecols = ['class', 'xtl', 'ytl', 'xbr', 'ybr']
        self.imgs = list(sorted(Path(img_root).rglob('*.jpg')))
        self.target = dict()
        annot = list(sorted(Path(ann_root).rglob('*.tsv')))
        classes = set()
        for ann, f in tqdm(zip(annot, self.imgs), total=len(annot)):
            df = pd.read_csv(ann, sep='\t', usecols=self.usecols)
            self.target[f] = df.values
            classes |= set(df['class'])
        self.idx2cls = dict(enumerate(classes))
        self.cls2idx = {v: k for k, v in self.idx2cls.items()}
        
    
    def __getitem__(self, idx):
        f = self.imgs[idx]
        img = Image.open(f)
        target = {}
        trg = self.target[f]
        cls = [self.cls2idx[c] for c in trg[:, 0]]
        boxes = torch.tensor(list(row for row in trg[:, 1:]), dtype=torch.float32)
        
        target['boxes'] = boxes
        target['labels'] = torch.tensor(cls, dtype=torch.int64).unsqueeze(1)
        
        if self.transforms is not None:
            img = self.transforms(img)
        return img, target
    
    def __len__(self):
        return len(self.imgs)

def collate_func(batch):
    imgs = []
    bbox = []
    labels = []
    for b in batch:
        imgs.append(b[0])
        bbox.append(b[1]['boxes'])
        labels.append(b[1]['labels'])
    imgs = torch.stack(imgs, dim=0)
    target = dict(
        boxes=bbox,
        labels=labels
    )
    return imgs, target