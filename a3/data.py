import torch
import json
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image
import pandas as pd
from pathlib import Path
import torchvision

INFO=dict(
        year=2019,
        version="1.0",
        description="Signs dataset",
        contributor="",
        url="",
        date_created="2019/01/01")

LICENSES=[dict(id=0, name="", url="")]





class SignsDataset(torch.utils.data.Dataset):
    def __init__(self, img_root, ann_root, transforms, keep_empty=False):
        self.img_root = img_root        
        self.transforms = transforms
        self.usecols = ['class', 'xtl', 'ytl', 'xbr', 'ybr']
        self.imgs = list(sorted(Path(img_root).rglob('*.jpg')))
        self.target = dict()
        annot = list(sorted(Path(ann_root).rglob('*.tsv')))
        classes = set()
        empty_images = set()
        for ann, f in tqdm(zip(annot, self.imgs), total=len(annot)):
            df = pd.read_csv(ann, sep='\t', usecols=self.usecols, dtype={
                'class': str,
                'xtl': np.float32, 
                'ytl': np.float32, 
                'xbr': np.float32, 
                'ybr': np.float32
            }).fillna('NA')
            if df.empty and not keep_empty:
                empty_images.add(f)
            else:
                self.target[f] = df.values
            classes |= set(df['class'])
        self.imgs = [f for f in self.imgs if f not in empty_images]
        self.idx2cls = dict(enumerate(classes))
        self.cls2idx = {v: k for k, v in self.idx2cls.items()}
        
    
    def __getitem__(self, idx):
        f = self.imgs[idx]
        img = Image.open(f)
        target = {}
        trg = self.target[f]
        num_objs = trg.shape[0]

        cls = [self.cls2idx[c] for c in trg[:, 0]]
        boxes = []
        for i in range(num_objs):
            boxes.append([trg[i, 1], trg[i, 2], trg[i, 3], trg[i, 4]]) 
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        area = torch.tensor([0])
        if boxes.nelement() != 0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])        
        
        target['boxes'] = boxes
        target['labels'] = torch.tensor(cls, dtype=torch.int64)
        target['image_id'] = torch.tensor([idx])
        target['area'] = area
        target['iscrowd'] = torch.zeros((num_objs,), dtype=torch.int64)
        
        if self.transforms is not None:
            img = self.transforms(img)
        return img, target
    
    def __len__(self):
        return len(self.imgs)

def copy_images(img_path, out_path):
    files = list(Path(img_path).rglob('*.jpg'))
    dest_base = Path(out_path)
    for f in tqdm(files):
        dest = dest_base / f"{f.parts[-2]}_{f.name}"
        shutil.copy(f, dest)


def annotations_to_coco(ann_path, img_path, out):
    ann_path = Path(ann_path)
    img_path = Path(img_path)
    
    annot = list(sorted(ann_path.rglob('*.tsv')))
    imgs = list(sorted(img_path.rglob('*.jpg')))

    dfs = []
    images = []
    # Coco ann should start from 1
    for ann, f, img_id in tqdm(zip(annot, imgs, range(len(annot))), total=len(annot)):
        img = Image.open(f).convert('RGB')
        images.append(dict(
            id=img_id,
            width=img.width,
            height=img.height,
            file_name=str(f.parts[-1]),
            license=0,
            flickr_url="",
            coco_url="",
            date_captured="2019-01-01"
        ))
        
        
        df = pd.read_csv(ann, sep='\t', usecols=['class', 'xtl', 'ytl', 'xbr', 'ybr'], 
        dtype={
            'class': str,
            'xtl': np.float32, 
            'ytl': np.float32, 
            'xbr': np.float32, 
            'ybr': np.float32
        })
        df['image_id'] = img_id
        dfs.append(df)
    df = pd.concat(dfs, axis=0)
    
    cat_ids, idx2cat = pd.factorize(df['class'].fillna('NA'))
    df['category_id'] = cat_ids
    categories = []
    for i, cat in enumerate(idx2cat):
        categories.append(dict(
            id=i,
            name=cat,
            supercategory=cat
        ))

    annotations = []

    for i, row in enumerate(df.itertuples()):
        bbox = [
            row.xtl,
            row.ytl,
            row.xbr - row.xtl,
            row.ybr - row.ytl
        ]
        if bbox[-1] < 0 or bbox[-2] < 0:
            raise Exception()
        area = bbox[-1] * bbox[-2]
        annotations.append(dict(
            id=i+1,
            image_id=row.image_id,
            category_id=row.category_id,
            area=area,
            bbox=bbox,
            iscrowd=0        
        ))
    result = dict(
        info=INFO,
        licenses=LICENSES,
        images=images,
        annotations=annotations,
        categories=categories
    )
    with open(out, 'w+') as fh:
        json.dump(result, fh)


def coco_collate_fn(batch):
    return tuple(zip(*batch))


def get_data_loader(ds, batch_size, shuffle=True):
    return torch.utils.data.DataLoader(ds, 
                batch_size=batch_size, 
                collate_fn=coco_collate_fn, 
                pin_memory=True,
                shuffle=shuffle
    )

if __name__ == 'main':
    annotations_to_coco('./data/annotations/train', './data/images/train', './data/coco_ds/train/annotations.json')
