import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib import patches
import numpy as np

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
        target['image_id'] = torch.tensor([idx])
        target['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target['iscrowd'] = torch.zeros((num_objs,), dtype=torch.int64)
        
        if self.transforms is not None:
            img = self.transforms(img)
        return img, target
    
    def __len__(self):
        return len(self.imgs)

def collate_func(batch):
    imgs = []
    bbox = []
    labels = []
    ids = []
    areas = []
    iscrowds = []
    for b in batch:
        imgs.append(b[0])
        bbox.append(b[1]['boxes'])
        labels.append(b[1]['labels'])        
        ids.append(b[1]['image_id'])
        areas.append(b[1]['area'])
        iscrowds.append(b[1]['iscrowd'])
    imgs = torch.stack(imgs, dim=0)
    target = dict(
        boxes=bbox,
        labels=labels,
        image_id=ids,
        area=areas,
        iscrowd=iscrowds
    )
    return imgs, target


def draw_boxes(img, boxes, labels, ax=None):
    img = img.numpy()
    img = np.transpose(img, (1, 2, 0))
    if ax is None:
        _, ax = plt.subplots(1, figsize=(12, 20))
    ax.imshow(img)
    for box, label in zip(boxes, labels):
        box = box.numpy()
        h = box[2] - box[0]
        w = box[3] - box[1]
        x, y = box[0], box[1]
        rec = patches.Rectangle((x, y), w, h, lw=2.0, color='r', fill=False)
        ax.add_patch(rec)
        ax.text(x+5, y+5, label.item(), fontsize=12, color='blue')
        plt.axis('off')
    return ax


def plot_grid(batch, rows=2, cols=2):
    fig = plt.figure()
    grid = ImageGrid(fig, 111, nrows_ncols=(2, 2), axes_pad=0.1)
    n = batch[0].shape[0]
    
    for i, ax in zip(range(n), grid):
        img = batch[0][i, :, :].squeeze()
        boxes = batch[1][0]['bbox'][i]
        labels = batch[1][0]['category_id'][i]
        print(labels)
        draw_boxes(img, boxes, labels, ax=ax)
    plt.axis('off')
    plt.show()

