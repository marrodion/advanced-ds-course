import torch
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from tqdm import tqdm

def evaluate(model, data_loader, device):
    model.eval()
    # coco api


class CocoEvaluator():
    pass

def get_coco_api_from_dataset(dataset):
    pass

def convert_to_coco_api(ds):
    coco_ds = COCO()
    ann_id = 1
    dataset = dict(images=[], categories=[], annotations=[])
    categories = set()
    for img_idx in tqdm(range(len(ds)), total=len(ds)):
        img, targets = ds[img_idx]
        image_id = targets["image_id"].item()
        img_dict = {}
        img_dict['id'] = image_id
        img_dict['height'] = img.shape[-2]
        img_dict['width'] = img.shape[-1]
        dataset['images'].append(img_dict)
        bboxes = targets["boxes"]
        if bboxes.nelement() != 0:
            bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()
        labels = targets['labels'].tolist()
        areas = targets['area'].tolist()
        iscrowd = targets['iscrowd'].tolist()
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann['image_id'] = image_id            
            ann['bbox'] = bboxes[i]            
            ann['category_id'] = labels[i]
            categories.add(labels[i])
            ann['area'] = areas[i]
            ann['iscrowd'] = iscrowd[i]
            ann['id'] = ann_id
            dataset['annotations'].append(ann)
            ann_id += 1
    dataset['categories'] = [{'id': i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds



