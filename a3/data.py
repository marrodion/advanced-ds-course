import torch
import json
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image
import pandas as pd
from pathlib import Path
import torchvision
from collections import defaultdict
from skmultilearn.model_selection import IterativeStratification
import tensorflow as tf
from object_detection.utils import dataset_util
import contextlib2
from object_detection.dataset_tools import tf_record_creation_util
from object_detection.protos.string_int_label_map_pb2 import StringIntLabelMap, StringIntLabelMapItem
from google.protobuf import text_format
import io

INFO=dict(
        year=2019,
        version="1.0",
        description="Signs dataset",
        contributor="",
        url="",
        date_created="2019/01/01")

LICENSES=[dict(id=0, name="", url="")]

CLS_MAP = {
    '2.1': '2.1',
    '2.4': '2.4',
    '3.1': '3.1',
    '3.24': '3.24',
    '3.27': '3.27',
    '4.1.1': '4.1',
    '4.1.2': '4.1',
    '4.1.3': '4.1',
    '4.1.4': '4.1',
    '4.1.5': '4.1',
    '4.1.6': '4.1',
    '4.2.1': '4.2',
    '4.2.2': '4.2',
    '4.2.3': '4.2',
    '5.19.1': '5.19',
    '5.19.2': '5.19',
    '5.20': '5.20',
    '8.22.1': '8.22',
    '8.22.2': '8.22',
    '8.22.3': '8.22'
}

class_mapping = defaultdict(lambda: 'OTH')
class_mapping.update(CLS_MAP)
    

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
            df = read_ann_df(ann)
            if df.empty and not keep_empty:
                empty_images.add(f)
            else:
                self.target[f] = df.values
            classes |= set(df['class'])
        self.imgs = [f for f in self.imgs if f not in empty_images]
        self.idx2cls = dict(enumerate(classes))
        ci = {v: k for k, v in self.idx2cls.items()}
        cls2idx = defaultdict(lambda: ci['OTH'])
        cls2idx.update(ci)
        self.cls2idx = cls2idx
        
    
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


def read_ann_df(path, usecols=('class', 'xtl', 'ytl', 'xbr', 'ybr')):
    df = pd.read_csv(path, sep='\t', usecols=usecols, dtype={
                'class': str,
                'xtl': np.float32, 
                'ytl': np.float32, 
                'xbr': np.float32, 
                'ybr': np.float32
            }).fillna('NA')

    df.loc[:, 'class'] = df.loc[:, 'class'].map(class_mapping)
    return df

def copy_images(img_path, out_path):
    files = list(Path(img_path).rglob('*.jpg'))
    dest_base = Path(out_path)
    for f in tqdm(files):
        dest = dest_base / f"{f.parts[-2]}_{f.name}"
        shutil.copy(f, dest)


def annotations_to_coco(out, annot=None, imgs=None, ann_path=None, img_path=None):
    if annot is None:
        ann_path = Path(ann_path)    
        annot = list(sorted(ann_path.rglob('*.tsv')))
    if imgs is None:
        img_path = Path(img_path)
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
        df = read_ann_df(ann)
        df['image_id'] = img_id
        dfs.append(df)
    df = pd.concat(dfs, axis=0)
    
    cat_ids, idx2cat = pd.factorize(df['class'])
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


def get_cls_mapping(annotations):
    classes = set()
    usecols = ['class', 'xtl', 'ytl', 'xbr', 'ybr']
    for ann in tqdm(annotations, total=len(annotations), desc='Cls mapping parse'):
        df = pd.read_csv(ann, sep='\t', usecols=usecols, dtype={
            'class': str,
            'xtl': np.float32, 
            'ytl': np.float32, 
            'xbr': np.float32, 
            'ybr': np.float32
        }).fillna('NA')
        classes |= set(df['class'].map(class_mapping))
    idx2cls = dict(enumerate(classes, 1))
    ci = {v: k for k, v in idx2cls.items()}
    cls2idx = defaultdict(lambda: ci['OTH'])
    cls2idx.update(ci)
    return idx2cls, cls2idx  


def coco_collate_fn(batch):
    return tuple(zip(*batch))


def get_data_loader(ds, batch_size, shuffle=True):
    return torch.utils.data.DataLoader(ds, 
                batch_size=batch_size, 
                collate_fn=coco_collate_fn, 
                pin_memory=True,
                shuffle=shuffle
    )


def train_test_split(ds, stratify=True, test_size=0.2, order=5):
    idx_class = np.zeros((len(ds), len(ds.cls2idx)))
    for i, k in tqdm(enumerate(ds.imgs), total=len(ds)):
        v = ds.target[k]
        label_idx = np.array([ds.cls2idx[l] for l in v[:, 0]])
        idx_class[i, label_idx] = 1    
    return get_train_test_idx(idx_class, test_size, order=order)


def get_train_test_idx(labels, test_size, order=5):
    itr = IterativeStratification(n_splits=2, order=order, 
                                  sample_distribution_per_fold=[test_size, 1.0-test_size])
    train, test = next(itr.split(X=np.arange(labels.shape[0]), y=labels))
    return train, test


def create_tf_example(img_fn, ann_fn, cls2idx):
    img = Image.open(img_fn).convert('RGB')
    height = img.height # Image height
    width = img.width # Image width
    filename = bytes(img_fn) # Filename of the image. Empty if image is not from file
    
    with io.BytesIO() as output:
        img.save(output, format="JPEG")
        contents = output.getvalue()
    encoded_image_data = contents # Encoded image bytes

    image_format = bytes(img_fn.suffix, 'utf-8') # b'jpeg' or b'png'
    
    ann = pd.read_csv(ann_fn, 
                      sep='\t', 
                      usecols=['class', 'xtl', 'ytl', 'xbr', 'ybr'], 
                      dtype={
                        'class': str,
                        'xtl': np.float32, 
                        'ytl': np.float32, 
                        'xbr': np.float32, 
                        'ybr': np.float32
                      }
                ).fillna('NA')

    xmins = ann['xtl'].values / width # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = ann['xbr'].values / width # List of normalized right x coordinates in bounding box
            # (1 per box)
    ymins = ann['ytl'].values / height # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = ann['ybr'].values / height # List of normalized bottom y coordinates in bounding box
            # (1 per box)
    classes_text = ann['class'].map(lambda x: bytes(x, 'utf-8')).tolist() # List of string class name of bounding box (1 per box)
    classes = [cls2idx[c] for c in ann['class']] # List of integer class id of bounding box (1 per box)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def to_tf_object_detection(ann_path, img_path, out, num_shards=100):
    ann_path = Path(ann_path)
    img_path = Path(img_path)
    
    annot = list(sorted(ann_path.rglob('*.tsv')))    
    imgs = list(sorted(img_path.rglob('*.jpg')))
    to_tf_record(annot, imgs, out, num_shards=num_shards)


def to_tf_record(ann_files, img_files, out, cls2idx=None, num_shards=100):
    if cls2idx is None:
        _, cls2idx = get_cls_mapping(annot)
    if num_shards == 1:
        writer = tf.python_io.TFRecordWriter(out)
        for img_fn, ann_fn in tqdm(zip(img_files, ann_files), total=len(img_files), desc='TFRecord write'):
            tf_example = create_tf_example(img_fn, ann_fn, cls2idx)
            writer.write(tf_example.SerializeToString())
        writer.close()
    else:
        with contextlib2.ExitStack() as tf_record_close_stack:
            output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
            tf_record_close_stack, out, num_shards)
            for index, (img_fn, ann_fn) in tqdm(enumerate(zip(img_files, ann_files)), 
                                              total=len(img_files), 
                                              desc='TFRecord write'):
                tf_example = create_tf_example(img_fn, ann_fn, cls2idx)
                output_shard_index = index % num_shards
                output_tfrecords[output_shard_index].write(tf_example.SerializeToString())


def write_label_map(idx2cls, out):
    msg = StringIntLabelMap()
    for i, name in idx2cls.items():
        msg.item.append(StringIntLabelMapItem(id=i, name=name))
    text = str(text_format.MessageToBytes(msg, as_utf8=True), 'utf-8')
    with open(out, 'w+') as fh:
        fh.write(text)


if __name__ == 'main':
    annotations_to_coco('./data/annotations/train', './data/images/train', './data/coco_ds/train/annotations.json')
