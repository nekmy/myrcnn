import os
import random
from functools import reduce
import pickle

import numpy as np
import scipy
from PIL import Image
import matplotlib.pylab as plt
from pycocotools.coco import COCO

class Dataset:
    COCO_DIR = "../../cocoapi"
    """
    IMG_DIR = os.path.join(COCO_DIR, "train2017", "train2017")
    ANNO_DIR = os.path.join(COCO_DIR, "annotations")
    JSON_PATH = os.path.join(ANNO_DIR, "instances_train2017.json")
    """
    IMG_DIR = os.path.join(COCO_DIR, "val2017", "val2017")
    ANNO_DIR = os.path.join(COCO_DIR, "annotations")
    JSON_PATH = os.path.join(ANNO_DIR, "instances_val2017.json")
    def __init__(self):
        self.coco = COCO(self.JSON_PATH)
        ids = sorted(self.coco.getCatIds())
        self.cls_ids = {0: 0}
        for i, cls_id in enumerate(ids):
            self.cls_ids[cls_id] = 1 + i
        pass
        #num_class(self.coco)
    
    def make_batch(self, batch_size, img_ids=None):
        if img_ids:
            pass
        else:
            # imageをランダムで選ぶ.
            choice = list(range(len(self.coco.dataset["images"])))
            random.shuffle(choice)
            # 無作為にimg_idを選ぶ.
            img_ids = [self.coco.dataset["images"][c]["id"] for c in choice[:batch_size]]
        images = self.coco.loadImgs(img_ids)
        images = [plt.imread(os.path.join(self.IMG_DIR, image["file_name"])) for image in images]
        images = [image if image.ndim == 3 else np.tile(np.expand_dims(image, axis=-1), 3) for image in images]
        anno_ids = [self.coco.getAnnIds(img_id) for img_id in img_ids]
        annos_batch = [self.coco.loadAnns(anno_id) for anno_id in anno_ids]
        # 追加
        #gt_boxes = [b if len(b) else np.zeros((1, 4)) for b in gt_boxes]
        #gt_masks = [[self.coco.annToMask(anno)[int(gt_boxes[i][j][0]):int(gt_boxes[i][j][2]), int(gt_boxes[i][j][1]):int(gt_boxes[i][j][3])] for j, anno in enumerate(annos)] for i, annos in enumerate(annos_batch)]
        gt_masks = [[self.coco.annToMask(anno) for j, anno in enumerate(annos)] for i, annos in enumerate(annos_batch)]
        gt_masks = [np.array(m) for m in gt_masks]
        gt_masks = [m if m.size else np.empty((0,) + im.shape[:2]) for m, im in zip(gt_masks, images)]
        gt_boxes_old = [[anno["bbox"] for anno in annos] for annos in annos_batch]
        gt_boxes_old = [np.array(b) for b in gt_boxes_old]
        gt_boxes_old = [b if b.size else np.empty((0, 4)) for b in gt_boxes_old]
        gt_boxes_old = convert_boxes(gt_boxes_old)
        gt_boxes = [[make_gt_box_from_mask(m) for m in gt_mask] for gt_mask in gt_masks]
        gt_boxes = [np.array(b) for b in gt_boxes]
        gt_boxes = [b if b.size else np.empty((0, 4)) for b in gt_boxes]
        #gt_boxes = [m if len(m) else np.zeros((1,) + im.shape[:2]) for m, im in zip(gt_masks, images)]
        gt_class_ids = [[anno["category_id"] for anno in annos] for annos in annos_batch]
        gt_class_ids = [[self.cls_ids[i] for i in gt_class_id] for gt_class_id in gt_class_ids]
        gt_class_ids = [np.array(ids) for ids in gt_class_ids]
        gt_class_ids = [ids if ids.size else np.empty((0,)) for ids in gt_class_ids]
        #gt_class_ids = [ids if ids.size else np.zeros((1,)) for ids in gt_class_ids]
        return images, gt_boxes, gt_masks, gt_class_ids, img_ids

def make_gt_box_from_mask(mask):
    arg = np.where(mask==True)
    y1 = np.min(arg[0])
    x1 = np.min(arg[1])
    y2 = np.max(arg[0])
    x2 = np.max(arg[1])
    y2 = y2 + 1
    x2 = x2 + 1
    return [y1, x1, y2, x2]
    

def num_class(coco):
    image_ids = [im["id"] for im in coco.dataset["images"]]
    anno_ids = [coco.getAnnIds(id) for id in image_ids]
    anno_ids = reduce(lambda x, y: x + y, anno_ids)
    class_ids = [anno["category_id"] for anno in coco.loadAnns(anno_ids)]
    return max(class_ids) + 1

def convert_boxes(boxes_batch):
    # x1, y1, w, h -> y1, x1, y2, x2
    lens = [len(boxes) for boxes in boxes_batch]
    boxes_batch = np.concatenate(boxes_batch)
    #if boxes_batch.shape[0] != 0:
    x1 = boxes_batch[:, 0]
    y1 = boxes_batch[:, 1]
    x2 = boxes_batch[:, 0] + boxes_batch[:, 2]
    y2 = boxes_batch[:, 1] + boxes_batch[:, 3]
    boxes_batch = np.stack([y1, x1, y2, x2], axis=1)
    #boxes_batch = boxes_batch.tolist()
    new_boxes_batch = []
    for l in lens:
        new_boxes_batch.append(boxes_batch[:l])
        #del(boxes_batch[:l])
        boxes_batch = np.delete(boxes_batch, slice(None, l), axis=0)
    return new_boxes_batch

def main():
    dataset = Dataset()
    images, gt_boxes, gt_masks, gt_class_ids = dataset.make_batch(batch_size=5, img_ids=[139, 285, 632])
    #images, gt_boxes, gt_masks, gt_class_ids = dataset.make_batch(batch_size=5)
    pass

if __name__ == "__main__":
    main()
    
