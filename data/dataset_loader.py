# data/dataset_loader.py
import os
import json
import numpy as np
import tensorflow as tf
from pycocotools.coco import COCO
from PIL import Image
from typing import Optional, Tuple, List

"""
RefCOCO dataset loader that integrates with pycocotools COCO API.

Expected directory layout:
data/
  refcoco/
    images/                # COCO images (train2014 / val2014)
    instances_trainval2014/annotations/instances_train2014.json  # COCO instances
    refs/                   # folder containing RefCOCO json files (refcoco_train.json, refcoco_val.json, etc.)
      refcoco_train.json
      refcoco_val.json
      ...
"""

class RefCOCODataset:
    def __init__(
        self,
        data_root: str = "data/refcoco",
        split: str = "train",                # 'train' / 'val' / 'testA' / 'testB' depending on the annotation split you have
        image_size: Tuple[int,int] = (224, 224),
        max_samples: Optional[int] = None,
        text_vectorizer: Optional[tf.keras.layers.TextVectorization] = None,
    ):
        self.data_root = data_root
        self.split = split
        self.image_size = image_size
        self.max_samples = max_samples
        self.text_vectorizer = text_vectorizer

        # paths
        self.images_dir = os.path.join(self.data_root, "images")
        self.coco_instances = os.path.join(self.data_root, "instances_trainval2014", "annotations", "instances_train2014.json")
        self.refcoco_json = os.path.join(self.data_root, "refs", f"refcoco_{split}.json")

        # checks
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}. Please download COCO images into this folder.")
        if not os.path.exists(self.coco_instances):
            raise FileNotFoundError(f"COCO instances JSON not found: {self.coco_instances}. Please download COCO annotations.")
        if not os.path.exists(self.refcoco_json):
            raise FileNotFoundError(f"RefCOCO annotation JSON not found: {self.refcoco_json}. Please place the RefCOCO refs JSON at {self.refcoco_json}.")

        # load refcoco annotations (format depends on source; this code handles common structures)
        with open(self.refcoco_json, "r") as f:
            self.ref_ann = json.load(f)

        # If the RefCOCO JSON follows many public releases, it is a list of dicts where each dict has:
        # { 'image_id': int, 'ann_id': int, 'ref_id': int, 'sentences': [{'sent': "..."}], 'ref': "..." } or similar.
        # We normalize to a list of entries with: image_id, ann_id, ref (string).
        self.entries = self._parse_refjson(self.ref_ann)

        if max_samples:
            self.entries = self.entries[:max_samples]

        # init COCO API for instance data
        self.coco = COCO(self.coco_instances)

    def _parse_refjson(self, loaded_json):
        entries = []
        # common variants:
        # 1) loaded_json is a dict containing 'annotations' or 'refs' list
        if isinstance(loaded_json, dict):
            if "annotations" in loaded_json and isinstance(loaded_json["annotations"], list):
                for a in loaded_json["annotations"]:
                    # each entry often contains 'image_id', 'ann_id', 'refex' or 'sentences'
                    image_id = a.get("image_id") or a.get("image")
                    ann_id = a.get("ann_id") or a.get("annotation_id") or a.get("annid")
                    ref = None
                    if "ref" in a:
                        ref = a["ref"]
                    elif "sentences" in a and isinstance(a["sentences"], list) and len(a["sentences"])>0:
                        # 'sentences' may be list of dicts with 'sent' key
                        s = a["sentences"][0]
                        ref = s.get("sent") if isinstance(s, dict) else s
                    if image_id is None or ann_id is None or ref is None:
                        continue
                    entries.append({"image_id": image_id, "ann_id": ann_id, "ref": ref})
            elif "refs" in loaded_json:
                for a in loaded_json["refs"]:
                    entries.append({"image_id": a["image_id"], "ann_id": a["ann_id"], "ref": a["ref"]})
            else:
                # might be a mapping: {ref_id: {...}}
                for v in loaded_json.values():
                    if isinstance(v, dict) and "image_id" in v and "ann_id" in v and "ref" in v:
                        entries.append({"image_id": v["image_id"], "ann_id": v["ann_id"], "ref": v["ref"]})
        elif isinstance(loaded_json, list):
            for a in loaded_json:
                # element formats vary; try to be flexible
                image_id = a.get("image_id") or a.get("image")
                ann_id = a.get("ann_id") or a.get("annid") or a.get("annotation_id") or a.get("annIdx")
                ref = a.get("ref") or (a.get("sentences")[0].get("sent") if a.get("sentences") else None)
                if image_id is None or ann_id is None or ref is None:
                    continue
                entries.append({"image_id": image_id, "ann_id": ann_id, "ref": ref})
        else:
            raise RuntimeError("Unsupported RefCOCO JSON structure. Inspect the file.")

        return entries

    def __len__(self):
        return len(self.entries)

    def _load_image(self, image_info):
        # image_info is a COCO image dict with 'file_name'
        img_path = os.path.join(self.images_dir, image_info["file_name"])
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            im = im.resize(self.image_size)
            arr = np.array(im, dtype=np.float32)
        return arr, image_info["width"], image_info["height"]

    def _get_gt_box_normalized(self, ann):
        # ann bbox format in COCO: [x, y, width, height] absolute pixels
        x, y, w, h = ann["bbox"]
        return np.array([x, y, x + w, y + h], dtype=np.float32)  # return absolute; normalization done after reading image size

    def get_tf_dataset(self, batch_size=8, shuffle=True, buffer_size=2048):
        """
        Returns a tf.data.Dataset that yields:
            image_tensor: [H, W, 3] float32 preprocessed by ResNet preprocess_input
            token_ids: integer tensor [T] (padded later)
            gt_box: normalized [x1,y1,x2,y2] in [0,1]
        """
        # Prepare lists of samples (we build indices because COCO API loads lazily)
        sample_list = []
        for e in self.entries:
            sample_list.append(e)

        def gen():
            for e in sample_list:
                image_id = e["image_id"]
                ann_id = e["ann_id"]
                ref_text = e["ref"]

                img_info = self.coco.loadImgs(image_id)[0]
                img_arr, orig_w, orig_h = self._load_image(img_info)
                ann = self.coco.loadAnns(ann_id)[0]
                x, y, w, h = ann["bbox"]
                # convert to normalized x1,y1,x2,y2 relative to original image dimensions BEFORE resize
                x1n = x / img_info["width"]
                y1n = y / img_info["height"]
                x2n = (x + w) / img_info["width"]
                y2n = (y + h) / img_info["height"]
                gt_box_norm = np.array([x1n, y1n, x2n, y2n], dtype=np.float32)

                # preprocess image using ResNet preprocessing
                img_pre = tf.keras.applications.resnet.preprocess_input(img_arr)
                # tokenization - if vectorizer is given, use it to convert to integers
                if self.text_vectorizer is not None:
                    toks = self.text_vectorizer([ref_text]).numpy()[0]  # vectorizer returns 2D
                else:
                    # naive whitespace hashing fallback: map tokens to small ints
                    toks = np.array([hash(w) % 10000 for w in ref_text.lower().split()], dtype=np.int32)

                yield img_pre.astype(np.float32), toks.astype(np.int32), gt_box_norm

        # determine output shapes: tokens variable length -> ragged or pad later
        output_signature = (
            tf.TensorSpec(shape=(self.image_size[0], self.image_size[1], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(4,), dtype=tf.float32),
        )
        ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
        if shuffle:
            ds = ds.shuffle(buffer_size)
        # pad tokens to max in batch with ragged/padded_batch
        ds = ds.padded_batch(batch_size, padded_shapes=([self.image_size[0], self.image_size[1], 3], [None], [4]))
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds
