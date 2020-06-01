"""
Faster R-CNN
Building dataset from csv file
The csv file just like
========================================================
image_path    annotation
path/to/img1,0 12 34 20 45;0 23 45 90 100
path/to/img2,0 20 14 59 109
path/to/img3,
path/to/img4,0 19 20 45 29;0 11 34 56 90;0 23 45 34 90
========================================================
0 represents that the instance is a rectangle box; each line a image

copyright (c) Harbin Medical University
Licensed under MIT license
written by Lei Cao
"""
import torch
from PIL import Image
import pandas as pd


class TCTDataset(object):
    """
    Build the TCTDataset from csv file
    """

    def __init__(self, datatype="train", transform=None, labels_dict={}):
        """
        datatype: "train", "val", or "test"
        transform: whether to do transform on image
        labels_dict: a little diff for train and val/test,\
                for train: just like this\
                {"path/to/img1": "0 12 34 23 45;0 123 345 902 454",
                 "path/to/img2": "0 23 45 46 90"}
                for val/test: just like this\
                {"path/to/img5": "",
                 "path/to/img6": "0 441 525 582 636"}
        """
        self.datatype = datatype
        self.transform = transform
        self.labels_dict = labels_dict
        self.image_files_list = list(self.labels_dict.keys())
        self.annotations = [labels_dict[i] for i in self.image_files_list]

    def __getitem__(self, idx):
        # load image
        img_path = self.image_files_list[idx]
        img = Image.open(img_path).convert("RGB")
        annotation = self.labels_dict[img_path]

        if self.datatype == "train":
            boxes = []
            if type(annotation) == str:
                annotation_list = annotation.split(";")
                for anno in annotation_list:
                    x = []
                    y = []
                    anno = anno[2:]  # one box coord str
                    anno = anno.split(" ")
                    for i in range(len(anno)):
                        if i % 2 == 0:
                            x.append(float(anno[i]))
                        else:
                            y.append(float(anno[i]))

                    xmin = min(x)
                    xmax = max(x)
                    ymin = min(y)
                    ymax = max(y)
                    boxes.append([xmin, ymin, xmax, ymax])

            # convert anything to torch tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # there is only one class
            labels = torch.ones((len(boxes),), dtype=torch.int64)
            # image name
            image_name = torch.tensor(
                [ord(i) for i in list(img_path)],
                dtype=torch.int8
            )
            # make annos on the image into target
            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["name"] = image_name

            if self.transform is not None:
                img, target = self.transform(img, target)

            return img, target

        if self.datatype in ["val", "test"]:
            if self.labels_dict[img_path] == "":
                label = 0
            else:
                label = 1
                
            if self.transform is not None:
                img = self.transform(img)

            return img, label, img_path

    def __len__(self):
        return len(self.image_files_list)


def get_dataset(label_csv_file, datatype, transform):
    """
    Prepare dataset
    label_csv_file: csv file containing image path and annotation
    datatype: "train", "val", or "test"
    transform: transform done on images and targets
    """
    labels_df = pd.read_csv(label_csv_file, na_filter=False)
    if datatype == "train":
        labels_df = labels_df.loc[
            labels_df["annotation"].astype(bool)
        ].reset_index(drop=True)
    img_class_dict = dict(zip(labels_df.image_path, labels_df.annotation))
    dataset = TCTDataset(datatype=datatype, transform=transform,
                         labels_dict=img_class_dict)
    return dataset


if __name__ == "__main__":
    train_csv = "../statistic_description/tmp/train.csv"
    val_csv = "../statistic_description/tmp/val.csv"
    dataset_tr = get_dataset(train_csv,
                             datatype="train", transform=None)
    dataset_val = get_dataset(val_csv,
                              datatype="val", transform=None)

    import ipdb;ipdb.set_trace()

