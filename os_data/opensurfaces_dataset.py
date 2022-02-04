
import csv
import json
import os
from collections import namedtuple

import numpy
from . import osseg

class OpenSurfacesDataset:

    def __init__(self):


        dataset_root = "./dataset"

        self.dataset = osseg.OpenSurfaceSegmentation(directory=os.path.join(dataset_root, "opensurfaces"))

        self.record_list = {"train": [], "validation": []}
        self.record_list['train'] = get_records(os.path.join(dataset_root, 'os_train.json'))
        self.record_list['validation'] = get_records(os.path.join(dataset_root, 'os_val.json'))
        
        self.assignments = {}
        for l in restore_csv(os.path.join(dataset_root, 'label_assignment.csv')):
            self.assignments[(int(l.raw_label))] = int(l.new_label) 
        index_max = len(self.assignments)-1
        self.index_mapping = numpy.zeros(index_max + 1, dtype=numpy.int16)
        for (old_index), new_index in list(self.assignments.items()):
            self.index_mapping[old_index] = new_index


    def resolve_record(self, record):
       
        ds = self.dataset
        md = ds.metadata(record["file_index"])
        full_seg, shape = ds.resolve_segmentation(md)

        img_info = md['filename'].split('\\')[-1]
        img = ds.image_data(record["file_index"])
        seg_material = numpy.zeros((img.shape[0], img.shape[1]), dtype=numpy.uint8)
        
        seg_material = self.index_mapping[full_seg] 
        seg_material = numpy.asarray(seg_material, dtype=numpy.uint8)

        data = {
            "img_data": img,
            "seg_label": seg_material,
            "info" : img_info
        }

        return data


def get_records(path):
    with open(path) as f:
        filelist_json = f.readlines()
    records = [json.loads(x) for x in filelist_json]
    return records


def restore_csv(csv_path):
    with open(csv_path) as f:
        f_csv = csv.reader(f)
        headings = next(f_csv)
        Row = namedtuple('Row', headings)
        lines = [Row(*r) for r in f_csv]
    return lines


os_dataset = OpenSurfacesDataset()


