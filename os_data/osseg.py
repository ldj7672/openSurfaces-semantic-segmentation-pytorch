
import os
from csv import DictReader

import numpy
from scipy.misc import imread


class AbstractSegmentation:
    def all_names(self, category, j):
        raise NotImplementedError

    def size(self, split=None):
        return 0

    def filename(self, i):
        raise NotImplementedError

    def metadata(self, i):
        return self.filename(i)

    @classmethod
    def resolve_segmentation(cls, m):
        return {}

    def name(self, category, i):
        """
        Default implementation for segmentation_data,
        utilizing all_names.
        """
        all_names = self.all_names(category, i)
        return all_names[0] if len(all_names) else ''

    def segmentation_data(self, category, i, c=0, full=False):
        """
        Default implementation for segmentation_data,
        utilizing metadata and resolve_segmentation.
        """
        segs, segs_shape = self.resolve_segmentation(
            self.metadata(i), categories=[category])
        if category not in segs:
            return 0
        data = numpy.asarray(segs[category])
        if not full and len(data.shape) >= 3:
            return data[0]
        return data

    def image_data(self, i):
        return imread(self.filename(i), mode='RGB')

class OpenSurfaceSegmentation(AbstractSegmentation):
    def __init__(self, directory):
        directory = os.path.expanduser(directory)
        self.directory = directory
        # Process material labels: open label-substance-colors.csv
        subst_name_map = {}
        with open(os.path.join('./dataset', 'label-substance-colors.csv')) as f:
            for row in DictReader(f):
                subst_name_map[row['substance_name']] = int(row['red_color'])
        # NOTE: substance names should be normalized. 
        self.substance_names = ['-'] * (1 + max(subst_name_map.values()))
        for k, v in list(subst_name_map.items()):
            self.substance_names[v] = k
        # Now load the metadata about images from photos.csv
        with open(os.path.join('./dataset/', 'photos.csv')) as f:
            self.image_meta = list(DictReader(f))
            #scenes = set(row['scene_category_name'] for row in self.image_meta)

    # def all_names(self, category, j):
    #     if j == 0:
    #         return []
    #     if category == 'material':
    #         return [norm_name(n) for n in self.substance_names[j].split('/')]
    #     return []

    # def size(self):
    #     """Returns the number of images in this dataset."""
    #     return len(self.image_meta)

    def filename(self, i):
        """Returns the filename for the nth dataset image."""
        photo_id = int(self.image_meta[i]['photo_id'])
        return os.path.join(self.directory, 'photos', '%d.jpg' % photo_id)

    def metadata(self, i):
        """Returns an object that can be used to create all segmentations."""
        #row = self.image_meta[i]
        return dict(
            filename=self.filename(i),
            seg_filename=self.seg_filename(i))

    def seg_filename(self, i):
        """ Return the seg filename for the nth dataset seg img. """
        photo_id = int(self.image_meta[i]['photo_id'])
        return os.path.join(self.directory, 'photos-labels', '%d.png' % photo_id)

    @classmethod
    def resolve_segmentation(cls, m):
        result = {}
        labels = imread(m['seg_filename'])
        result = labels[:, :, 0]
        arrs = [a for a in list(result) if len(numpy.shape(a)) >= 2]
        shape = arrs[0].shape[-2:] if arrs else (1, 1)
        return result, shape



# def norm_name(s):
#     return s.replace(' - ', '-').replace('/', '-').strip().lower()

