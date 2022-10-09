# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import os.path as osp
import re
import warnings

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class PatentTest(ImageDataset):
    """Market1501.             

    Reference:
        Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: `<http://www.liangzheng.org/Project/project_reid.html>`_

    Dataset statistics:
        - identities: 1501 (+1 for background).
        - images: 12936 (train) + 3368 (query) + 15913 (gallery).
    """
    _junk_pids = [0, -1]
    dataset_url = 'http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip'
    dataset_name = "patent_data"
    dataset_test_dir = "codalab_test_set"
    def __init__(self, root='datasets', **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))
        self.root = root
        self.test_dir  = osp.join(self.root, self.dataset_test_dir)
        # self.image_dir = osp.join(self.root,"train_data","patent_data")
        # self.train_path = osp.join(self.root, "train_data","train_patent_trn.txt")
        self.test_db = osp.join(self.test_dir,"lists/test_database.txt") 
        self.test_query = osp.join(self.test_dir,"lists/test_queries.txt")
        # train = lambda: self.process_dir(self.image_dir, self.train_path, is_train=True)
        train = []
        query = lambda: self.process_test_dir(self.test_dir,self.test_query,  is_query=True)
        gallery = lambda: self.process_test_dir(self.test_dir,self.test_db, is_query=False) 
        super(PatentTest, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, path_file, is_train=False, is_query=False):
        with open(path_file, 'r') as f:
            gt = f.readlines()
        data = []
        for img in gt:
            img_path, pid = img.strip().split(" ")
            if is_train:
                data.append((osp.join(dir_path, img_path), self.dataset_name+"_"+str(pid), 1))
            elif is_query:      
                data.append((osp.join(dir_path, img_path),  int(pid), 1))
            else:
                data.append((osp.join(dir_path, img_path),  int(pid), -1))
                
        # img_paths = [osp.join(dir_path, img.strip().split(" ")) for img in gt]
        # glob.glob(osp.join(dir_path, '*.jpg'))
        # pattern = re.compile(r'([-\d]+)_c(\d)')
        # for img_path in img_paths:
        #     pid, camid = map(int, pattern.search(img_path).groups())
        #     if pid == -1:
        #         continue  # junk images are just ignored
        #     assert 0 <= pid <= 1501  # pid == 0 means background
        #     assert 1 <= camid <= 6
        #     camid -= 1  # index starts from 0
        #     if is_train:
        #         pid = self.dataset_name + "_" + str(pid)
        #         camid = self.dataset_name + "_" + str(camid)
        #     data.append((img_path, pid, camid))

        return data


    def process_test_dir(self, dir_path, path_file, is_train=False, is_query=False):
        with open(path_file, 'r') as f:
            gt = f.readlines()
        data = []
        for img_path in gt:
            img_path = img_path.strip()
            if is_query:      
                data.append((osp.join(dir_path, img_path),  0, 1))
            else:
                data.append((osp.join(dir_path, img_path),  0, -1))
                
        # img_paths = [osp.join(dir_path, img.strip().split(" ")) for img in gt]
        # glob.glob(osp.join(dir_path, '*.jpg'))
        # pattern = re.compile(r'([-\d]+)_c(\d)')
        # for img_path in img_paths:
        #     pid, camid = map(int, pattern.search(img_path).groups())
        #     if pid == -1:
        #         continue  # junk images are just ignored
        #     assert 0 <= pid <= 1501  # pid == 0 means background
        #     assert 1 <= camid <= 6
        #     camid -= 1  # index starts from 0
        #     if is_train:
        #         pid = self.dataset_name + "_" + str(pid)
        #         camid = self.dataset_name + "_" + str(camid)
        #     data.append((img_path, pid, camid))

        return data