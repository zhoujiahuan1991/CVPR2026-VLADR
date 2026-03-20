from __future__ import division, print_function, absolute_import
import glob
import numpy as np
import os.path as osp
import zipfile
import os
import copy
from lreid_dataset.incremental_datasets import IncrementalPersonReIDSamples
from reid.utils.serialization import read_json, write_json


class IncrementalSamples4cuhk01(IncrementalPersonReIDSamples):

    dataset_dir = 'CUHK01'
    def __init__(self, datasets_root, relabel=True, combineall=False, split_id = 0):
        self.root = datasets_root
        self.relabel = relabel
        self.combineall = combineall
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        self.zip_path = osp.join(self.dataset_dir, 'CUHK01.zip')
        self.campus_dir = osp.join(self.dataset_dir, 'campus') # CUHK01数据集里面直接包含/campus
        self.split_path = osp.join(self.dataset_dir, 'splits.json')

        self.extract_file()
        self.prepare_split()
        splits = read_json(self.split_path)
        if split_id >= len(splits):
            raise ValueError(
                'split_id exceeds range, received {}, but expected between 0 and {}'
                    .format(split_id,
                            len(splits) - 1)
            )
        split = splits[split_id]

        train = split['train']
        query = split['query']
        gallery = split['gallery']

        # train = [tuple(item + ['cuhk01'] + [item[1]]) for item in train]
        # query = [tuple(item + ['cuhk01'] + [item[1]]) for item in query]
        # gallery = [tuple(item + ['cuhk01'] + [item[1]]) for item in gallery]
        # train = [tuple(item + ['10']) for item in train]
        # query = [tuple(item + ['10']) for item in query]
        # gallery = [tuple(item + ['10']) for item in gallery]

        
        if len(train[0])==3:
            train = [tuple(item + ['10']) for item in train]
        else:
            train = [tuple(item ) for item in train]
            
        # print(train)
        # print(gallery)
        # exit(0)
        query = [tuple(item ) for item in query]
        gallery = [tuple(item ) for item in gallery]
        
        self.train, self.query, self.gallery = train, query, gallery
        self._show_info(self.train, self.query, self.gallery)

    def extract_file(self):
        if not osp.exists(self.campus_dir):
            print('Extracting files')
            zip_ref = zipfile.ZipFile(self.zip_path, 'r')
            zip_ref.extractall(self.dataset_dir)
            zip_ref.close()

    def prepare_split(self):
        """
        Image name format: 0001001.png, where first four digits represent identity
        and last four digits represent cameras. Camera 1&2 are considered the same
        view and camera 3&4 are considered the same view.
        """
        # 如果切分文件已经存在，就不重复创建
        if not osp.exists(self.split_path):
            print('Creating 10 random splits of train ids and test ids')

            # 收集所有图像路径
            # glob.glob(...) 是进行文件匹配，根据通配符模式*.png找到所有以.png结尾的文件
            # 返回匹配到的所有PNG文件的完整路径列表
            img_paths = sorted(glob.glob(osp.join(self.campus_dir, '*.png')))

            img_list = []
            pid_container = set() # 因为去重
            for img_path in img_paths:
                img_name = osp.basename(img_path) # 从完整路径中提取文件名
                pid = int(img_name[:4]) - 1 # 转换成从0开始的索引
                camid = (int(img_name[4:7]) - 1) // 2  # result is either 0 or 1
                img_list.append((img_path, pid, camid)) 
                pid_container.add(pid)

            num_pids = len(pid_container)
            num_train_pids = num_pids // 2 #  取一半ID数作train

            splits = []
            for _ in range(10): # 进行10次不同的数据划分
                order = np.arange(num_pids) # 所有人员ID顺序
                # [0,1,2,3,4,5]
                np.random.shuffle(order) 
                # [0,3,1,5,4,2]
                train_idxs = order[:num_train_pids] # 打乱后前N个作为训练集ID
                # [0,3,1]
                train_idxs = np.sort(train_idxs) # 打乱后的升序排列
                # [0,1,3]
                idx2label = { 
                    idx: label # 键值对：原始ID-> 新标签
                    # 因为训练分类器时要求所有的pid从0开始，但在test阶段不用
                    for label, idx in enumerate(train_idxs) # 遍历训练集人员ID并且编号
                }# [0:0, 1:1, 3:2]

                train, test_a, test_b = [], [], []
                for img_path, pid, camid in img_list:
                    if pid in train_idxs: 
                        train.append((img_path, idx2label[pid], camid,10))
                        # 因为在train阶段将pid当作分类标签输入网络，要求label从0开始且连续不调号，因此要做重映射
                    else:
                        if camid == 0:
                            test_a.append((img_path, pid, camid,10))
                        else:
                            test_b.append((img_path, pid, camid,10))

                # use cameraA as query and cameraB as gallery
                split = {
                    'train': train,
                    'query': test_a,
                    'gallery': test_b,
                    'num_train_pids': num_train_pids,
                    'num_query_pids': num_pids - num_train_pids,
                    'num_gallery_pids': num_pids - num_train_pids
                }
                splits.append(split)

                # use cameraB as query and cameraA as gallery
                split = {
                    'train': train,
                    'query': test_b,
                    'gallery': test_a,
                    'num_train_pids': num_train_pids,
                    'num_query_pids': num_pids - num_train_pids,
                    'num_gallery_pids': num_pids - num_train_pids
                }
                splits.append(split)

            print('Totally {} splits are created'.format(len(splits)))
            write_json(splits, self.split_path)
            print('Split file saved to {}'.format(self.split_path))


class CUHK01(IncrementalPersonReIDSamples):
    """CUHK01.

    Reference:
        Li et al. Human Reidentification with Transferred Metric Learning. ACCV 2012.

    URL: `<http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html>`_
    
    Dataset statistics:
        - identities: 971.
        - images: 3884.
        - cameras: 4.
    """
    dataset_dir = 'cuhk01'
    dataset_url = None

    def __init__(self, root='', split_id=0, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        # osp.expanduser() 展开用户目录符号，例如，~ -> /home/usename/
        # osp.abspath() 转换成绝对路径
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.download_dataset(self.dataset_dir, self.dataset_url)

        self.zip_path = osp.join(self.dataset_dir, 'CUHK01.zip')
        self.campus_dir = osp.join(self.dataset_dir, 'campus')
        self.split_path = osp.join(self.dataset_dir, 'splits.json')

        self.extract_file()

        required_files = [self.dataset_dir, self.campus_dir]
        self.check_before_run(required_files)

        self.prepare_split()
        splits = read_json(self.split_path) # self.split_path和/campus相同路径
        if split_id >= len(splits):
            raise ValueError(
                'split_id exceeds range, received {}, but expected between 0 and {}'
                .format(split_id,
                        len(splits) - 1)
            )
        split = splits[split_id]

        train = split['train']
        query = split['query']
        gallery = split['gallery']

        train = (tuple(item + ['cuhk01'] + [item[1]]) for item in train) 
        # 3元item扩展成5元item
        # (image_path, pid, camid) -> (image_path, pid, camid, 'cuhk01', pid)
        query = (tuple(item + ['cuhk01'] + [item[1]]) for item in query)
        gallery = (tuple(item + ['cuhk01'] + [item[1]]) for item in gallery)

        super(CUHK01, self).__init__(train, query, gallery, **kwargs)

    def extract_file(self): # 在未解压的时候才做这一步操作
        if not osp.exists(self.campus_dir):
            print('Extracting files')
            zip_ref = zipfile.ZipFile(self.zip_path, 'r')
            zip_ref.extractall(self.dataset_dir)
            zip_ref.close()

    def prepare_split(self):
        """
        Image name format: 0001001.png, where first four digits represent identity
        and last four digits represent cameras. Camera 1&2 are considered the same
        view and camera 3&4 are considered the same view.
        """
        if not osp.exists(self.split_path):
            print('Creating 10 random splits of train ids and test ids')
            img_paths = sorted(glob.glob(osp.join(self.campus_dir, '*.png')))
            img_list = []
            pid_container = set()
            for img_path in img_paths:
                img_name = osp.basename(img_path)
                pid = int(img_name[:4]) - 1
                camid = (int(img_name[4:7]) - 1) // 2 # result is either 0 or 1
                img_list.append((img_path, pid, camid))
                pid_container.add(pid)

            num_pids = len(pid_container)
            num_train_pids = num_pids // 2

            splits = []
            for _ in range(10):
                order = np.arange(num_pids)
                np.random.shuffle(order)
                train_idxs = order[:num_train_pids]
                train_idxs = np.sort(train_idxs)
                idx2label = {
                    idx: label
                    for label, idx in enumerate(train_idxs)
                }

                train, test_a, test_b = [], [], []
                for img_path, pid, camid in img_list:
                    if pid in train_idxs:
                        train.append((img_path, idx2label[pid], camid))
                    else:
                        if camid == 0:
                            test_a.append((img_path, pid, camid))
                        else:
                            test_b.append((img_path, pid, camid))

                # use cameraA as query and cameraB as gallery
                split = {
                    'train': train,
                    'query': test_a,
                    'gallery': test_b,
                    'num_train_pids': num_train_pids,
                    'num_query_pids': num_pids - num_train_pids,
                    'num_gallery_pids': num_pids - num_train_pids
                }
                splits.append(split)

                # use cameraB as query and cameraA as gallery
                split = {
                    'train': train,
                    'query': test_b,
                    'gallery': test_a,
                    'num_train_pids': num_train_pids,
                    'num_query_pids': num_pids - num_train_pids,
                    'num_gallery_pids': num_pids - num_train_pids
                }
                splits.append(split)

            print('Totally {} splits are created'.format(len(splits)))
            write_json(splits, self.split_path)
            print('Split file saved to {}'.format(self.split_path))
