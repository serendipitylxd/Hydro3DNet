import copy
import pickle
import os

import numpy as np

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, common_utils
from ..dataset import DatasetTemplate
from datetime import datetime

class TROUTDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, mode=None ):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        # 新增内部变量存储 mode
        self._mode = mode
        # 根据 mode 或 training 状态确定 split
        if self._mode is not None:
            self.split = self.dataset_cfg.DATA_SPLIT[self._mode]
        else:
            self.split = self.dataset_cfg.DATA_SPLIT['train' if self.training else 'test']

        split_dir = os.path.join(self.root_path, 'ImageSets', (self.split + '.txt'))
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if os.path.exists(split_dir) else None

        self.TROUT_infos = []
        self.include_data(self.mode)
        self.map_class_to_kitti = self.dataset_cfg.MAP_CLASS_TO_KITTI

        # 加载额外信息（水位、时间戳、时间段）
        self.additional_info_dict = {}
        self.load_additional_info()

    @property
    def mode(self):
        return self._mode if self._mode is not None else ('train' if self.training else 'test')

    def load_additional_info(self):
        """从add_info文件加载每帧的额外信息"""
        if self.split == 'test':
            add_info_path = self.root_path / 'add_info_testing.txt'
        else:
            add_info_path = self.root_path / 'add_info_training.txt'

        if not add_info_path.exists():
            self.logger.warning(f"Additional info file {add_info_path} not found!")
            return

        with open(add_info_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                frame_id = parts[0]
                self.additional_info_dict[frame_id] = {
                    'timestamp': parts[1],
                    'water_level': float(parts[2]),
                    'time_period': int(parts[3])
                }

    def include_data(self, mode):
        self.logger.info('Loading TROUT dataset.')
        TROUT_infos = []
        print('include_data mode %s' % mode)
        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            print('include_data info_path %s' % info_path)
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                TROUT_infos.extend(infos)

        self.TROUT_infos.extend(TROUT_infos)
        self.logger.info('Total samples for TROUT dataset: %d' % (len(TROUT_infos)))

    def get_label(self, idx):
        if self.split == 'test':  # Check whether it is in test mode
            label_file = self.root_path / 'labels_test' / ('%s.txt' % idx)  # Use the point_test folde
        else:
            label_file = self.root_path / 'labels' / ('%s.txt' % idx)  # Other modes use the points folder

        assert label_file.exists(), f"Label file {label_file} not found!"

        with open(label_file, 'r') as f:
            lines = f.readlines()

        # [N, 8]: (x y z dx dy dz heading_angle category_id)
        gt_boxes = []
        gt_names = []
        for line in lines:
            line_list = line.strip().split(' ')
            gt_boxes.append(line_list[:7])
            gt_names.append(line_list[7])

        return np.array(gt_boxes, dtype=np.float32), np.array(gt_names)

    def get_lidar(self, idx):
        if self.split == 'test':  # Check whether it is in test mode
            lidar_file = self.root_path / 'points_test' / ('%s.bin' % idx)  # Use the point_test folder
        else:
            lidar_file = self.root_path / 'points' / ('%s.bin' % idx)  # Other modes use the points folder

        assert lidar_file.exists()  # Check whether the file exists

        point_features = np.fromfile(lidar_file,
                                     dtype=np.float32)  # Read point cloud data using np.fromfile(), each point contains 4

        point_features = point_features.reshape(-1,
                                                4)  # Reshape to (num_points, 4), where each point contains (x, y, z, intensity)

        return point_features

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training,
            root_path=self.root_path, logger=self.logger
        )
        self.split = split

        print("set_split", split)
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.sample_id_list) * self.total_epochs

        return len(self.TROUT_infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.TROUT_infos)

        info = copy.deepcopy(self.TROUT_infos[index])
        sample_idx = info['point_cloud']['lidar_idx']
        points = self.get_lidar(sample_idx)
        input_dict = {
            'frame_id': self.sample_id_list[index],
            'points': points
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            gt_names = annos['name']
            gt_boxes_lidar = annos['gt_boxes_lidar']
            add_info = info['add_info']
            water_level_arr = add_info['water_level']
            water_level = water_level_arr[0]
            #timestamp = add_info['timestamp']
            timestamp_str = add_info['timestamp'][0]  # 提取成纯 Python str
            dt = datetime.strptime(timestamp_str, '%Y_%m_%d_%H_%M_%S_%f')
            timestamp = np.array([dt.timestamp()], dtype=np.float32)  # float32!
            time_period = add_info['time_period']

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar,
                'water_level': water_level,
                'timestamp': timestamp,
                'time_period': time_period
            })

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict


    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.TROUT_infos[0].keys():
            return 'No ground-truth boxes for evaluation', {}

        def trout_eval(eval_det_annos, eval_gt_annos, map_name_to_kitti):
            from ..trout.trout_object_eval_python import eval as trout_eval
            from ..kitti import kitti_utils

            kitti_utils.transform_annotations_to_kitti_format(eval_det_annos, map_name_to_kitti=map_name_to_kitti)
            kitti_utils.transform_annotations_to_kitti_format(
                eval_gt_annos, map_name_to_kitti=map_name_to_kitti,
                info_with_fakelidar=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            kitti_class_names = [map_name_to_kitti[x] for x in class_names]
            ap_result_str, ap_dict = trout_eval.get_official_eval_result(
                gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names
            )
            return ap_result_str, ap_dict

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.TROUT_infos]

        if kwargs['eval_metric'] == 'trout':
            ap_result_str, ap_dict = trout_eval(eval_det_annos, eval_gt_annos, self.map_class_to_kitti)
        else:
            raise NotImplementedError

        #print('self.split:', self.split )

        return ap_result_str, ap_dict

    def get_infos(self, class_names, num_workers=4, has_label=True, sample_id_list=None, num_features=4):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': num_features, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            # Obtain data from the preloaded dictionary of additional information
            additional_info = self.additional_info_dict.get(sample_idx, {})

            if has_label:
                annotations = {}
                add_info = {}
                gt_boxes_lidar, name = self.get_label(sample_idx)

                # Obtain water level from additional information (Ensure type conversion)
                water_level = np.array(
                    [additional_info.get('water_level', 0.0)],
                    dtype=np.float32
                )
                timestamp = np.array(
                    [additional_info.get('timestamp', '')],
                    dtype=np.str_
                )
                time_period = np.array(
                    [additional_info.get('time_period', 0)],
                    dtype = np.int32

                )

                annotations.update({
                    'name': name,
                    'gt_boxes_lidar': gt_boxes_lidar[:, :7],
                })
                add_info.update({
                    'water_level': water_level,  # Add water level information
                    'timestamp': timestamp,
                    'time_period': time_period
                })
                info['annos'] = annotations
                info['add_info'] = add_info

            return info


        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list

        # create a thread pool to improve the velocity
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('TROUT_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_lidar(sample_idx)
            annos = info['annos']
            names = annos['name']
            gt_boxes = annos['gt_boxes_lidar']



            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]

        # Output the num of all classes in database
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    @staticmethod
    def create_label_file_with_name_and_box(class_names, gt_names, gt_boxes, save_label_path):
        with open(save_label_path, 'w') as f:
            for idx in range(gt_boxes.shape[0]):
                boxes = gt_boxes[idx]
                name = gt_names[idx]
                if name not in class_names:
                    continue
                line = "{x} {y} {z} {l} {w} {h} {angle} {name}\n".format(
                    x=boxes[0], y=boxes[1], z=(boxes[2]), l=boxes[3],
                    w=boxes[4], h=boxes[5], angle=boxes[6], name=name
                )
                f.write(line)


def create_TROUT_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = TROUTDataset(
        dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path,
        training=False, logger=common_utils.create_logger()
    )
    train_split, val_split, test_split = 'train', 'val', 'test'  # Add test
    num_features = len(dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list)

    train_filename = save_path / ('TROUT_infos_waterlevel_%s.pkl' % train_split)
    val_filename = save_path / ('TROUT_infos_waterlevel_%s.pkl' % val_split)
    test_filename = save_path / ('TROUT_infos_waterlevel_%s.pkl' % test_split)  # Add the file name of test

    print('------------------------Start to generate data infos------------------------')

    # Process the train dataset
    dataset.set_split(train_split)
    TROUT_infos_train = dataset.get_infos(
        class_names, num_workers=workers, has_label=True, num_features=num_features
    )
    with open(train_filename, 'wb') as f:
        pickle.dump(TROUT_infos_train, f)
    print('TROUT info train file is saved to %s' % train_filename)

    # Handle val datasets
    dataset.set_split(val_split)
    TROUT_infos_val = dataset.get_infos(
        class_names, num_workers=workers, has_label=True, num_features=num_features
    )
    with open(val_filename, 'wb') as f:
        pickle.dump(TROUT_infos_val, f)
    print('TROUT info val file is saved to %s' % val_filename)

    # Handle test datasets
    dataset.set_split(test_split)
    TROUT_infos_test = dataset.get_infos(
        class_names, num_workers=workers, has_label=True, num_features=num_features
    )
    with open(test_filename, 'wb') as f:
        pickle.dump(TROUT_infos_test, f)
    print('TROUT info test file is saved to %s' % test_filename)

    print('------------------------Start create groundtruth database for data augmentation------------------------')

    # Create the groundtruth database just for the train set

    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)
    print('------------------------Data preparation done------------------------')


if __name__ == '__main__':
    import sys

    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_TROUT_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict

        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        # Modified based on trout's data set
        create_TROUT_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Building', 'Fully_loaded_cargo_ship', 'Fully_loaded_container_ship', 'Lock_gate', 'Tree',
                         'Unladen_cargo_ship'],
            data_path=ROOT_DIR / 'data' / 'TROUT',
            save_path=ROOT_DIR / 'data' / 'TROUT',
        )
