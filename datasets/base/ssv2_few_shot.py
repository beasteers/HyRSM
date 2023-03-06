import os
import random
import torch
import torch.utils.data
import utils.logging as logging
import numpy as np

import time
import oss2 as oss
import traceback


from torchvision.transforms import Compose
import torchvision.transforms._transforms_video as transforms
import torch.nn.functional as F
from datasets.utils.transformations import (
    ColorJitter, CustomResizedCropVideo, 
    AutoResizedCropVideo,
    KineticsResizedCrop,
    KineticsResizedCropFewshot
)
from datasets.utils.create_splits import add_noaction
# from datasets.utils.shuffle import shuffle
# from datasets.utils.unfold import unfold

from datasets.base.base_dataset import BaseVideoDataset

import utils.bucket as bu
from utils.misc import call_recursive

from datasets.base.builder import DATASET_REGISTRY
from datasets.utils.random_erasing import RandomErasing

logger = logging.get_logger(__name__)


class Split_few_shot():
    """Contains video frame paths and ground truth labels for a single split (e.g. train videos). """
    def __init__(self, folder, split_dataset='train', dataset="Ssv2_few_shot", file_pattern=""):
        # self.args = args
        
        
        self.gt_a_list = []
        self.videos = []
        self.split_dataset = split_dataset
        self.dataset_name = dataset
        self.file_pattern = file_pattern or "{video_id}"

        for row in folder:
            cls_id, path = row.strip().split(',')
            self.add_vid(path, int(cls_id))
        # if dataset == 'Ssv2_few_shot':
        #     for class_folder in folder:
        #         self.add_vid_from_folder(class_folder, split_dataset, sep='/')
        # else:
        #     for class_folder in folder:
        #         self.add_vid_from_folder(class_folder, split_dataset, sep='//')

        logger.info("loaded {} videos from {} dataset: {} !".format(len(self.gt_a_list), split_dataset, dataset))

    # def add_vid_from_folder(self, class_folder,split_dataset, sep='/'):
    #     class_id_str, paths = class_folder.strip().split(sep, 1)
    #     class_id = int(class_id_str[len(split_dataset):]) # class_folders.index(class_folder)
    #     self.add_vid(paths, class_id)

    def add_vid(self, paths, gt_a):
        self.videos.append(paths)
        self.gt_a_list.append(gt_a)

    def get_rand_vid(self, label, idx=-1):
        match_idxs = [i for i, l in enumerate(self.gt_a_list) if label == l]
        i = np.random.choice(match_idxs) if idx == -1 else match_idxs[idx]
        return self.videos[i], i

    def get_single_video(self, index):
        return self.videos[index], self.gt_a_list[index]

    def get_num_videos_for_class(self, label):
        return sum(gt == label for gt in self.gt_a_list)

    def get_unique_classes(self):
        return list(set(self.gt_a_list))

    def __len__(self):
        return len(self.gt_a_list)

    def get_rand_video_info(self, label, idx, data_root_dir):
        paths, vid_id = self.get_rand_vid(label, idx) 
        return {
            "path": os.path.join(data_root_dir, self.file_pattern.format(video_id=paths)),
            # "supervised_label": class_,
        }, vid_id



@DATASET_REGISTRY.register()
class Ssv2_few_shot(BaseVideoDataset):
    def __init__(self, cfg, split):
        super(Ssv2_few_shot, self).__init__(cfg, split) 
        if self.split == "test" and self.cfg.PRETRAIN.ENABLE == False:
            self._pre_transformation_config_required = True
        # if hasattr(self.cfg.TRAIN, "DATASET_FEW"):
        #     self.dataset_name = self.cfg.TRAIN.DATASET_FEW
        self.split_dataset = split
    
    def _get_ssl_label(self, frames):
        pass
        
        
    def _get_dataset_list_name(self):
        """
            Returns:
                dataset_list_name (string)
        """

        name = "{}_few_shot.txt".format(   
            "train" if self.split == "train" else "test",
        )
        logger.info("Reading video list from file: {}".format(name))
        return name

    def _get_sample_info(self, index):
        """
            Input: 
                index (int): video index
            Returns:
                sample_info (dict): contains different informations to be used later
                    Things that must be included are:
                    "video_path" indicating the video's path w.r.t. index
                    "supervised_label" indicating the class of the video 
        """
        return {
            "path": os.path.join(self.data_root_dir, self._samples[index]["id"]+".mp4"),
            "supervised_label": self._samples[index]["label_idx"],
        }

    def _construct_dataset(self, cfg):

        if hasattr(self.cfg.TRAIN, "DATASET_FEW"):
            self.dataset_name = self.cfg.TRAIN.DATASET_FEW
        
        self._num_clips = 1

        self._samples = []
        self._spatial_temporal_index = []
        dataset_list_name = self._get_dataset_list_name()

        for retry in range(5):
            try:
                logger.info("Loading {} dataset list for split '{}'...".format(self.dataset_name, self.split))
                local_file = os.path.join(cfg.OUTPUT_DIR, dataset_list_name)
                local_file = self._get_object_to_file(os.path.join(self.anno_dir, dataset_list_name), local_file)
                logger.info(local_file)
                if local_file.endswith(".csv"):
                    import pandas
                    lines = pandas.read_csv(local_file).values.tolist()
                elif local_file.endswith(".json"):
                    import json
                    with open(local_file, "r") as f:
                        lines = json.load(f)
                else:
                    with open(local_file, 'r') as f:
                        lines = f.readlines()
                for line in lines:
                    if any(x in line for x in ['_0.']):
                        continue
                    for idx in range(self._num_clips):
                        self._samples.append(line.strip())
                        self._spatial_temporal_index.append(idx)
                self.split_few_shot = Split_few_shot(
                    lines, self.split, dataset=self.dataset_name, 
                    file_pattern=getattr(self.cfg.DATA, 'VIDEO_FILE_PATTERN', None) or "")
                logger.info("Dataset {} split {} loaded. Length {}.".format(self.dataset_name, self.split, len(self._samples)))
                break
            except:
                if retry<4:
                    continue
                else:
                    raise ValueError("Data list {} not found.".format(os.path.join(self.anno_dir, dataset_list_name)))

        if hasattr(self.cfg.TRAIN, "FEW_SHOT") and self.cfg.TRAIN.FEW_SHOT and self.split == "train":
            """ Sample number setting for training in few-shot settings: 
                During few shot training, the batch size could be larger than the size of the training samples.
                Therefore, the number of samples in the same sample is multiplied by 10 times, and the training schedule is reduced by 10 times. 
            """
            self._samples = self._samples * 10
            self._spatial_temporal_index = self._spatial_temporal_index * 10
            print("10 FOLD FEW SHOT SAMPLES")
            
        assert len(self._samples) != 0, "Empty sample list {}".format(os.path.join(self.anno_dir, dataset_list_name))


    def __getitem__(self, index):
        """
            Returns:
                frames (dict): {
                    "video": (tensor), 
                    "text_embedding" (optional): (tensor)
                }
                labels (dict): {
                    "supervised": (tensor),
                    "self-supervised" (optional): (...)
                }
        """
        # print(len(self), index, len(self._samples), len(self.split_few_shot))
        if self.cfg.TRAIN.META_BATCH:
            """returns dict of support and target images and labels for a meta training task"""
            #select classes to use for this task
            c = self.split_few_shot
            classes = c.get_unique_classes()
            K = self.split != "train" and getattr(self.cfg.TRAIN, "WAT_TEST", None) or self.cfg.TRAIN.WAY
            batch_classes = random.sample(classes, K)

            n_queries = self.cfg.TRAIN.QUERY_PER_CLASS if self.split == "train" else self.cfg.TRAIN.QUERY_PER_CLASS_TEST
            is_query = self.split_dataset=="train" and hasattr(self.cfg.AUGMENTATION, "SUPPORT_QUERY_DIFF_SUPPORT")
            retries = 5
            for retry in range(retries):
                support_set = []
                support_labels = []
                target_set = []
                target_labels = []
                real_support_labels = []
                real_target_labels = []
                bc = idxs = vid_id = None
                try:
                    for bl, bc in enumerate(batch_classes):
                        n_total = c.get_num_videos_for_class(bc)
                        idxs = random.sample([i for i in range(n_total)], self.cfg.TRAIN.SHOT + n_queries)

                        for idx in idxs[:self.cfg.TRAIN.SHOT]:
                            vid, vid_id = self.get_seq(bc, idx, is_query=is_query)
                            # print('support', vid_id)
                            support_set.append(vid)
                            support_labels.append(bl)
                            real_support_labels.append(bc)
                        
                        for idx in idxs[self.cfg.TRAIN.SHOT:]:
                            vid, vid_id = self.get_seq(bc, idx, is_query=is_query)
                            # print('target', vid_id)
                            target_set.append(vid)
                            target_labels.append(bl)
                            real_target_labels.append(bc)
                    break       
                except Exception as e:
                    logger.exception(e)
                    logger.warning("Error at META_BATCH decoding. {}/{}. Vid index: {}, Vid path: {}".format(
                        retry+1, retries, idxs, bc))
            else:
                return self.__getitem__(index - 1)
            
            s = list(zip(support_set, support_labels, real_support_labels))
            random.shuffle(s)
            support_set, support_labels, real_support_labels = tuple(zip(*s)) or ((),(),())
            
            t = list(zip(target_set, target_labels, real_target_labels))
            random.shuffle(t)
            target_set, target_labels, real_target_labels = tuple(zip(*t)) or ((),(),())
            
            support_set = torch.cat(support_set)  # [200, 3, 224, 224]

            target_set = torch.cat(target_set)    # [200, 3, 224, 224]
            support_labels = torch.FloatTensor(support_labels)
            target_labels = torch.FloatTensor(target_labels)
            real_target_labels = torch.FloatTensor(real_target_labels)  # shape: [25]
            real_support_labels = torch.FloatTensor(real_support_labels)
            # [45., 59., 45., 11., 39., 39., 39., 11., 11., 25., 25., 25., 59., 45., 11., 25., 59., 25., 45., 39., 45., 59., 39., 59., 11.]
            batch_classes = torch.FloatTensor(batch_classes) # [45., 11., 59., 25., 39.]

            # {
            #     # shape min max
            #     'support_set': 'torch.Size([200, 3, 224, 224]) -2.0 2.4444446563720703', 
            #     'support_labels': 'torch.Size([25]) 0.0 4.0', 
            #     'target_set': 'torch.Size([40, 3, 224, 224]) -2.0 2.4444446563720703', 
            #     'target_labels': 'torch.Size([5]) 0.0 4.0', 
            #     'r$al_target_labels': 'torch.Size([5]) 7.0 18.0', 
            #     'batch_class_list': 'torch.Size([5]) 7.0 18.0', 
            #     'real_support_labels': 'torch.Size([25]) 7.0 18.0'
            # }
            data = {
                "support_set": support_set, 
                "support_labels": support_labels, 
                "target_set": target_set, 
                "target_labels": target_labels, 
                "real_target_labels": real_target_labels, 
                "batch_class_list": batch_classes, 
                "real_support_labels": real_support_labels
            }
            # print(call_recursive(lambda x: f'{x.shape} {x.min()} {x.max()}' if isinstance(x, torch.Tensor) else x, data))
            # print("----")
            return data

        else:
            sample_info = self._get_sample_info(index)
            try:
                data, labels = self._get_seq(sample_info, index)
            except Exception as e:
                logger.info("Error at __getitem__. Vid index: {}, Vid path: {}".format(index, sample_info["path"]))
                return self.__getitem__(index + (-1 if index else 1))
            return data, labels, index, sample_info

    def get_seq(self, label, idx=-1, **kw):
        """Gets a single video sequence for a meta batch.  """
        sample_info, vid_id = self.split_few_shot.get_rand_video_info(label, idx, self.data_root_dir)
        data, labels = self._get_seq(sample_info, vid_id, **kw)
        return data, vid_id

    def _get_seq(self, sample_info, index, retries=3, is_query=False):
        """Gets a single video sequence for a meta batch.  """
        # retries = 5 if self.split == "train" else 10   # 1
        # retry loading data
        for retry in range(retries or 1):
            try:
                num_clips = getattr(self, 'num_clips_per_video', 1)
                data, file_to_remove, success = self.decode(sample_info, index, num_clips_per_video=num_clips)
                break
            except Exception as e:
                success = False
                logger.exception(e)
                logger.warning("Error at decoding. {}/{}. Vid index: {}, Vid path: {}".format(
                    retry+1, retries, index, sample_info["path"]))
        else: 
            raise Exception(f"no video after {retries} retries.")
        if not success:
            raise Exception("unable to read video")
        if not data or data["video"].numel() == 0:
            raise Exception("data[video].numel()=0")

        # zero-shot get word embedding
        if self.split in ["test"] and getattr(self.cfg.TEST, 'ZERO_SHOT', False):
            if not hasattr(self, "label_embd"):
                self.label_embd = self.word_embd(self.words_to_ids(self.label_names))
            data["text_embedding"] = self.label_embd

        # gpu stuff
        if self.gpu_transform:
            for k, v in data.items():
                data[k] = v.cuda(non_blocking=True)
        if self._pre_transformation_config_required:
            self._pre_transformation_config()

        labels = {}
        labels["supervised"] = sample_info["supervised_label"] if "supervised_label" in sample_info.keys() else {}
        if self.cfg.PRETRAIN.ENABLE:
            try:
                data, labels["self-supervised"] = self.ssl_generator(data, index)
            except Exception as e:
                traceback.print_exc()
                print("Error generating self-supervised label: {}, Vid path: {}, Vid shape: {}".format(
                    index, sample_info["path"], data["video"].shape
                ))
                raise 
        else:
            labels["self-supervised"] = {}
            if "flow" in data.keys() and "video" in data.keys():
                data = self.transform(data)
            elif "video" in data.keys():  # [8, 240, 428, 3] --> [3, 8, 224, 224]
                transform = self.transform_query if is_query else self.transform
                data["video"] = transform(data["video"])
                # C, T, H, W = 3, 16, 240, 320, RGB
        
        # get slowfast slow and fast windows of the video
        if  "Slowfast" in self.cfg.VIDEO.BACKBONE.META_ARCH and self.split not in ['extract_feat']:
            slow_idx = torch.linspace(0, data["video"].shape[1], data["video"].shape[1]//self.cfg.VIDEO.BACKBONE.SLOWFAST.ALPHA+1).long()[:-1]
            fast_frames = data["video"].clone()
            slow_frames = data["video"][:,slow_idx,:,:].clone()
            data["video"] = [slow_frames, fast_frames]
        bu.clear_tmp_file(file_to_remove)

        return data["video"].permute(1,0,2,3), labels
    

    def __len__(self):
        if hasattr(self.cfg.TRAIN, "META_BATCH") and self.split == 'train' and self.cfg.TRAIN.META_BATCH:
            return self.cfg.TRAIN.NUM_SAMPLES
        elif hasattr(self.cfg.TRAIN, "NUM_TEST_TASKS") and self.cfg.TRAIN.NUM_TEST_TASKS:
            return self.cfg.TRAIN.NUM_TEST_TASKS
        else:
            return len(self.split_few_shot)  # len(self._samples)
    

    def _config_transform(self):
        self.transform = None
        if self.split == 'train' and not self.cfg.PRETRAIN.ENABLE:
            std_transform_list_query = [
                transforms.ToTensorVideo(),
                transforms.RandomHorizontalFlipVideo(),
                KineticsResizedCropFewshot(
                    short_side_range = [self.cfg.DATA.TRAIN_JITTER_SCALES[0], self.cfg.DATA.TRAIN_JITTER_SCALES[1]],
                    crop_size = self.cfg.DATA.TRAIN_CROP_SIZE,
                ),]  
            if hasattr(self.cfg.AUGMENTATION, "RANDOM_FLIP") and self.cfg.AUGMENTATION.RANDOM_FLIP:
                std_transform_list = [
                transforms.ToTensorVideo(),
                transforms.RandomHorizontalFlipVideo(),
                KineticsResizedCropFewshot(
                    short_side_range = [self.cfg.DATA.TRAIN_JITTER_SCALES[0], self.cfg.DATA.TRAIN_JITTER_SCALES[1]],
                    crop_size = self.cfg.DATA.TRAIN_CROP_SIZE,
                ),]     # KineticsResizedCrop
            else:
                std_transform_list = [
                    transforms.ToTensorVideo(),
                    KineticsResizedCropFewshot(
                        short_side_range = [self.cfg.DATA.TRAIN_JITTER_SCALES[0], self.cfg.DATA.TRAIN_JITTER_SCALES[1]],
                        crop_size = self.cfg.DATA.TRAIN_CROP_SIZE,
                    ),
                    # transforms.RandomHorizontalFlipVideo()
                ]
            # Add color aug
            if self.cfg.AUGMENTATION.COLOR_AUG:
                std_transform_list.append(
                    ColorJitter(
                        brightness=self.cfg.AUGMENTATION.BRIGHTNESS,
                        contrast=self.cfg.AUGMENTATION.CONTRAST,
                        saturation=self.cfg.AUGMENTATION.SATURATION,
                        hue=self.cfg.AUGMENTATION.HUE,
                        grayscale=self.cfg.AUGMENTATION.GRAYSCALE,
                        consistent=self.cfg.AUGMENTATION.CONSISTENT,
                        shuffle=self.cfg.AUGMENTATION.SHUFFLE,
                        gray_first=self.cfg.AUGMENTATION.GRAY_FIRST,
                        is_split=self.cfg.AUGMENTATION.IS_SPLIT
                    ),
                )
            std_transform_list_query.append(
                    ColorJitter(
                        brightness=self.cfg.AUGMENTATION.BRIGHTNESS,
                        contrast=self.cfg.AUGMENTATION.CONTRAST,
                        saturation=self.cfg.AUGMENTATION.SATURATION,
                        hue=self.cfg.AUGMENTATION.HUE,
                        grayscale=self.cfg.AUGMENTATION.GRAYSCALE,
                        consistent=self.cfg.AUGMENTATION.CONSISTENT,
                        shuffle=self.cfg.AUGMENTATION.SHUFFLE,
                        gray_first=self.cfg.AUGMENTATION.GRAY_FIRST,
                        is_split=self.cfg.AUGMENTATION.IS_SPLIT
                    ),
                )
            std_transform_list_query += [
                transforms.NormalizeVideo(
                    mean=self.cfg.DATA.MEAN,
                    std=self.cfg.DATA.STD,
                    inplace=True
                ),
                RandomErasing(self.cfg)
                ]

            if hasattr(self.cfg.AUGMENTATION, "NO_RANDOM_ERASE") and self.cfg.AUGMENTATION.NO_RANDOM_ERASE:
                std_transform_list += [
                    transforms.NormalizeVideo(
                        mean=self.cfg.DATA.MEAN,
                        std=self.cfg.DATA.STD,
                        inplace=True
                    ),
                    # RandomErasing(self.cfg)
                ]
            else:
                std_transform_list += [
                transforms.NormalizeVideo(
                    mean=self.cfg.DATA.MEAN,
                    std=self.cfg.DATA.STD,
                    inplace=True
                ),
                RandomErasing(self.cfg)
                ]
            self.transform = Compose(std_transform_list)
            self.transform_query = Compose(std_transform_list_query)
        elif self.split == 'val' or self.split == 'test':
            idx = -1
            if hasattr(self.cfg.DATA, "TEST_CENTER_CROP"):
                idx = self.cfg.DATA.TEST_CENTER_CROP
            
            if isinstance(self.cfg.DATA.TEST_SCALE, list):
                self.resize_video = KineticsResizedCropFewshot(
                    short_side_range = [self.cfg.DATA.TEST_SCALE[0], self.cfg.DATA.TEST_SCALE[1]],
                    crop_size = self.cfg.DATA.TEST_CROP_SIZE,
                    num_spatial_crops = self.cfg.TEST.NUM_SPATIAL_CROPS,
                    idx = idx
                )   # KineticsResizedCrop
            else:
                self.resize_video = KineticsResizedCropFewshot(
                        short_side_range = [self.cfg.DATA.TEST_SCALE, self.cfg.DATA.TEST_SCALE],
                        crop_size = self.cfg.DATA.TEST_CROP_SIZE,
                        num_spatial_crops = self.cfg.TEST.NUM_SPATIAL_CROPS,
                        idx = idx
                    )   # KineticsResizedCrop
            std_transform_list = [
                transforms.ToTensorVideo(),
                self.resize_video,
                transforms.NormalizeVideo(
                    mean=self.cfg.DATA.MEAN,
                    std=self.cfg.DATA.STD,
                    inplace=True
                )
            ]
            self.transform = Compose(std_transform_list)


    def _pre_transformation_config(self):
        """
            Set transformation parameters if required.
        """
        self.resize_video.set_spatial_index(self.spatial_idx)

    def _custom_sampling(self, vid_length, clip_idx, num_clips, num_frames, interval=2, random_sample=True):
        return self._interval_based_sampling(vid_length, clip_idx, num_clips, num_frames, interval)

import cv2
import pandas as pd

class VideoFrameReader:
    def __init__(self, dataset_dir, video_id, start_frame, stop_frame, fps, format='frame_{:010d}.jpg', retry_tol=5):
        self.dataset_dir = dataset_dir
        self.video_id = video_id
        self.start_frame = start_frame
        self.stop_frame = stop_frame
        self.fps = fps
        self.fname_format = format
        self.retry_tol = retry_tol
        # f = os.path.join(self.dataset_dir, self.video_id)
        # if not os.path.exists(f):
        #     print('missing', f)
        #     input()
        # print(f, os.path.exists(f))

    def format_path(self, i):
        return os.path.join(self.dataset_dir, self.fname_format.format(video_id=self.video_id, i=i))

    def __len__(self):
        return self.stop_frame - self.start_frame

    def __getitem__(self, i, _tries=0):
        # load batch of video frames
        if isinstance(i, (list, tuple)):
            return self.get_batch(i)
        if isinstance(i, slice):
            return self.get_batch(range(len(self))[i])
        # load single video frame
        f = self.format_path(self.start_frame + i + 1)
        # print(f, os.path.isfile(f))
        if not os.path.isfile(f):
            # if not os.path.isdir(os.path.dirname(f)):
            #     raise RuntimeError(f"Missing video: {os.path.dirname(f)}")
            if i < len(self) and _tries < self.retry_tol:
                logger.warning(f"Missing video frame (try {_tries+1}/{self.retry_tol}): {f}")
                return self.__getitem__(i+1, _tries=_tries+1)
            raise RuntimeError(f"Missing video frame: {f}")
        im = cv2.imread(f)
        if im is None:
            raise RuntimeError(f"Error reading: {f}")
        # else:
        #     print(f, im.shape)
        return torch.tensor(im)

    def get_batch(self, xs):
        return torch.stack([self.__getitem__(ii) for ii in xs])

    def get_avg_fps(self):
        return self.fps


@DATASET_REGISTRY.register()
class Epic_few_shot_frames(Ssv2_few_shot):
    FRAME_FILE_FORMAT = '{video_id}/frame_{i:010d}.jpg'
    def __init__(self, cfg, split):
        super().__init__(cfg, split)
        self._load_dataset(cfg)

    def _get_dataset_list_name(self):
        name = super()._get_dataset_list_name()
        if getattr(self.cfg.DATA, 'UNSPECIFIED', False):
            name = '{}_noaction.{}'.format(*name.rsplit('.', 1))
        return name

    def _load_dataset(self, cfg):
        self.dataset_dir = cfg.DATA.DATA_ROOT_DIR

        # load all narrations
        narr_df = pd.concat([
            pd.read_csv(
                os.path.join(cfg.DATA.SUPP_ANNO_DIR, f'EPIC_100_{split}.csv')
            ).assign(split=split)
            for split in ['train', 'validation']
        ])
        
        # get video fps
        video_info = pd.read_csv(os.path.join(
            cfg.DATA.SUPP_ANNO_DIR, 'EPIC_100_video_info.csv')).set_index('video_id')
        
        narr_df = narr_df.groupby('video_id').apply(add_noaction, video_info=video_info)
        
        narr_df['fps'] = narr_df.video_id.apply(lambda x: video_info.fps[x])
        self.narr_df = narr_df.set_index('narration_id')

    def _read_video(self, video_path, index):
        narr_id = video_path.split('/')[-1].split('.')[0]
        row = self.narr_df.loc[narr_id]
        if isinstance(row, pd.DataFrame):
            print(row)
            row = row.iloc[0]
        vr = VideoFrameReader(
            self.dataset_dir, row.video_id, 
            row.start_frame, row.stop_frame, 
            fps=row.fps,
            format=self.FRAME_FILE_FORMAT,
        )
        return vr, [], True
    


@DATASET_REGISTRY.register()
class Egtea_few_shot_frames(Epic_few_shot_frames):
    FPS = 24
    # OP01-R01-PastaSalad_000014.jpg
    # FRAME_FILE_FORMAT = '{video_id}_{i:06d}.jpg'
    def __init__(self, cfg, split):
        super().__init__(cfg, split)
        self._load_dataset(cfg)

    def _load_dataset(self, cfg):
        self.dataset_dir = os.path.join(cfg.DATA.DATA_ROOT_DIR, "frames")

        # load all narrations
        df = pd.read_csv(os.path.join(
            cfg.DATA.DATA_ROOT_DIR,
            'raw_annotations/action_labels.csv'
        ), delimiter=r';\s*', engine='python').rename({
            'Clip Prefix (Unique)': 'narration_id',
            'Video Session': 'video_id',
            'Starting Time (ms)': 'start_frame',
            'Ending Time (ms)': 'stop_frame',
            # not necessary, but for completeness
            'Action Label': 'action',
            'Verb Label': 'verb',
            'Noun Label(s)': 'nouns',
        }, axis=1).set_index('narration_id')
        # print(df)
        # constant fps
        df['fps'] = self.FPS
        # convert start frame from ms to frames
        df['start_frame'] = (df['start_frame'] * self.FPS / 1000).astype(int)
        df['stop_frame'] = (df['stop_frame'] * self.FPS / 1000).astype(int)
        # nouns as csv
        df['nouns'] = df.nouns.apply(lambda x: x.split(','))
        self.narr_df = df


@DATASET_REGISTRY.register()
class Meccano_few_shot_frames(Epic_few_shot_frames):
    FPS = 12
    # OP01-R01-PastaSalad_000014.jpg
    FRAME_FILE_FORMAT = '{video_id}/{i:05d}.jpg'
    def __init__(self, cfg, split):
        super().__init__(cfg, split)
        self._load_dataset(cfg)

    def _load_dataset(self, cfg):
        self.dataset_dir = os.path.join(cfg.DATA.DATA_ROOT_DIR, "RGB_frames")

        kw=dict(dtype={'video_id': str})
        df = pd.concat([
            pd.read_csv(os.path.join(cfg.DATA.DATA_ROOT_DIR, 'MECCANO_train_actions.csv'), **kw).assign(split='Train'),
            pd.read_csv(os.path.join(cfg.DATA.DATA_ROOT_DIR, 'MECCANO_test_actions.csv'), **kw).assign(split='Test'),
            pd.read_csv(os.path.join(cfg.DATA.DATA_ROOT_DIR, 'MECCANO_val_actions.csv'), **kw).assign(split='Val'),
        ]).rename({'end_frame': 'stop_frame'}, axis=1)
        # convert existing columns
        df['start_frame'] = df.start_frame.apply(lambda x: int(x.split('.')[0]))
        df['stop_frame'] = df.stop_frame.apply(lambda x: int(x.split('.')[0]))
        df['narration_id'] = df.apply(lambda x: f'{x.video_id}-{x.start_frame}-{x.stop_frame}', axis=1)
        df['video_id'] = df.apply(lambda x: f'{x.split}/{x.video_id}', axis=1)
        # convert action to verb
        df['verb'] = df.action_name.apply(lambda x: x.split('_')[0])
        verb_index = sorted(df.verb.unique())
        verb_class = pd.Series(range(len(verb_index)), verb_index)
        df['verb_class'] = df.verb.apply(lambda x: verb_class[x])
        df['fps'] = self.FPS
        self.narr_df = df.set_index('narration_id')
