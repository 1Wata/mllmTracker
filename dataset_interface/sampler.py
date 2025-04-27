import random
import torch.utils.data
import cv2 as cv
from typing import List, Optional, Callable


def opencv_loader(path):
    """Read image using opencv's imread function and returns it in rgb format"""
    try:
        im = cv.imread(path, cv.IMREAD_COLOR)

        # convert to rgb and return
        return cv.cvtColor(im, cv.COLOR_BGR2RGB)
    except Exception as e:
        print('ERROR: Could not read image "{}"'.format(path))
        print(e)
        return None


def no_processing(data):
    return data


import torch
import torch.utils.data
import random


def no_processing(x):
    return x


class TrackingSampler(torch.utils.data.Dataset):
    """Class responsible for sampling frames from training sequences to form batches.

    The sampling is done in the following ways. First a dataset is selected at random. Next, a sequence is selected
    from that dataset. A base frame is then sampled randomly from the sequence. Next, a set of 'train frames' and
    'test frames' are sampled from the sequence from the range [base_frame_id - max_gap, base_frame_id]  and
    (base_frame_id, base_frame_id + max_gap] respectively. Only the frames in which the target is visible are sampled.
    If enough visible frames are not found, the 'max_gap' is increased gradually till enough frames are found.

    The sampled frames are then passed through the input 'processing' function for the necessary processing-
    """

    def __init__(
        self,
        datasets: List,
        p_datasets: Optional[List[float]] = None,
        samples_per_epoch: int = 1000,
        max_gap: int = 100,
        num_search_frames: int = 1,
        num_template_frames: int = 4,
        processing: Callable = no_processing,
        frame_sample_mode: str = "causal",
        train_cls: bool = False,
        pos_prob: float = 0.5,
        # max_prev_template_frames: int = 3,  # 修改为最大前序帧数，随机采样0-3个前序帧
        unify_out_format: bool = True,
    ):
        """
        args:
            datasets - List of datasets to be used for training
            p_datasets - List containing the probabilities by which each dataset will be sampled
            samples_per_epoch - Number of training samples per epoch
            max_gap - Maximum gap, in frame numbers, between the train frames and the test frames.
            num_search_frames - Number of search frames to sample.
            processing - An instance of Processing class which performs the necessary processing of the data.
            frame_sample_mode - Either 'causal' or 'interval'. If 'causal', then the test frames are sampled in a causally,
                                otherwise randomly within the interval.
            train_cls - Whether we are training classification
            pos_prob - Probability of sampling positive class when making classification
            max_prev_template_frames - 最大前序模板帧数，会随机采样0-3个前序帧
        """
        self.datasets = datasets
        self.train_cls = train_cls  # whether we are training classification
        self.pos_prob = pos_prob  # probability of sampling positive class when making classification
        # self.max_prev_template_frames = max_prev_template_frames
        self.num_template_frames = num_template_frames
        self.unify_out_format = unify_out_format

        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [len(d) for d in self.datasets]

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x / p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.max_gap = max_gap
        self.num_search_frames = num_search_frames

        self.processing = processing
        self.frame_sample_mode = frame_sample_mode

        if self.num_search_frames != 1:
            raise ValueError(
                "num_search_frames must be 1 for the desired modification."
            )

    def __len__(self):
        return self.samples_per_epoch

    def _sample_visible_ids(
        self,
        visible: torch.Tensor,
        num_ids: int = 1,
        min_id: Optional[int] = None,
        max_id: Optional[int] = None,
        # allow_invisible: bool = False,
        allow_invisible: bool = True,
        force_invisible: bool = False,
        # force_invisible: bool = True,
    ) -> Optional[List[int]]:
        """Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be samples
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if num_ids == 0:
            return
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible):
            max_id = len(visible)
        # get valid ids
        if force_invisible:
            valid_ids = [i for i in range(min_id, max_id) if not visible[i]]
        else:
            if allow_invisible:
                valid_ids = [i for i in range(min_id, max_id)]
            else:
                valid_ids = [i for i in range(min_id, max_id) if visible[i]]

        # No visible ids
        if len(valid_ids) == 0:
            return None

        # return random.choices(valid_ids, k=num_ids)
        return random.sample(valid_ids, k=num_ids)

    def __getitem__(self, index):

        num_sample = random.randint(1, 1)  # Placeholder for the number of samples
        return self.getitem_withoutpreprocess(num_sample=num_sample)

    def getitem_withoutpreprocess(self, num_sample):
        """
        returns:
            TensorDict - dict containing all the data blocks
        """
        # Select a dataset
        dataset = random.choices(self.datasets, self.p_datasets)[0]

        is_video_dataset = dataset.is_video_sequence()

        # sample a sequence from the given dataset
        seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(
            dataset, is_video_dataset
        )

        if is_video_dataset:
            template_frame_ids = None
            search_frame_ids = None
            gap_increase = 0

            if self.frame_sample_mode == 'causal':
                # Sample template and search frames in a causal manner
                while search_frame_ids is None:
                    base_frame_id = self._sample_visible_ids(
                        visible, 
                        num_ids=1, 
                        min_id=0,
                        max_id=len(visible) - self.num_template_frames,  # 确保后面有足够的帧可以作为搜索帧
                        allow_invisible=False  # 确保基准帧是可见的
                    )
                    
                    if base_frame_id is None:
                        gap_increase += 5
                        continue
                    
                    # 设置模板帧为基准帧
                    # template_frame_ids = base_frame_id
                    template_frame_ids = self._sample_visible_ids(
                        visible,
                        num_ids=self.num_template_frames - 1,  # 采样num_template_frames-1个前序帧
                        min_id=base_frame_id[0] - self.max_gap - gap_increase,
                        max_id=base_frame_id[0] - 1,  # 确保前序帧在基准帧之前
                    )
                    template_frame_ids = template_frame_ids + base_frame_id
                    
                    # gap = 200
                    gap = self.num_template_frames + 1
                    search_frame_ids = self._sample_visible_ids(
                        visible,
                        num_ids=self.num_template_frames - 1,
                        min_id=template_frame_ids[-1] + 1,
                        max_id=min(template_frame_ids[-1] + gap, len(visible)),
                    )
                    # Make sure search_frame_ids is sorted to maintain temporal order

                    search_frame_ids.sort()
                    
                    gap_increase += 5

#######################################################################################################################################
            # if self.frame_sample_mode == 'causal':
            #     # Sample test and train frames in a causal manner, i.e. search_frame_ids > template_frame_ids
            #     while search_frame_ids is None:
            #         base_frame_id = self._sample_visible_ids(visible, num_ids=1, min_id=self.num_template_frames - 1,
            #                                                     max_id=len(visible) - self.num_search_frames)
            #         prev_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_template_frames - 1,
            #                                                     min_id=base_frame_id[0] - self.max_gap - gap_increase,
            #                                                     max_id=base_frame_id[0])
            #         # Sort prev_frame_ids to ensure causal ordering
            #         if prev_frame_ids is not None:
            #             prev_frame_ids.sort()  # Sort in ascending order for causal relationship
            #         if prev_frame_ids is None:
            #             gap_increase += 5
            #             continue
            #         template_frame_ids = prev_frame_ids + base_frame_id
            #         search_frame_ids = self._sample_visible_ids(visible, min_id=template_frame_ids[0] + 1,
            #                                                     max_id=template_frame_ids[0] + self.max_gap + gap_increase,
            #                                                     num_ids=self.num_search_frames)
            #         # Increase gap until a frame is found
            #         gap_increase += 5


# #######################################################################################################################################
#             if self.frame_sample_mode == 'causal':
#                 # Sample test and train frames in a causal manner, i.e. search_frame_ids > template_frame_ids
#                 while search_frame_ids is None:
#                     # 首先采样最后一个模板帧 (base_frame_id)
#                     base_frame_id = self._sample_visible_ids(
#                         visible, 
#                         num_ids=1, 
#                         min_id=self.num_template_frames - 1,
#                         max_id=len(visible)
#                     )
#                     if base_frame_id is None:
#                         gap_increase += 5
#                         continue

#                     # 采样之前的模板帧
#                     prev_frame_ids = self._sample_visible_ids(
#                         visible, 
#                         num_ids=self.num_template_frames - 1,
#                         min_id=base_frame_id[0] - self.max_gap - gap_increase,
#                         max_id=base_frame_id[0] - 1
#                     )
                    
#                     # 确保前序帧按顺序排列
#                     if prev_frame_ids is not None:
#                         prev_frame_ids.sort()
#                     else:
#                         gap_increase += 5
#                         continue
                        
#                     template_frame_ids = prev_frame_ids

#                     # search_frame_ids = base_frame_id
#                     # search_frame_ids = self._sample_visible_ids(
#                     #     visible, 
#                     #     min_id=max(template_frame_ids) + 1,
#                     #     max_id=min(max(template_frame_ids) + self.max_gap + gap_increase, len(visible)),
#                     #     num_ids=self.num_search_frames
#                     # )
#                     search_frame_ids = self._sample_visible_ids(
#                         visible, 
#                         min_id=max(template_frame_ids) + 1,
#                         max_id=min(max(template_frame_ids) + self.max_gap + gap_increase, len(visible)),
#                         num_ids=self.num_search_frames
#                     )
                    
#                     gap_increase += 5




            elif (
                self.frame_sample_mode == "trident"
                or self.frame_sample_mode == "trident_pro"
            ):
                template_frame_ids, search_frame_ids = self.get_frame_ids_trident(
                    visible
                )
                if search_frame_ids is not None:
                    search_frame_ids = [
                        max(template_frame_ids) + 1
                    ]  # Modify for the desired logic
            elif self.frame_sample_mode == "stark":
                template_frame_ids, search_frame_ids = self.get_frame_ids_stark(
                    visible, seq_info_dict["valid"]
                )
                if search_frame_ids is not None:
                    search_frame_ids = [
                        max(template_frame_ids) + 1
                    ]  # Modify for the desired logic
            else:
                raise ValueError("Illegal frame sample mode")
        else:
            # In case of image dataset, just repeat the image to generate synthetic video
            # 随机决定模板帧数量（0-4，其中0表示无模板，1表示只有基准帧，2-4表示基准帧+1-3个前序帧）
            num_prev_frames = random.randint(0, self.max_prev_template_frames + 1)
            
            # 根据num_prev_frames决定模板帧数量
            if num_prev_frames > 0:
                template_frame_ids = [1] * num_prev_frames
            else:
                template_frame_ids = []  # 无模板帧

        if self.unify_out_format:
            # Validate that search_frame_ids come after template_frame_ids (causal logic)

            if max(template_frame_ids) >= min(search_frame_ids):
                raise ValueError(f"Causal logic violation: template frames {template_frame_ids} should come before search frames {search_frame_ids}")
            
            frames, anno, meta_obj = dataset.get_frames(
                seq_id, template_frame_ids + search_frame_ids, seq_info_dict
            )
            
            data = dict(
                {
                    "images": frames,
                    "anno": anno,
                    "dataset_name": dataset.get_name(),
                    "test_class": meta_obj.get("object_class_name"),
                    "exp_str": meta_obj.get("exp_str"),
                }
            )


        else:
            template_frames, template_anno, meta_obj_train = dataset.get_frames(
                seq_id, template_frame_ids, seq_info_dict, use_first_frame_template=False
            )
            search_frames, search_anno, meta_obj_test = dataset.get_frames(
                seq_id, search_frame_ids, seq_info_dict, use_first_frame_template=False
            )

            data = dict(
                {
                    "template_images": template_frames,
                    "template_anno": template_anno,
                    "search_images": search_frames,  # ori image
                    "search_anno": search_anno,
                    "exp_str": meta_obj_train.get("exp_str"),
                    # "search_masks": search_masks,
                    "dataset_name": dataset.get_name(),
                    "test_class": meta_obj_test.get("object_class_name"),
                }
            )

        return data


    def get_center_box(self, H, W, ratio=1 / 8):
        cx, cy, w, h = W / 2, H / 2, W * ratio, H * ratio
        return torch.tensor([int(cx - w / 2), int(cy - h / 2), int(w), int(h)])

    def sample_seq_from_dataset(self, dataset, is_video_dataset):
        # Sample a sequence with enough visible frames
        enough_visible_frames = False
        while not enough_visible_frames:
            # Sample a sequence
            seq_id = random.randint(0, dataset.get_num_sequences() - 1)

            # Sample frames
            seq_info_dict = dataset.get_sequence_info(seq_id)
            visible = seq_info_dict["visible"]

            # 只要确保序列足够长，能提供至少一个基准帧和一个搜索帧就可以了
            # 前序帧会在采样时根据实际情况决定采取多少
            enough_visible_frames = (
                visible.type(torch.int64).sum().item() > self.num_search_frames + 1  # 至少需要一个基准帧和一个搜索帧
                and len(visible) >= 2  # 至少有2帧（基准帧和搜索帧）
            )

            enough_visible_frames = enough_visible_frames or not is_video_dataset
        return seq_id, visible, seq_info_dict

    def get_one_search(self):
        # Select a dataset
        dataset = random.choices(self.datasets, self.p_datasets)[0]

        is_video_dataset = dataset.is_video_sequence()
        # sample a sequence
        seq_id, visible, seq_info_dict = self.sample_seq_from_dataset(
            dataset, is_video_dataset
        )
        # sample a frame
        if is_video_dataset:
            if self.frame_sample_mode == "stark":
                search_frame_ids = self._sample_visible_ids(
                    seq_info_dict["valid"], num_ids=1
                )
            else:
                search_frame_ids = self._sample_visible_ids(
                    visible, num_ids=1, allow_invisible=True
                )
        else:
            search_frame_ids = [1]
        # get the image, bounding box and other info
        search_frames, search_anno, meta_obj_test = dataset.get_frames(
            seq_id, search_frame_ids, seq_info_dict
        )

        return search_frames, search_anno, meta_obj_test

    def get_frame_ids_trident(self, visible):
        # get template and search ids in a 'trident' manner
        template_frame_ids_extra = list()
        while None in template_frame_ids_extra or len(template_frame_ids_extra) == 0:
            template_frame_ids_extra = list()
            # first randomly sample two frames from a video
            template_frame_id1 = self._sample_visible_ids(
                visible, num_ids=1
            )  # the initial template id
            if template_frame_id1 is None:
                return None, None
            search_frame_ids = self._sample_visible_ids(
                visible, num_ids=1
            )  # the search region id
            if search_frame_ids is None:
                return None, None
            # get the dynamic template id
            for max_gap in self.max_gap:
                if template_frame_id1[0] >= search_frame_ids[0]:
                    min_id, max_id = search_frame_ids[0], search_frame_ids[0] + max_gap
                else:
                    min_id, max_id = search_frame_ids[0] - max_gap, search_frame_ids[0]
                if self.frame_sample_mode == "trident_pro":
                    f_id = self._sample_visible_ids(
                        visible,
                        num_ids=1,
                        min_id=min_id,
                        max_id=max_id,
                        allow_invisible=True,
                    )
                else:
                    f_id = self._sample_visible_ids(
                        visible, num_ids=1, min_id=min_id, max_id=max_id
                    )
                if f_id is None:
                    template_frame_ids_extra += [None]
                else:
                    template_frame_ids_extra += f_id

        template_frame_ids = template_frame_id1 + template_frame_ids_extra
        return template_frame_ids, search_frame_ids

    def get_frame_ids_stark(self, visible, valid):
        # get template and search ids in a 'stark' manner
        template_frame_ids_extra = list()
        while None in template_frame_ids_extra or len(template_frame_ids_extra) == 0:
            template_frame_ids_extra = list()
            # first randomly sample two frames from a video
            template_frame_id1 = self._sample_visible_ids(
                visible, num_ids=1
            )  # the initial template id
            if template_frame_id1 is None:
                return None, None
            search_frame_ids = self._sample_visible_ids(
                visible, num_ids=1
            )  # the search region id
            if search_frame_ids is None:
                return None, None
            # get the dynamic template id
            for max_gap in self.max_gap:
                if template_frame_id1[0] >= search_frame_ids[0]:
                    min_id, max_id = search_frame_ids[0], search_frame_ids[0] + max_gap
                else:
                    min_id, max_id = search_frame_ids[0] - max_gap, search_frame_ids[0]
                """we require the frame to be valid but not necessary visible"""
                f_id = self._sample_visible_ids(
                    valid, num_ids=1, min_id=min_id, max_id=max_id
                )
                if f_id is None:
                    template_frame_ids_extra += [None]
                else:
                    template_frame_ids_extra += f_id

        template_frame_ids = template_frame_id1 + template_frame_ids_extra
        return template_frame_ids, search_frame_ids