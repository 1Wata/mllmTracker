import os
import torch
import sys
import json
from tqdm import tqdm
import cv2


# datasets related
# from dataset import Lasot, Got10k, MSCOCOSeq, ImagenetVID
from train import Lasot, Got10k, TrackingNet
from train import TNL2k, TNL2k_Lang, Lasot_Lang, OTB_Lang, RefCOCOSeq
# from dataset import (
#     Lasot_lmdb,
#     Got10k_lmdb,
#     MSCOCOSeq_lmdb,
#     ImagenetVID_lmdb,
#     TrackingNet_lmdb,
# )
from sampler import TrackingSampler, opencv_loader
from datasets import Dataset


def update_settings(settings, cfg):
    settings.print_interval = cfg.TRAIN.PRINT_INTERVAL
    settings.search_area_factor = {
        "template": cfg.DATA.TEMPLATE.FACTOR,
        "search": cfg.DATA.SEARCH.FACTOR,
    }
    settings.output_sz = {
        "template": cfg.DATA.TEMPLATE.SIZE,
        "search": cfg.DATA.SEARCH.SIZE,
    }
    settings.center_jitter_factor = {
        "template": cfg.DATA.TEMPLATE.CENTER_JITTER,
        "search": cfg.DATA.SEARCH.CENTER_JITTER,
    }
    settings.scale_jitter_factor = {
        "template": cfg.DATA.TEMPLATE.SCALE_JITTER,
        "search": cfg.DATA.SEARCH.SCALE_JITTER,
    }
    settings.grad_clip_norm = cfg.TRAIN.GRAD_CLIP_NORM
    settings.print_stats = None
    settings.batchsize = cfg.TRAIN.BATCH_SIZE
    settings.scheduler_type = cfg.TRAIN.SCHEDULER.TYPE


def names2datasets(name_list: list, settings, image_loader):
    assert isinstance(name_list, list)
    datasets = []
    for name in name_list:
        assert name in [
            "LASOT",
            "GOT10K_vottrain",
            "GOT10K_votval",
            "GOT10K_train_full",
            "GOT10K_official_val",
            # "COCO17",
            # "VID",
            "TRACKINGNET",
            "LASOT_Lang",
            "TNL2K",
            "TNL2K_Lang",
            "OTB_Lang",
            "RefCOCO14",
        ]
        # Tracking Task
        if name == "TNL2K":
            datasets.append(TNL2k(settings.env.tnl2k_dir, split="train"))
        if name == "LASOT":
            datasets.append(
                Lasot(
                    settings.env.lasot_dir, split="train", image_loader=image_loader
                )
            )
        if name == "GOT10K_vottrain":
            datasets.append(
                Got10k(
                    settings.env.got10k_dir,
                    split="vottrain",
                    image_loader=image_loader,
                )
            )
        if name == "GOT10K_train_full":
            datasets.append(
                Got10k(
                    settings.env.got10k_dir,
                    split="train_full",
                    image_loader=image_loader,
                )
            )
        if name == "GOT10K_votval":

            datasets.append(
                Got10k(
                    settings.env.got10k_dir,
                    split="votval",
                    image_loader=image_loader,
                )
            )
        if name == "GOT10K_official_val":
            if settings.use_lmdb:
                raise ValueError("Not implement")
            else:
                datasets.append(
                    Got10k(
                        settings.env.got10k_val_dir,
                        split=None,
                        image_loader=image_loader,
                    )
                )
        # if name == "COCO17":
        #     if settings.use_lmdb:
        #         print("Building COCO2017 from lmdb")
        #         datasets.append(
        #             MSCOCOSeq_lmdb(
        #                 settings.env.coco_lmdb_dir,
        #                 version="2017",
        #                 image_loader=image_loader,
        #             )
        #         )
        #     else:
        #         datasets.append(
        #             MSCOCOSeq(
        #                 settings.env.coco_dir, version="2017", image_loader=image_loader
        #             )
        #         )
        # if name == "VID":
        #     if settings.use_lmdb:
        #         print("Building VID from lmdb")
        #         datasets.append(
        #             ImagenetVID_lmdb(
        #                 settings.env.imagenet_lmdb_dir, image_loader=image_loader
        #             )
        #         )
        #     else:
        #         datasets.append(
        #             ImagenetVID(settings.env.imagenet_dir, image_loader=image_loader)
        #         )
        if name == "TRACKINGNET":
            datasets.append(
                TrackingNet(settings.env.trackingnet_dir, image_loader=image_loader)
            )

        # Visual-Language Task
        if name == "TNL2K_Lang":
            datasets.append(TNL2k_Lang(settings.env.tnl2k_dir, split="TNL2K_train_subset"))
        if name == "LASOT_Lang":
            datasets.append(
                Lasot_Lang(
                    settings.env.lasot_dir, split="train", image_loader=image_loader
                )
            )
        if name == "OTB_Lang":
            datasets.append(
                OTB_Lang(
                    settings.env.otb_lang_dir, split="train", image_loader=image_loader
                )
            )
        if name == "RefCOCO14":
            datasets.append(
                RefCOCOSeq(
                    settings.env.ref_coco_dir,
                    refcoco_type="refcoco-unc",
                    version="2014",
                    image_loader=image_loader,
                )
            )

    return datasets


def build_dataset():
    from easydict import EasyDict
    import yaml
    import os

    cfg = EasyDict(
        yaml.safe_load(
            open(os.path.join(os.path.dirname(__file__), "dataset_config.yaml"))
        )
    )

    dataset_train = TrackingSampler(
        datasets=names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, cfg, opencv_loader),
        p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
        samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
        max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL,
        num_search_frames=cfg.DATA.SEARCH.NUMBER,
        num_template_frames=cfg.DATA.TEMPLATE.NUMBER,
        # max_prev_template_frames=cfg.DATA.PREV_TEMPLATE.NUMBER - 1,
    )

    # Validation samplers and loaders
    dataset_val = TrackingSampler(
        datasets=names2datasets(cfg.DATA.VAL.DATASETS_NAME, cfg, opencv_loader),
        p_datasets=cfg.DATA.VAL.DATASETS_RATIO,
        samples_per_epoch=cfg.DATA.VAL.SAMPLE_PER_EPOCH,
        max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL,
        num_search_frames=cfg.DATA.SEARCH.NUMBER,
        num_template_frames=cfg.DATA.TEMPLATE.NUMBER,
        # max_prev_template_frames=cfg.DATA.PREV_TEMPLATE.NUMBER - 1,
    )

    return dataset_train, dataset_val

if __name__ == "__main__":
    train_dataset, val_dataset = build_dataset()
    train_data0 = train_dataset[0]
    print(train_data0.keys())
    print(train_data0["template_images"])
    print(train_data0["template_anno"])
    print(train_data0["dataset_name"])

    # print(train_data0["template_anno"])
    # for i in range(len(train_data0["template_images"])):
    #     image_path = train_data0["template_images"][i]
    #     bbox = train_data0["template_anno"][i]
    #     image = cv2.imread(image_path)
    #     x, y, w, h = map(int, bbox)
    #     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    #     cv2.imshow("Template Image Annotation", image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    # print()
    # for i in range(len(train_data0["search_images"])):
    #     print(train_data0["search_images"][i])
    #     print(train_data0["search_anno"][i])

    # val_data0 = val_dataset[0]
    # print(val_data0.keys())
    # print(len(val_data0["template_images"]))
    # for i in range(len(val_data0["template_images"])):
    #     print(val_data0["template_images"][i])
    # train_dataset.samples_per_epoch = 10
    # build_alpaca_vqa_dataset(train_dataset)