import jpeg4py
import cv2 as cv
import re
import numpy as np
import pandas as pd
import importlib
import os
from collections import OrderedDict
from PIL import Image, ImageDraw

def load_text_numpy(path, delimiter, dtype):
    if isinstance(delimiter, (tuple, list)):
        for d in delimiter:
            try:
                ground_truth_rect = np.loadtxt(path, delimiter=d, dtype=dtype)
                return ground_truth_rect
            except:
                pass

        raise Exception('Could not read file {}'.format(path))
    else:
        ground_truth_rect = np.loadtxt(path, delimiter=delimiter, dtype=dtype)
        return ground_truth_rect


def load_text_pandas(path, delimiter, dtype):
    if isinstance(delimiter, (tuple, list)):
        for d in delimiter:
            try:
                ground_truth_rect = pd.read_csv(path, delimiter=d, header=None, dtype=dtype, na_filter=False,
                                                low_memory=False).values
                return ground_truth_rect
            except Exception as e:
                pass

        raise Exception('Could not read file {}'.format(path))
    else:
        ground_truth_rect = pd.read_csv(path, delimiter=delimiter, header=None, dtype=dtype, na_filter=False,
                                        low_memory=False).values
        return ground_truth_rect


def load_text(path, delimiter=' ', dtype=np.float32, backend='numpy'):
    if backend == 'numpy':
        return load_text_numpy(path, delimiter, dtype)
    elif backend == 'pandas':
        return load_text_pandas(path, delimiter, dtype)


def load_str(path):
    with open(path, "r") as f:
        text_str = f.readline().strip().lower()
    return text_str



def clean_string(expression):
    return re.sub(r"([.,'!?\"()*#:;])", '', expression.lower()).replace('-', ' ').replace('/', ' ')

def jpeg4py_loader_w_failsafe(path):
    """ Image reading using jpeg4py https://github.com/ajkxyz/jpeg4py"""
    try:
        return jpeg4py.JPEG(path).decode()
    except:
        try:
            im = cv.imread(path, cv.IMREAD_COLOR)

            # convert to rgb and return
            return cv.cvtColor(im, cv.COLOR_BGR2RGB)
        except Exception as e:
            print('ERROR: Could not read image "{}"'.format(path))
            print(e)
            return None
        


def create_default_local_file():
    path = os.path.join(os.path.dirname(__file__), 'local.py')

    empty_str = '\'\''
    default_settings = OrderedDict({
        'workspace_dir': empty_str,
        'tensorboard_dir': 'self.workspace_dir + \'/tensorboard/\'',
        'pretrained_networks': 'self.workspace_dir + \'/pretrained_networks/\'',
        'lasot_dir': empty_str,
        'got10k_dir': empty_str,
        'trackingnet_dir': empty_str,
        'coco_dir': empty_str,
        'lvis_dir': empty_str,
        'sbd_dir': empty_str,
        'imagenet_dir': empty_str,
        'imagenetdet_dir': empty_str,
        'ecssd_dir': empty_str,
        'hkuis_dir': empty_str,
        'msra10k_dir': empty_str,
        'davis_dir': empty_str,
        'youtubevos_dir': empty_str})

    comment = {'workspace_dir': 'Base directory for saving network checkpoints.',
               'tensorboard_dir': 'Directory for tensorboard files.'}

    with open(path, 'w') as f:
        f.write('class EnvironmentSettings:\n')
        f.write('    def __init__(self):\n')

        for attr, attr_val in default_settings.items():
            comment_str = None
            if attr in comment:
                comment_str = comment[attr]
            if comment_str is None:
                f.write('        self.{} = {}\n'.format(attr, attr_val))
            else:
                f.write('        self.{} = {}    # {}\n'.format(attr, attr_val, comment_str))


def create_default_local_file_ITP_train(workspace_dir, data_dir):
    path = os.path.join(os.path.dirname(__file__), 'local.py')

    empty_str = '\'\''
    default_settings = OrderedDict({
        'workspace_dir': workspace_dir,
        'tensorboard_dir': os.path.join(workspace_dir, 'tensorboard'),    # Directory for tensorboard files.
        'pretrained_networks': os.path.join(workspace_dir, 'pretrained_networks'),
        'lasot_dir': os.path.join(data_dir, 'lasot'),
        'got10k_dir': os.path.join(data_dir, 'got10k/train'),
        'got10k_val_dir': os.path.join(data_dir, 'got10k/val'),
        'lasot_lmdb_dir': os.path.join(data_dir, 'lasot_lmdb'),
        'got10k_lmdb_dir': os.path.join(data_dir, 'got10k_lmdb'),
        'trackingnet_dir': os.path.join(data_dir, 'trackingnet'),
        'trackingnet_lmdb_dir': os.path.join(data_dir, 'trackingnet_lmdb'),
        'coco_dir': os.path.join(data_dir, 'coco'),
        'coco_lmdb_dir': os.path.join(data_dir, 'coco_lmdb'),
        'lvis_dir': empty_str,
        'sbd_dir': empty_str,
        'imagenet_dir': os.path.join(data_dir, 'vid'),
        'imagenet_lmdb_dir': os.path.join(data_dir, 'vid_lmdb'),
        'imagenetdet_dir': empty_str,
        'ecssd_dir': empty_str,
        'hkuis_dir': empty_str,
        'msra10k_dir': empty_str,
        'davis_dir': os.path.join(data_dir, 'davis'),
        'youtubevos_dir': os.path.join(data_dir, 'youtubevos'),
        'tracking_masks_dir': os.path.join(data_dir, 'tracking_masks'),
        # 'tnl2k_dir': os.path.join(data_dir, 'tnl2k'),
        'tnl2k_dir': os.path.join(data_dir, 'TNL2K_CVPR2021'),
        'otb_lang_dir': os.path.join(data_dir, 'otb_lang'),
        'refer_youtubevos_dir': os.path.join(data_dir, 'refer_youtubevos'),
        'ref_coco_dir': os.path.join(data_dir, 'ref_coco'),
        })

    comment = {'workspace_dir': 'Base directory for saving network checkpoints.',
               'tensorboard_dir': 'Directory for tensorboard files.'}

    with open(path, 'w') as f:
        f.write('class EnvironmentSettings:\n')
        f.write('    def __init__(self):\n')

        for attr, attr_val in default_settings.items():
            comment_str = None
            if attr in comment:
                comment_str = comment[attr]
            if comment_str is None:
                if attr_val == empty_str:
                    f.write('        self.{} = {}\n'.format(attr, attr_val))
                else:
                    f.write('        self.{} = \'{}\'\n'.format(attr, attr_val))
            else:
                f.write('        self.{} = \'{}\'    # {}\n'.format(attr, attr_val, comment_str))


def env_settings():
    env_module_name = 'lib.train.admin.local'
    try:
        env_module = importlib.import_module(env_module_name)
        return env_module.EnvironmentSettings()
    except:
        env_file = os.path.join(os.path.dirname(__file__), 'local.py')

        create_default_local_file()
        raise RuntimeError('YOU HAVE NOT SETUP YOUR local.py!!!\n Go to "{}" and set all the paths you need. Then try to run again.'.format(env_file))



def jpeg4py_loader(path):
    """ Image reading using jpeg4py https://github.com/ajkxyz/jpeg4py"""
    try:
        return jpeg4py.JPEG(path).decode()
    except Exception as e:
        print('ERROR: Could not read image "{}"'.format(path))
        print(e)
        return None
    

def normalize_bbox_xyhw(bbox, img_width, img_height):
    """
    将边界框坐标从绝对像素值归一化为相对于图像尺寸的比例值，并乘以1000转为整数。
    
    Args:
        bbox (list): 包含[x, y, w, h]格式的边界框坐标
        img_width (int): 图像宽度
        img_height (int): 图像高度
        
    Returns:
        list: 归一化后的边界框坐标 [x_min, y_min, x_max, y_max]，乘以1000转为整数
    """
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    
    x, y, w, h = bbox
    return [
        int(float(x) / img_width * 1000),          # x_min 归一化，乘以1000转为整数
        int(float(y) / img_height * 1000),         # y_min 归一化，乘以1000转为整数
        int(float(x + w) / img_width * 1000),      # x_max 归一化，乘以1000转为整数
        int(float(y + h) / img_height * 1000)      # y_max 归一化，乘以1000转为整数
    ]

def normalize_bbox_xyxy(bbox, img_width, img_height):
    """
    将边界框坐标从绝对像素值归一化为相对于图像尺寸的比例值，并乘以1000转为整数。
    
    Args:
        bbox (list): 包含[x_min, y_min, x_max, y_max]格式的边界框坐标
        img_width (int): 图像宽度
        img_height (int): 图像高度
        
    Returns:
        list: 归一化后的边界框坐标 [x_min, y_min, x_max, y_max]，乘以1000转为整数
    """
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    
    x_min, y_min, x_max, y_max = bbox
    return [
        int(float(x_min) / img_width * 1000),      # x_min 归一化，乘以1000转为整数
        int(float(y_min) / img_height * 1000),     # y_min 归一化，乘以1000转为整数
        int(float(x_max) / img_width * 1000),      # x_max 归一化，乘以1000转为整数
        int(float(y_max) / img_height * 1000)      # y_max 归一化，乘以1000转为整数
    ]

def unnormalize_bbox(bbox, img_width, img_height):
    """
    将归一化的边界框坐标（乘以1000的整数）转换回绝对像素值，保留一位小数。
    
    Args:
        bbox (list): 包含[x_min, y_min, x_max, y_max]格式的归一化边界框坐标
        img_width (int): 图像宽度
        img_height (int): 图像高度
        
    Returns:
        list: 转换回绝对像素值的边界框坐标 [x_min, y_min, width, height]，保留一位小数
    """
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    
    x_min, y_min, x_max, y_max = bbox
    return [
        round(float(x_min) * img_width / 1000, 1),
        round(float(y_min) * img_height / 1000, 1),
        round(float(x_max - x_min) * img_width / 1000, 1),
        round(float(y_max - y_min) * img_height / 1000, 1)
    ]


def draw_normed_bbox(image, bbox):
    """
    Draw normalized bounding box on image
    """
    if not bbox:
        return image
        
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = bbox
    W, H = image.size
    draw.rectangle([x1 * W / 1000, y1 * H / 1000, x2 * W / 1000, y2 * H / 1000], outline="red", width=2)
    return image


