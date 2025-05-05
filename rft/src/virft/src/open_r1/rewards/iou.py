import math

def miou(bbox1, bbox2):
    """计算两个边界框之间的IoU"""
    if bbox1 is None or bbox2 is None:
        return 0.0

    if bbox1 == [0, 0, 0, 0] or bbox2 == [0, 0, 0, 0]:
        return 0.0

    x1, y1, x2, y2 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # 确保坐标有效 (x1 < x2, y1 < y2)
    if x1 > x2: x1, x2 = x2, x1
    if y1 > y2: y1, y2 = y2, y1
    if x1_2 > x2_2: x1_2, x2_2 = x2_2, x1_2
    if y1_2 > y2_2: y1_2, y2_2 = y2_2, y1_2

    # 计算交集区域坐标
    xi1 = max(x1, x1_2)
    yi1 = max(y1, y1_2)
    xi2 = min(x2, x2_2)
    yi2 = min(y2, y2_2)

    # 计算交集面积
    intersection_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # 计算两个边界框的面积
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    # 检查面积是否有效
    if area1 <= 0 or area2 <= 0:
        return 0.0

    # 计算并集面积
    union_area = area1 + area2 - intersection_area

    # 避免除以零
    if union_area == 0:
        return 0.0 # 或者根据情况返回 1.0 如果两个框完全重合且面积为0?

    # 计算IoU
    iou_value = intersection_area / union_area
    return iou_value


def giou(bbox1, bbox2):
    """计算两个边界框之间的GIoU"""
    iou_value = miou(bbox1, bbox2)

    x1, y1, x2, y2 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # 确保坐标有效
    if x1 > x2: x1, x2 = x2, x1
    if y1 > y2: y1, y2 = y2, y1
    if x1_2 > x2_2: x1_2, x2_2 = x2_2, x1_2
    if y1_2 > y2_2: y1_2, y2_2 = y2_2, y1_2

    # 计算两个边界框的面积
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    if area1 <= 0 or area2 <= 0:
        return 0.0 # 如果原始框无效，GIoU也无意义

    # 计算交集面积
    xi1 = max(x1, x1_2)
    yi1 = max(y1, y1_2)
    xi2 = min(x2, x2_2)
    yi2 = min(y2, y2_2)
    intersection_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # 计算并集面积
    union_area = area1 + area2 - intersection_area
    if union_area == 0:
        return 0.0 # 或者根据情况处理

    # 计算最小闭包区域坐标 (C)
    xc1 = min(x1, x1_2)
    yc1 = min(y1, y1_2)
    xc2 = max(x2, x2_2)
    yc2 = max(y2, y2_2)

    # 计算最小闭包区域面积
    c_area = (xc2 - xc1) * (yc2 - yc1)
    if c_area == 0:
        return iou_value # 如果闭包面积为0，退化为IoU

    # 计算GIoU
    giou_value = iou_value - (c_area - union_area) / c_area
    return giou_value


def diou(bbox1, bbox2):
    """计算两个边界框之间的DIoU"""
    iou_value = miou(bbox1, bbox2)

    x1, y1, x2, y2 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # 确保坐标有效
    if x1 > x2: x1, x2 = x2, x1
    if y1 > y2: y1, y2 = y2, y1
    if x1_2 > x2_2: x1_2, x2_2 = x2_2, x1_2
    if y1_2 > y2_2: y1_2, y2_2 = y2_2, y1_2

    # 计算两个边界框的中心点
    center_x1 = (x1 + x2) / 2
    center_y1 = (y1 + y2) / 2
    center_x2 = (x1_2 + x2_2) / 2
    center_y2 = (y1_2 + y2_2) / 2

    # 计算中心点之间的平方距离 (rho^2)
    center_distance_sq = (center_x1 - center_x2)**2 + (center_y1 - center_y2)**2

    # 计算最小闭包区域坐标 (C)
    xc1 = min(x1, x1_2)
    yc1 = min(y1, y1_2)
    xc2 = max(x2, x2_2)
    yc2 = max(y2, y2_2)

    # 计算最小闭包区域对角线的平方距离 (c^2)
    diagonal_distance_sq = (xc2 - xc1)**2 + (yc2 - yc1)**2
    if diagonal_distance_sq == 0:
        return iou_value # 如果闭包面积为0，退化为IoU

    # 计算DIoU
    diou_value = iou_value - (center_distance_sq / diagonal_distance_sq)
    return diou_value


def ciou(bbox1, bbox2, eps=1e-7):
    """计算两个边界框之间的CIoU"""
    iou_value = miou(bbox1, bbox2)

    x1, y1, x2, y2 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # 确保坐标有效
    if x1 > x2: x1, x2 = x2, x1
    if y1 > y2: y1, y2 = y2, y1
    if x1_2 > x2_2: x1_2, x2_2 = x2_2, x1_2
    if y1_2 > y2_2: y1_2, y2_2 = y2_2, y1_2

    # 计算宽度和高度
    w1, h1 = x2 - x1, y2 - y1
    w2, h2 = x2_2 - x1_2, y2_2 - y1_2

    # 检查宽度和高度是否有效
    if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
        return iou_value # 如果原始框无效，退化为IoU

    # 计算中心点之间的平方距离 (rho^2)
    center_x1 = (x1 + x2) / 2
    center_y1 = (y1 + y2) / 2
    center_x2 = (x1_2 + x2_2) / 2
    center_y2 = (y1_2 + y2_2) / 2
    center_distance_sq = (center_x1 - center_x2)**2 + (center_y1 - center_y2)**2

    # 计算最小闭包区域坐标 (C)
    xc1 = min(x1, x1_2)
    yc1 = min(y1, y1_2)
    xc2 = max(x2, x2_2)
    yc2 = max(y2, y2_2)

    # 计算最小闭包区域对角线的平方距离 (c^2)
    diagonal_distance_sq = (xc2 - xc1)**2 + (yc2 - yc1)**2
    if diagonal_distance_sq == 0:
        return iou_value # 如果闭包面积为0，退化为IoU

    # 计算DIoU惩罚项
    diou_penalty = center_distance_sq / diagonal_distance_sq

    # 计算长宽比 v
    arctan1 = math.atan(w1 / (h1 + eps))
    arctan2 = math.atan(w2 / (h2 + eps))
    v = (4 / (math.pi**2)) * ((arctan1 - arctan2)**2)

    # 计算 alpha
    # 添加 eps 防止 iou_value 为 1 时分母为 0
    alpha = v / (1 - iou_value + v + eps)

    # 计算CIoU
    ciou_value = iou_value - (diou_penalty + alpha * v)
    return ciou_value


def calculate_iou(bbox1, bbox2, style='ciou'):
    
    

    if style == 'giou':
        return giou(bbox1, bbox2)
    elif style == 'diou':
        return diou(bbox1, bbox2)
    elif style == 'ciou':
        return ciou(bbox1, bbox2)
    else:
        return miou(bbox1, bbox2)