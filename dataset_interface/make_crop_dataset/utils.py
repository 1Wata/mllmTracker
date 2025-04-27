import random
from PIL import Image
from PIL import ImageDraw



def convert_bbox_format(bbox):
    """
    将[x, y, w, h]格式的边界框转换为[x1, y1, x2, y2]格式。
    
    Args:
        bbox: [x, y, w, h]格式的边界框
        
    Returns:
        [x1, y1, x2, y2]格式的边界框
    """
    if len(bbox) != 4:
        raise ValueError(f"边界框格式错误: {bbox}")
        
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def jitter_bbox(bbox, jitter_scale=0.1):
    """
    对边界框的中心点进行抖动，保持宽高基本不变
    """
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    
    # 只对中心点进行抖动
    jitter_cx = random.uniform(-jitter_scale * w, jitter_scale * w)
    jitter_cy = random.uniform(-jitter_scale * h, jitter_scale * h)
    
    # 允许宽高有极小的变化
    jitter_w = random.uniform(-jitter_scale * w * 0.1, jitter_scale * w * 0.1)
    jitter_h = random.uniform(-jitter_scale * h * 0.1, jitter_scale * h * 0.1)
    
    new_center_x = center_x + jitter_cx
    new_center_y = center_y + jitter_cy
    new_w = max(2, w + jitter_w)  # 确保宽高至少为2
    new_h = max(2, h + jitter_h)
    
    new_x1 = int(new_center_x - new_w / 2)
    new_y1 = int(new_center_y - new_h / 2)
    new_x2 = int(new_center_x + new_w / 2)
    new_y2 = int(new_center_y + new_h / 2)
    
    return [new_x1, new_y1, new_x2, new_y2]

def crop_and_pad_template(img, bbox, scale=2.0, resize=320, return_bbox=False):
    if isinstance(img, str):
        img = Image.open(img)
    w, h = img.size
    crop_size = scale * max(bbox[2] - bbox[0], bbox[3] - bbox[1])

    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2

    # 改进的裁剪区域大小调整逻辑
    dist_to_left = center_x
    dist_to_right = w - center_x
    dist_to_top = center_y
    dist_to_bottom = h - center_y
    
    max_crop_half_size = min(dist_to_left, dist_to_right, dist_to_top, dist_to_bottom)
    
    # 如果裁剪区域太大，调整大小
    if crop_size/2 > max_crop_half_size:
        # 尝试保持裁剪区域在图像内，同时尽量接近原始比例
        crop_size = min(crop_size, 2 * max(dist_to_left, dist_to_right, dist_to_top, dist_to_bottom))
        # 如果裁剪区域仍然很大，退化为使用整个图像
        if crop_size > 1.5 * min(w, h):
            crop_size = min(w, h)

    left = int(center_x - crop_size / 2)
    top = int(center_y - crop_size / 2)
    right = int(center_x + crop_size / 2)
    bottom = int(center_y + crop_size / 2)

    # Calculate padding if needed
    pad_left = abs(min(0, left))
    pad_top = abs(min(0, top))
    pad_right = abs(min(0, w - right))
    pad_bottom = abs(min(0, h - bottom))

    # Adjust crop coordinates to be within image boundaries
    left = max(0, left)
    top = max(0, top)
    right = min(w, right)
    bottom = min(h, bottom)

    # Crop the image
    cropped_img = img.crop((left, top, right, bottom))

    # Pad the image if needed
    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        width = int(right - left + pad_left + pad_right)
        height = int(bottom - top + pad_top + pad_bottom)
        padded_img = Image.new(cropped_img.mode, (width, height))
        padded_img.paste(cropped_img, (int(pad_left), int(pad_top)))
        img = padded_img
    else:
        img = cropped_img
    
    if resize is not None:
        img = img.resize((int(resize), int(resize)), Image.BILINEAR)
    if return_bbox:
        new_bbox = convert_bbox_into_cropped_img([left, top, right, bottom], bbox, resize)
        return img, new_bbox
    else:
        return img


def crop_and_pad_search(img, prev_bbox, abs_bbox, scale=2.0, resize=320):
    if isinstance(img, str):
        img = Image.open(img)
    w, h = img.size
    crop_size = scale * max(prev_bbox[2] - prev_bbox[0], prev_bbox[3] - prev_bbox[1])

    center_x = (prev_bbox[0] + prev_bbox[2]) / 2
    center_y = (prev_bbox[1] + prev_bbox[3]) / 2

    # 改进的裁剪区域大小调整逻辑
    dist_to_left = center_x
    dist_to_right = w - center_x
    dist_to_top = center_y
    dist_to_bottom = h - center_y
    
    max_crop_half_size = min(dist_to_left, dist_to_right, dist_to_top, dist_to_bottom)
    
    # 如果裁剪区域太大，调整大小
    if crop_size/2 > max_crop_half_size:
        # 尝试保持裁剪区域在图像内，同时尽量接近原始比例
        crop_size = min(crop_size, 2 * max(dist_to_left, dist_to_right, dist_to_top, dist_to_bottom))
        # 如果裁剪区域仍然很大，退化为使用整个图像
        if crop_size > 1.5 * min(w, h):
            crop_size = min(w, h)

    left = int(center_x - crop_size / 2)
    top = int(center_y - crop_size / 2)
    right = int(center_x + crop_size / 2)
    bottom = int(center_y + crop_size / 2)
    
    # 记录原始裁剪区域坐标
    crop_region = [left, top, right, bottom]
    
    # 检查目标是否完全在裁剪区域内
    
    
    # 计算调整后的裁剪区域 (考虑边界)
    adjusted_left = max(0, left)
    adjusted_top = max(0, top)
    adjusted_right = min(w, right)
    adjusted_bottom = min(h, bottom)
    
    # 检查目标是否完全在裁剪区域内
    if abs_bbox is not None and len(abs_bbox) == 4:
        target_x1, target_y1, target_x2, target_y2 = abs_bbox
        if (target_x1 < adjusted_left or target_y1 < adjusted_top or
            target_x2 > adjusted_right or target_y2 > adjusted_bottom):
            raise ValueError("目标不完全在裁剪区域内")
    
    # Calculate padding if needed
    pad_left = abs(min(0, left))
    pad_top = abs(min(0, top))
    pad_right = abs(min(0, w - right))
    pad_bottom = abs(min(0, h - bottom))
    
    # Adjust crop coordinates to be within image boundaries
    left = adjusted_left
    top = adjusted_top
    right = adjusted_right
    bottom = adjusted_bottom
    
    cropped_img = img.crop((left, top, right, bottom))

    # Pad the image if needed
    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        width = int(right - left + pad_left + pad_right)
        height = int(bottom - top + pad_top + pad_bottom)
        padded_img = Image.new(cropped_img.mode, (width, height))
        padded_img.paste(cropped_img, (int(pad_left), int(pad_top)))
        img = padded_img
    else:
        img = cropped_img
        
    if resize is not None:
        img = img.resize((int(resize), int(resize)), Image.BILINEAR)

    if abs_bbox is not None:
        new_gt_bbox = convert_bbox_into_cropped_img(crop_region, abs_bbox, resize)
        return img, new_gt_bbox, crop_region, resize  # 返回裁剪区域坐标和resized_size
    else:
        return img, None, crop_region, resize


def convert_bbox_into_cropped_img(resize_bbox, abs_bbox, resized_size=320):
    """
    将原始图像中的边界框坐标转换为裁剪并缩放后的图像中的坐标。
    
    Args:
        resize_bbox (list): 裁剪区域在原始图像中的坐标 [left, top, right, bottom]
        abs_bbox (list): 目标在原始图像中的坐标 [x1, y1, x2, y2]
        resized_size (float): 裁剪并调整大小后的输出尺寸(边长)
        
    Returns:
        list: 在裁剪缩放后图像中的坐标 [x1', y1', x2', y2']
    """
    # 提取裁剪区域的坐标
    crop_left, crop_top, crop_right, crop_bottom = resize_bbox
    
    # 原始裁剪区域的实际尺寸
    crop_width = crop_right - crop_left
    crop_height = crop_bottom - crop_top
    
    # 如果裁剪区域无效，返回 [0,0,0,0]
    if crop_width <= 0 or crop_height <= 0:
        return [0, 0, 0, 0]
    
    # 获取目标框坐标
    x1, y1, x2, y2 = abs_bbox
    
    # 计算目标框相对于裁剪区域的坐标
    rel_x1 = x1 - crop_left
    rel_y1 = y1 - crop_top
    rel_x2 = x2 - crop_left
    rel_y2 = y2 - crop_top
    
    # 计算缩放比例 (从实际裁剪尺寸到最终输出尺寸)
    scale_x = resized_size / crop_width
    scale_y = resized_size / crop_height
    
    # 应用缩放
    new_x1 = rel_x1 * scale_x
    new_y1 = rel_y1 * scale_y
    new_x2 = rel_x2 * scale_x
    new_y2 = rel_y2 * scale_y
    
    # 裁剪到输出图像范围内 [0, resized_size]
    new_x1 = max(0, min(resized_size, new_x1))
    new_y1 = max(0, min(resized_size, new_y1))
    new_x2 = max(0, min(resized_size, new_x2))
    new_y2 = max(0, min(resized_size, new_y2))
    
    # 确保 x1 <= x2, y1 <= y2
    if new_x1 >= new_x2 or new_y1 >= new_y2:
        return [0, 0, 0, 0]  # 无效框
        
    return [new_x1, new_y1, new_x2, new_y2]


def convert_bbox_from_cropped_img(resize_bbox, cropped_bbox, resized_size=320):
    """
    将裁剪图像中的边界框坐标转换回原始图像中的坐标 (convert_bbox_into_cropped_img的逆操作)。
    
    Args:
        resize_bbox (list): 裁剪区域在原始图像中的坐标 [left, top, right, bottom]
        cropped_bbox (list): 目标在裁剪图像中的坐标 [x1', y1', x2', y2']
        resized_size (float): 裁剪并调整大小后的输出尺寸(边长)
        
    Returns:
        list: 在原始图像中的坐标 [x1, y1, x2, y2]
    """
    # 提取裁剪区域的坐标
    crop_left, crop_top, crop_right, crop_bottom = resize_bbox
    
    # 原始裁剪区域的实际尺寸
    crop_width = crop_right - crop_left
    crop_height = crop_bottom - crop_top
    
    # 如果裁剪区域无效或边界框无效，返回 [0,0,0,0]
    if crop_width <= 0 or crop_height <= 0 or not cropped_bbox:
        return [0, 0, 0, 0]
    
    x1_c, y1_c, x2_c, y2_c = cropped_bbox
    
    # 验证裁剪图像中的坐标是否有效
    if x1_c >= x2_c or y1_c >= y2_c or x1_c < 0 or y1_c < 0 or x2_c > resized_size or y2_c > resized_size:
        return [0, 0, 0, 0]  # 无效坐标
    
    # 计算缩放比例 (从最终输出尺寸到实际裁剪尺寸)
    scale_x = crop_width / resized_size
    scale_y = crop_height / resized_size
    
    # 反向应用缩放，获取在裁剪窗口坐标系中的位置
    rel_x1 = x1_c * scale_x
    rel_y1 = y1_c * scale_y
    rel_x2 = x2_c * scale_x
    rel_y2 = y2_c * scale_y
    
    # 加上裁剪窗口在原图中的偏移，得到原图坐标
    abs_x1 = rel_x1 + crop_left
    abs_y1 = rel_y1 + crop_top
    abs_x2 = rel_x2 + crop_left
    abs_y2 = rel_y2 + crop_top
    
    return [abs_x1, abs_y1, abs_x2, abs_y2]



if __name__ == "__main__":
    template_img = "/data1/lihaobo/tracking/data/lasot/airplane/airplane-1/img/00000001.jpg"
    # template_img = "/data1/lihaobo/tracking/data/lasot/person/person-1/img/00000050.jpg"
    search_img = "/data1/lihaobo/tracking/data/lasot/airplane/airplane-1/img/00000041.jpg"
    # search_img = "/data1/lihaobo/tracking/data/lasot/person/person-1/img/00000100.jpg"
    template_bbox = [367, 101, 402, 117]  # Example bounding box
    # template_bbox = [505, 234, 505+178, 234+314]
    template_bbox = jitter_bbox(template_bbox, jitter_scale=0.2)
    print("Jittered template bbox:", template_bbox)
    search_bbox = [350, 137, 398, 154]  # Example bounding box
    # search_bbox = [582, 201, 582+119, 201+502]

    scale = 1.5
    search_scale = 3.0  # 搜索区域使用的缩放因子

    cropped_template_img = crop_and_pad_template(template_img, template_bbox, scale)
    cropped_search_img, new_bbox, crop_region, resized_size = crop_and_pad_search(
        search_img, template_bbox, search_bbox, scale=search_scale
    )
    cropped_template_img.save("cropped_template_image.jpg")
    cropped_search_img.save("cropped_search_image.jpg")
    print("New bbox in cropped image:", new_bbox)

    draw_crop = ImageDraw.Draw(cropped_search_img)
    draw_crop.rectangle(
        [(new_bbox[0], new_bbox[1]), (new_bbox[2], new_bbox[3])],
        outline="red",
        width=2
    )
    cropped_search_img.save("cropped_search_with_bbox.jpg")
    
    # 现在使用正确的参数
    abs_bbox_recovered = convert_bbox_from_cropped_img(crop_region, new_bbox, resized_size)
    print("原始搜索框坐标:", search_bbox)
    print("恢复后的坐标:", abs_bbox_recovered)
    
    original_search = Image.open(search_img)
    draw = ImageDraw.Draw(original_search)
    # 绘制恢复后的边界框(红色)
    draw.rectangle(
        [(abs_bbox_recovered[0], abs_bbox_recovered[1]), (abs_bbox_recovered[2], abs_bbox_recovered[3])],
        outline="red",
        width=2
    )
    original_search.save("search_image_with_bbox.jpg")