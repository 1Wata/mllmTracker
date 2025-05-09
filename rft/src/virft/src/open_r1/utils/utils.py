import math

def transform_bbox(bbox, original_size, resized_size, direction='resized_to_original'):
    """Transforms bounding box coordinates between original and resized image spaces."""
    if bbox is None or original_size is None or resized_size is None:
        return None

    orig_w, orig_h = original_size
    res_w, res_h = resized_size

    # Avoid division by zero or invalid sizes
    if orig_w <= 0 or orig_h <= 0 or res_w <= 0 or res_h <= 0:
        print(f"Warning: Invalid image sizes for transform_bbox. Original: {original_size}, Resized: {resized_size}")
        return None # Or return original bbox, depending on desired behavior

    x1, y1, x2, y2 = bbox

    if direction == 'resized_to_original':
        # Handle potential zero resized dimensions gracefully
        scale_x = orig_w / res_w if res_w > 0 else 1
        scale_y = orig_h / res_h if res_h > 0 else 1
    elif direction == 'original_to_resized':
        scale_x = res_w / orig_w if orig_w > 0 else 1
        scale_y = res_h / orig_h if orig_h > 0 else 1
    else:
        raise ValueError("Invalid direction for transform_bbox. Use 'resized_to_original' or 'original_to_resized'.")

    new_x1 = x1 * scale_x
    new_y1 = y1 * scale_y
    new_x2 = x2 * scale_x
    new_y2 = y2 * scale_y

    # Clamp coordinates to the target image bounds
    target_w, target_h = original_size if direction == 'resized_to_original' else resized_size
    new_x1 = max(0, min(new_x1, target_w - 1))
    new_y1 = max(0, min(new_y1, target_h - 1))
    new_x2 = max(0, min(new_x2, target_w - 1))
    new_y2 = max(0, min(new_y2, target_h - 1))

    # Ensure x1 <= x2 and y1 <= y2
    final_x1 = min(new_x1, new_x2)
    final_y1 = min(new_y1, new_y2)
    final_x2 = max(new_x1, new_x2)
    final_y2 = max(new_y1, new_y2)


    return [final_x1, final_y1, final_x2, final_y2]




def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
):
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.

    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar



def extract_single_bbox(response):
    """
    从模型响应中提取单个边界框，自动适应多种响应格式
    
    Args:
        response (str): 模型的响应文本
        
    Returns:
        list: 提取的边界框坐标 [x1, y1, x2, y2]，如果无法提取则返回 None
    """
    # 尝试从不同格式中提取内容
    content_str = None
    
    # 检查是否有 thinking/answer 格式
    thinking_start = "<thinking>"
    thinking_end = "</thinking>"
    answer_start = "<answer>"
    answer_end = "</answer>"
    
    # 如果存在 thinking 和 answer 标签
    if thinking_start in response and answer_start in response:
        # 提取 answer 部分
        start_idx = response.find(answer_start) + len(answer_start)
        end_idx = response.find(answer_end) if answer_end in response else len(response)
        content_str = response[start_idx:end_idx].strip()
    
    # 如果只有 answer 标签
    elif answer_start in response:
        start_idx = response.find(answer_start) + len(answer_start)
        end_idx = response.find(answer_end) if answer_end in response else len(response)
        content_str = response[start_idx:end_idx].strip()
    
    # 如果没有标签，则尝试直接提取
    else:
        content_str = response.strip()
    
    # 如果没有内容可提取
    if not content_str:
        return None
    
    # 替换单引号为双引号以兼容JSON格式
    content_str = content_str.replace("'", '"')
    
    # 尝试解析为JSON格式
    # 方法1: 直接的坐标列表 [x1, y1, x2, y2]
    if content_str.startswith('[') and content_str.endswith(']'):
        # 尝试解析为JSON数组
        import json
        
        # 尝试解析整个内容
        bbox_data = None
        try:
            bbox_data = json.loads(content_str)
        except json.JSONDecodeError:
            # 如果解析失败，继续尝试下一种方法
            pass
        
        if bbox_data is not None:
            # 检查是直接的坐标列表还是带有Position键的对象列表
            if isinstance(bbox_data, list):
                if len(bbox_data) == 4 and all(isinstance(x, (int, float)) for x in bbox_data):
                    # 直接返回坐标列表 [x1, y1, x2, y2]
                    return bbox_data
                elif bbox_data and isinstance(bbox_data[0], dict) and 'Position' in bbox_data[0]:
                    # 返回第一个边界框的Position
                    return bbox_data[0]['Position']
    
    # 方法2: 尝试解析为字典形式 {'Position': [x1, y1, x2, y2]}
    if content_str.startswith('{') and content_str.endswith('}'):
        import json
        
        bbox_dict = None
        try:
            bbox_dict = json.loads(content_str)
        except json.JSONDecodeError:
            # 如果解析失败，继续尝试下一种方法
            pass
        
        if bbox_dict is not None and isinstance(bbox_dict, dict) and 'Position' in bbox_dict:
            return bbox_dict['Position']
    
    # 方法3: 使用正则表达式提取数字列表
    import re
    matches = re.findall(r'\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]', content_str)
    if matches:
        return [int(x) for x in matches[0]]
    
    return None