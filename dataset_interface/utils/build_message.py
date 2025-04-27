import os
from PIL import Image
from typing import Optional, List, Dict, Union


def build_sft_message(dataset_name, root_dir=None, exp_str=None, search_img=None,
                  template_imgs=None, normalized_search_bbox=None,
                  normalized_template_bboxes=None, search_visible=None, search_valid=None):
    """
    构建数据集条目的消息结构。
    
    参数:
        dataset_name (str): 数据集名称
        root_dir (str): 图像根目录
        exp_str (str): 类别/对象名称
        search_img (str | Image.Image): 搜索图像的路径或PIL Image对象
        template_imgs (list[str] | list[Image.Image]): 模板图像路径或PIL Image对象列表
        normalized_search_bbox (list, optional): 归一化边界框坐标 [x1, y1, x2, y2]
        normalized_template_bboxes (list, optional): 归一化模板边界框列表
        search_visible (bool, optional): 对象在搜索图像中是否可见
        search_valid (bool, optional): 对象在搜索图像中是否有效（在画面内）
        
    返回:
        dict: 数据集的消息结构和图像路径/对象
    """

    images = []



    for img in template_imgs:
        if isinstance(img, str):
            images.append(os.path.join(root_dir, img))
        else:
            images.append(img)
    
    if isinstance(search_img, str):
        images.append(os.path.join(root_dir, search_img))
    else:
        images.append(search_img)
    
    # 为了生成正确的消息内容，需要构建图像引用
    template_refs = []
    for i, img in enumerate(template_imgs):
        if isinstance(img, str):
            template_refs.append(os.path.basename(img))
        else:
            template_refs.append(f"template_image_{i+1}")
            
    search_ref = os.path.basename(search_img) if isinstance(search_img, str) else "search_image"
    

    instruction = (f"You are an AI assistant for single object tracking. Given the template "
                  f"images showing '{exp_str}', identify and locate this object in the "
                  f"provided search image.")
    

    template_image_info = ""
    for idx, ref in enumerate(template_refs):
        template_image_info += f" <image>{ref}"

        if normalized_template_bboxes and idx < len(normalized_template_bboxes) and normalized_template_bboxes[idx]:
            norm_bbox = normalized_template_bboxes[idx]
            template_image_info += (f" (Template image {idx+1} has normalized bounding box: "
                                   f"[{norm_bbox[0]}, {norm_bbox[1]}, "
                                   f"{norm_bbox[2]}, {norm_bbox[3]}])")
    

    if dataset_name == 'lasot':
        task_instruction = (" If the object is present and visible in the search image, "
                           "provide its bounding box coordinates in the format [x1, y1, x2, y2]. "
                           "If the object is present but occluded, provide an estimated bounding box. "
                           "If the object is not visible or not present in the search image, "
                           "explicitly state that.")
    else:
        task_instruction = (" If the object is present and visible in the search image, "
                           "provide its bounding box coordinates in the format [x1, y1, x2, y2]. "
                           "If the object is not visible or not present in the search image, "
                           "explicitly state that.")
    

    user_content = f"{instruction} Template Images:{template_image_info} Search Image: <image>{search_ref}{task_instruction}"
    

    if dataset_name == 'lasot':
        assert isinstance(normalized_search_bbox, list) and len(normalized_search_bbox) == 4

        norm_bbox = normalized_search_bbox
                    

        if search_visible:
            output_answer = (f"The object of category '{exp_str}' is visible in the search image, "
                            f"with a bounding box at [{norm_bbox[0]}, {norm_bbox[1]}, "
                            f"{norm_bbox[2]}, {norm_bbox[3]}].")
        elif search_valid:
            output_answer = (f"The object of category '{exp_str}' is present in the search image "
                            f"but occluded, with an estimated bounding box at [{norm_bbox[0]}, "
                            f"{norm_bbox[1]}, {norm_bbox[2]}, {norm_bbox[3]}].")
        else:
            output_answer = (f"The object of category '{exp_str}' has left the frame and "
                            f"is not present in the search image.")
    else:

        if search_visible and normalized_search_bbox:
            norm_bbox = normalized_search_bbox
            output_answer = (f"The object of category '{exp_str}' is visible in the search image, "
                            f"with a bounding box at [{norm_bbox[0]}, {norm_bbox[1]}, "
                            f"{norm_bbox[2]}, {norm_bbox[3]}].")
        else:
            output_answer = f"The object of category '{exp_str}' is not visible in the search image."

    message = {
        "messages": [
            {
                "content": user_content,
                "role": "user"
            },
            {
                "content": output_answer,
                "role": "assistant"
            }
        ],
        "images": images
    }
    
    return message




def build_input_message(
    dataset_name: str,
    exp_str: str,
    search_img: Union[str, Image.Image],
    template_imgs: List[Union[str, Image.Image]],
    normalized_template_bboxes: Optional[List] = None
) -> List[Dict]:
    """
    构建符合视觉信息处理要求的数据集消息结构。
    
    参数:
        dataset_name (str): 数据集名称
        exp_str (str): 类别/对象名称
        search_img (Union[str, Image.Image]): 搜索图像的路径或PIL Image对象
        template_imgs (List[Union[str, Image.Image]]): 模板图像路径或PIL Image对象列表
        normalized_template_bboxes (Optional[List]): 归一化模板边界框列表
        
    返回:
        List[Dict]: 直接包含对话消息的列表（不含外层的messages键）
    """
    # 构建视觉内容列表（图像和文本字典）
    content = []
    if exp_str and exp_str.endswith('\n'):
        exp_str = exp_str.rstrip('\n')
        
    # 构建基础指令文本
    instruction = (
        f"You are an AI assistant for single object tracking. Given the template "
        f"images showing '{exp_str}', identify and locate this object in the "
        f"provided search image."
    )
    content.append({"type": "text", "text": instruction})
    
    # 添加模板图像和其对应的边界框信息
    for idx, img in enumerate(template_imgs):
        # 添加图像
        content.append({"type": "image", "image": img})
        
        # 如果有对应的边界框信息，紧随图像后添加
        if normalized_template_bboxes and idx < len(normalized_template_bboxes) and normalized_template_bboxes[idx]:
            norm_bbox = normalized_template_bboxes[idx]
            bbox_text = (
                f"Template image {idx+1} has normalized bounding box: "
                f"[{norm_bbox[0]}, {norm_bbox[1]}, {norm_bbox[2]}, {norm_bbox[3]}]."
            )
            content.append({"type": "text", "text": bbox_text})
    
    # 添加搜索图像
    content.append({"type": "image", "image": search_img})
    
    # 添加任务说明
    if dataset_name == 'lasot':
        task_instruction = (
            "If the object is present and visible in the search image, "
            "provide its bounding box coordinates in the format [x1, y1, x2, y2]. "
            "If the object is present but occluded, provide an estimated bounding box. "
            "If the object is not visible or not present in the search image, "
            "explicitly state that."
        )
    else:
        task_instruction = (
            "If the object is present and visible in the search image, "
            "provide its bounding box coordinates in the format [x1, y1, x2, y2]. "
            "If the object is not visible or not present in the search image, "
            "explicitly state that."
        )
    content.append({"type": "text", "text": task_instruction})
    
    # 直接返回对话消息列表（不包含外层的messages键）
    return [
        {
            "content": content,
            "role": "user"
        }
    ]




