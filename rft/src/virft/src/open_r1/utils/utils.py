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