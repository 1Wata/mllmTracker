from PIL import Image
from PIL import ImageDraw


img = Image.open("/data1/lihaobo/tracking/data/cropped_sft/cropped_images/sample_000001/search.jpg")
bbox = [105, 5, 216, 232]
draw = ImageDraw.Draw(img)
draw.rectangle(bbox, outline="red", width=3)
img.save("image_with_bbox.jpg")
