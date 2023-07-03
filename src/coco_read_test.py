import os
from pycocotools.coco import COCO
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


# json_path = "/home/nm6114091/share_storage/dataset/COCO/annotations/instances_val2017.json"
# img_path = "/home/nm6114091/share_storage/dataset/COCO/val2017"

json_path = "/home/nm6114091/share_storage/dataset/VOC_COCO/annotations/pascal_test2020.json"
img_path = "/home/nm6114091/share_storage/dataset/VOC_COCO/val"


coco = COCO(annotation_file = json_path )

ids = list(sorted(coco.imgs.keys()))
print("number of images = {}".format(len(ids)))



coco_classes = dict([(value["id"],value["name"]) for keys , value in coco.cats.items()])


for img_id in ids[:3]:
    
    ann_ids = coco.getAnnIds(imgIds=img_id)
    
    targets = coco.loadAnns(ann_ids)
    
    path = coco.loadImgs(img_id)[0]['file_name']
    
    
    img = Image.open(os.path.join(img_path,path)).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    for target in targets:
        x,y,w,h = target["bbox"]
        x1,y1,x2,y2 = x,y, int(x+w),int(y+h)
        draw.rectangle((x1,y1,x2,y2))
        draw.text((x1,y1-10),coco_classes[target["category_id"]])
        
    plt.imshow(img)
    plt.show()
    






