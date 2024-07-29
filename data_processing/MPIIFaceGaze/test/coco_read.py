import os
from pycocotools.coco import COCO
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import cv2


json_path = "/home/nm6114091/Python/Gaze_DataProcessing/datasets/VOC_format_MPIIFaceGaze/annotation/gaze_point.json"
img_dir = "/home/nm6114091/Python/Gaze_DataProcessing/datasets/VOC_format_MPIIFaceGaze/image"


coco = COCO(annotation_file = json_path )

ids = list(sorted(coco.imgs.keys()))
print("number of images = {}".format(len(ids)))



coco_classes = dict([(value["id"],value["name"]) for keys , value in coco.cats.items()])

max_objs = 50
for img_id in ids[:3]:
    
    ann_ids = coco.getAnnIds(imgIds=img_id)
    
    targets = coco.loadAnns(ann_ids)
    num_objs = min(len(targets), max_objs)
    print(len(targets))
    
    file_name = coco.loadImgs(img_id)[0]['file_name']
    img_path = os.path.join(img_dir, file_name)
    
    # img = cv2.imread(img_path)
    
    white = np.full((960,1440,3), 255).astype(np.uint8)
    
    for target in targets:
        x,y = target["gazepoint"]
        # draw.point((x,y), fill = (255, 0, 0))
        # draw.text((x,y-10),"Point", fill = (255,0,0))
        cameraMatrix,distCoeffs = target["camera"]
        rvects,tvecs = target["monitorPose"]
        unit_mm,unit_pixel = target["screenSize"]
        print(x,y)
        point_size = 5  # 设置点的大小

        # 绘制点
        cv2.circle(white, (x, y), point_size, (0, 0, 255), -1)
        
    image_rgb = cv2.cvtColor(white, cv2.COLOR_BGR2RGB)

    # 使用plt显示图像
    plt.imshow(image_rgb)
    plt.axis('off')  # 关闭坐标轴
    plt.show()


    






