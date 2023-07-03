import os
from pycocotools.coco import COCO
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


json_path = "/home/nm6114091/Python/Gaze_DataProcessing/datasets/VOC_format_MPIIFaceGaze/gaze_point.json"
img_path = "/home/nm6114091/Python/Gaze_DataProcessing/datasets/VOC_format_MPIIFaceGaze/image"


coco = COCO(annotation_file = json_path )

ids = list(sorted(coco.imgs.keys()))
print("number of images = {}".format(len(ids)))



coco_classes = dict([(value["id"],value["name"]) for keys , value in coco.cats.items()])


for img_id in ids[:3]:
    
    ann_ids = coco.getAnnIds(imgIds=img_id)
    
    targets = coco.loadAnns(ann_ids)
    
    path = coco.loadImgs(img_id)[0]['file_name']
    print(path)

    img = Image.open(os.path.join(img_path,path)).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    white = Image.new("RGB", (500, 100), "white")
    
    for target in targets:
        x,y = target["gazepoint"]
        # draw.point((x,y), fill = (255, 0, 0))
        # draw.text((x,y-10),"Point", fill = (255,0,0))
        cameraMatrix,distCoeffs = target["camera"]
        rvects,tvecs = target["monitorPose"]
        unit_mm,unit_pixel = target["screenSize"]
        white.point((x,y), fill = (255, 0, 0))

        
    plt.imshow(img)
    plt.show()
    






