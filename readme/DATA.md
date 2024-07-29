# Dataset preparation



### COCO
- Download the images (2017 Train, 2017 Val, 2017 Test) from [coco website](http://cocodataset.org/#download).
- Download annotation files (2017 train/val and test image info) from [coco website](http://cocodataset.org/#download). 
- Place the data (or create symlinks) to make the data folder like:

  ~~~
  ${CenterNet_ROOT}
  |-- data
  `-- |-- coco
      `-- |-- annotations
          |   |-- instances_train2017.json
          |   |-- instances_val2017.json
          |   |-- person_keypoints_train2017.json
          |   |-- person_keypoints_val2017.json
          |   |-- image_info_test-dev2017.json
          |---|-- train2017
          |---|-- val2017
          `---|-- test2017
  ~~~

- [Optional] If you want to train ExtremeNet, generate extreme point annotation from segmentation:
    
    ~~~
    cd $CenterNet_ROOT/tools/
    python gen_coco_extreme_points.py
    ~~~
  It generates `instances_extreme_train2017.json` and `instances_extreme_val2017.json` in `data/coco/annotations/`. 


## Plot data
Due to GitHub's data limitations, I have placed the data for plotting dataset distribution on AISLab-NAS  
[Plot data](https://aislabnas.ee.ncku.edu.tw/sharing/XwKB0XgZL)

~~~
/Data-Weight/Gaze/EVG/plot_data/  
└── violin_chart/  
    └── EVE_data
    └── GazeCapture_data
    └── MPIIFaceGaze_data
    └── MPIIFaceGaze_dataset_original
~~~

