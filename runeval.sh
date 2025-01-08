
# Activate Conda environment
# conda activate CenterNet_38

# Change directory to the evaluation script location
cd src/tools/mpiifacegaze_eval

# Run the Python script with specified arguments
python live.py ctdet_gaze --arch resdcnface_18 --keep_res --resize_raw_image --resize_raw_image_h 270 --resize_raw_image_w 480 --vp_h 2160 --vp_w 3840

