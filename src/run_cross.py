import subprocess

# 要執行的檔案及對應的參數
file = "main.py"
file_params = [
    "ctdet_gaze --exp_id gaze_resdcn18_csp_kr_resize_pl01_p04 --arch resdcn_18 --dataset mpiifacegaze --keep_res --resize_raw_image --camera_screen_pos --pog_offset --pog_weight 0.1 --num_epochs 70 --lr_step 45,60 --heat_map_debug --data_train_person_id 4 --data_train_person_id 4",
    "ctdet_gaze --exp_id gaze_resdcn18_csp_kr_resize_pl01_p05 --arch resdcn_18 --dataset mpiifacegaze --keep_res --resize_raw_image --camera_screen_pos --pog_offset --pog_weight 0.1 --num_epochs 70 --lr_step 45,60 --heat_map_debug --data_train_person_id 5 --data_train_person_id 5",
    "ctdet_gaze --exp_id gaze_resdcn18_csp_kr_resize_pl01_p12 --arch resdcn_18 --dataset mpiifacegaze --keep_res --resize_raw_image --camera_screen_pos --pog_offset --pog_weight 0.1 --num_epochs 70 --lr_step 45,60 --heat_map_debug --data_train_person_id 12 --data_train_person_id 12"

]

    # "ctdet_gaze --exp_id gaze_resdcn18_csp_p04 --arch resdcn_18 --dataset mpiifacegaze --camera_screen_pos --num_epochs 70 --lr_step 45,60 --heat_map_debug --data_person_id  4",
    # "ctdet_gaze --exp_id gaze_resdcn18_csp_p12 --arch resdcn_18 --dataset mpiifacegaze --camera_screen_pos --num_epochs 70 --lr_step 45,60 --heat_map_debug --data_person_id 12",
    # "ctdet_gaze --exp_id gaze_resdcn18_csp_kr_resize_p04 --arch resdcn_18 --dataset mpiifacegaze --keep_res --resize_raw_image --camera_screen_pos --num_epochs 70 --lr_step 45,60 --heat_map_debug --data_person_id  4",
    # "ctdet_gaze --exp_id gaze_resdcn18_csp_kr_resize_p12 --arch resdcn_18 --dataset mpiifacegaze --keep_res --resize_raw_image --camera_screen_pos --num_epochs 70 --lr_step 45,60 --heat_map_debug --data_person_id 12",
    # "ctdet_gaze --exp_id gaze_resdcn18_csp_kr_resize_pl001_p04 --arch resdcn_18 --dataset mpiifacegaze --keep_res --resize_raw_image --camera_screen_pos --pog_offset --pog_weight 0.01 --num_epochs 70 --lr_step 45,60 --heat_map_debug --data_person_id  4",
    # "ctdet_gaze --exp_id gaze_resdcn18_csp_kr_resize_pl001_p12 --arch resdcn_18 --dataset mpiifacegaze --keep_res --resize_raw_image --camera_screen_pos --pog_offset --pog_weight 0.01 --num_epochs 70 --lr_step 45,60 --heat_map_debug --data_person_id 12"

# 使用迴圈依次執行每個檔案及其參數
for params in file_params:
    # 在這裡加入你希望執行的程式碼，file代表檔案名稱，params代表參數
    print(f"執行檔案: {file}，參數: {params}")
    # 示範如何執行檔案並將參數傳遞給它
    subprocess.run(["python", file] + params.split())