import subprocess

# 要執行的檔案及對應的參數
file = "main.py"
file_params = [
    # "ctdet_gaze --exp_id gaze_resdcncut_18_mpii_ep70_all_norm_p05_2 --arch resdcncut_18 --down_ratio 2 --dataset mpiifacegaze --num_epochs 70 --lr_step 45,60 --vp_pixel_per_mm 5  --data_train_person_id  5 --data_test_person_id  5",
    # "ctdet_gaze --exp_id gaze_effv2s_mpii_ep70_all_norm_p08 --arch effv2s --dataset mpiifacegaze --num_epochs 70 --lr_step 45,60 --vp_pixel_per_mm 5 --data_train_person_id  8 --data_test_person_id  8",
    # "ctdet_gaze --exp_id gaze_effv2s_mpii_ep70_all_norm_p10 --arch effv2s --dataset mpiifacegaze --num_epochs 70 --lr_step 45,60 --vp_pixel_per_mm 5 --data_train_person_id 10 --data_test_person_id 10",
    "ctdet_gaze --exp_id gaze_resdcncut_18_mpii_tr_sc_r10_ep70_all_norm_csp_kr_pl001_p05 --arch resdcncut_18 --dataset mpiifacegaze --num_epochs 60 --lr 5e-3 --lr_step 10,30,50 --keep_res --resize_raw_image --camera_screen_pos --vp_h 2100 --vp_w 3360 --vp_pixel_per_mm 5 --pog_offset --pog_weight 0.01 --data_train_person_id  5 --data_test_person_id  5",
    # "ctdet_gaze --exp_id gaze_effv2s_mpii_ep70_all_norm_csp_kr_pl001_p08 --arch effv2s --dataset mpiifacegaze --num_epochs 70 --lr_step 45,60 --keep_res --resize_raw_image --camera_screen_pos --vp_h 2100 --vp_w 3360 --vp_pixel_per_mm 5 --pog_offset --pog_weight 0.01 --data_train_person_id  8 --data_test_person_id  8",
    # "ctdet_gaze --exp_id gaze_effv2s_mpii_ep70_all_norm_csp_kr_pl001_p10 --arch effv2s --dataset mpiifacegaze --num_epochs 70 --lr_step 45,60 --keep_res --resize_raw_image --camera_screen_pos --vp_h 2100 --vp_w 3360 --vp_pixel_per_mm 5 --pog_offset --pog_weight 0.01 --data_train_person_id 10 --data_test_person_id 10",

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