import subprocess

# 要執行的檔案及對應的參數
file = "main.py"
file_params = [
    # "ctdet_gaze --exp_id gaze_resdcn18_mpii_ep70_all_norm_p08_estop_reg_test --arch resdcn_18 --dataset mpiifacegaze --not_data_train_val_exclude  --num_epochs 70 --lr_step 45,60 --weight_decay 1 --vp_pixel_per_mm 5  --data_train_person_id  8 --data_test_person_id  8",
    # "ctdet_gaze --exp_id gaze_resdcncut_18_mpii_ep70_all_norm_p05_2 --arch resdcncut_18 --down_ratio 2 --dataset mpiifacegaze --num_epochs 70 --lr_step 45,60 --vp_pixel_per_mm 5  --data_train_person_id  5 --data_test_person_id  5",
    # "ctdet_gaze --exp_id gaze_effv2s_mpii_ep70_all_norm_p08 --arch effv2s --dataset mpiifacegaze --num_epochs 70 --lr_step 45,60 --vp_pixel_per_mm 5 --data_train_person_id  8 --data_test_person_id  8",
    # "ctdet_gaze --exp_id gaze_effv2s_mpii_ep70_all_norm_p10 --arch effv2s --dataset mpiifacegaze --num_epochs 70 --lr_step 45,60 --vp_pixel_per_mm 5 --data_train_person_id 10 --data_test_person_id 10",
    # "ctdet_gaze --exp_id gaze_eve_sc_kr_test  --arch resdcn_18 --dataset eve  --num_epochs 20 --vp_h 2160 --vp_w 3840 --vp_pixel_per_mm 3.3 --camera_screen_pos --keep_res --resize_raw_image --resize_raw_image_h 270 --resize_raw_image_w 480 --not_data_train_val_exclude",
    
    "ctdet_gaze --exp_id gaze_impii_resdcn18_p05_test_himax_laptop_sc_kr_mono_vp_large --arch resdcn_18 --dataset himax   --num_epochs 70 --lr_step 45,60 --keep_res --resize_raw_image --camera_screen_pos --resize_raw_image_h 270 --resize_raw_image_w 480 --vp_h 2400 --vp_w 6200 --vp_pixel_per_mm 0 --data_train_person_id  5 --data_test_person_id  5 --gray_image",
    
    # "ctdet_gaze --exp_id gaze_impii_resdcn18_p05_test_himax_sc_kr_mono --arch resdcn_18 --dataset mpiifacegaze   --num_epochs 70 --lr_step 45,60 --keep_res --resize_raw_image --camera_screen_pos --resize_raw_image_h 270 --resize_raw_image_w 480 --vp_h 2100 --vp_w 3360 --vp_pixel_per_mm 5 --data_train_person_id  5 --data_test_person_id  5 --gray_image",
    # "ctdet_gaze --exp_id gaze_impii_resdcn18_p10_test_himax_sc_kr_mono --arch resdcn_18 --dataset mpiifacegaze   --num_epochs 70 --lr_step 45,60 --keep_res --resize_raw_image --camera_screen_pos --resize_raw_image_h 270 --resize_raw_image_w 480 --vp_h 2100 --vp_w 3360 --vp_pixel_per_mm 5 --data_train_person_id  10 --data_test_person_id  5 --gray_image"
    
        
    # "ctdet_gaze --exp_id gaze_resdcn18_csp_kr_resize_p10_petrain_eve --arch resdcn_18 --dataset mpiifacegaze --num_epochs 70  --lr_step 45,60 --keep_res --vp_h 2160 --vp_w 3840 --resize_raw_image --resize_raw_image_h 270 --resize_raw_image_w 480  --camera_screen_pos --vp_pixel_per_mm 3.3 --data_train_person_id 10 --data_test_person_id  10 --load_model /home/owenserver/Python/CenterNet_gaze/exp/ctdet_gaze/eve/resdcn_18/gaze_eve_sc_kr/model_5.pth",
    # "ctdet_gaze --exp_id gaze_resdcn18_csp_kr_resize_p14_petrain_eve --arch resdcn_18 --dataset mpiifacegaze --num_epochs 70  --lr_step 45,60 --keep_res --vp_h 2160 --vp_w 3840 --resize_raw_image --resize_raw_image_h 270 --resize_raw_image_w 480  --camera_screen_pos --vp_pixel_per_mm 3.3 --data_train_person_id 14 --data_test_person_id  14 --load_model /home/owenserver/Python/CenterNet_gaze/exp/ctdet_gaze/eve/resdcn_18/gaze_eve_sc_kr/model_5.pth",
    # "ctdet_gaze --exp_id gaze_resdcn18_csp_kr_resize_p08_petrain_eve --arch resdcn_18 --dataset mpiifacegaze --num_epochs 70  --lr_step 45,60 --keep_res --vp_h 2160 --vp_w 3840 --resize_raw_image --resize_raw_image_h 270 --resize_raw_image_w 480  --camera_screen_pos --vp_pixel_per_mm 3.3 --data_train_person_id 8 --data_test_person_id  8 --load_model /home/owenserver/Python/CenterNet_gaze/exp/ctdet_gaze/eve/resdcn_18/gaze_eve_sc_kr/model_5.pth",
    # "ctdet_gaze --exp_id gaze_resdcn18_csp_kr_resize_p03_petrain_eve --arch resdcn_18 --dataset mpiifacegaze --num_epochs 70  --lr_step 45,60 --keep_res --vp_h 2160 --vp_w 3840 --resize_raw_image --resize_raw_image_h 270 --resize_raw_image_w 480  --camera_screen_pos --vp_pixel_per_mm 3.3 --data_train_person_id 3 --data_test_person_id  3 --load_model /home/owenserver/Python/CenterNet_gaze/exp/ctdet_gaze/eve/resdcn_18/gaze_eve_sc_kr/model_5.pth",
    # "ctdet_gaze --exp_id gaze_resdcn18_csp_kr_resize_p12_petrain_eve --arch resdcn_18 --dataset mpiifacegaze --num_epochs 70  --lr_step 45,60 --keep_res --vp_h 2160 --vp_w 3840 --resize_raw_image --resize_raw_image_h 270 --resize_raw_image_w 480  --camera_screen_pos --vp_pixel_per_mm 3.3 --data_train_person_id 12 --data_test_person_id  12 --load_model /home/owenserver/Python/CenterNet_gaze/exp/ctdet_gaze/eve/resdcn_18/gaze_eve_sc_kr/model_5.pth",    
    
    # "ctdet_gaze --exp_id gaze_resdcn18_all_wd1_L1_base_sp_norm_p05 --arch resdcn_18 --dataset mpiifacegaze --num_epochs 70 --lr_step 45,60 --weight_decay 1  --data_train_person_id  5 --data_test_person_id  5",
    # "ctdet_gaze --exp_id gaze_resdcn18_all_wd01_L1_base_sp_norm_p05 --arch resdcn_18 --dataset mpiifacegaze --num_epochs 70 --lr_step 45,60 --weight_decay 0.1  --data_train_person_id  5 --data_test_person_id  5",
    # "ctdet_gaze --exp_id gaze_resdcn18_all_wd001_L1_base_sp_norm_p05 --arch resdcn_18 --dataset mpiifacegaze --num_epochs 70 --lr_step 45,60 --weight_decay 0.01  --data_train_person_id  5 --data_test_person_id  5",
    
    
    # "ctdet_gaze --exp_id gaze_resdcncut_18_mpii_sc_r10_wd1_all_norm_csp_kr_pl001_p05 --arch resdcncut_18 --dataset mpiifacegaze --num_epochs 70 --keep_res --resize_raw_image --camera_screen_pos --vp_h 2100 --vp_w 3360 --vp_pixel_per_mm 5 --pog_offset --pog_weight 0.01 --weight_decay 1 --data_train_person_id  5 --data_test_person_id  5",
    # "ctdet_gaze --exp_id gaze_resdcncut_18_mpii_sc_r10_wd01_all_norm_csp_kr_pl001_p05 --arch resdcncut_18 --dataset mpiifacegaze --num_epochs 70 --keep_res --resize_raw_image --camera_screen_pos --vp_h 2100 --vp_w 3360 --vp_pixel_per_mm 5 --pog_offset --pog_weight 0.01 --weight_decay 0.1 --data_train_person_id  5 --data_test_person_id  5",
    # "ctdet_gaze --exp_id gaze_resdcncut_18_mpii_sc_r10_wd001_all_norm_csp_kr_pl001_p05 --arch resdcncut_18 --dataset mpiifacegaze --num_epochs 70 --keep_res --resize_raw_image --camera_screen_pos --vp_h 2100 --vp_w 3360 --vp_pixel_per_mm 5 --pog_offset --pog_weight 0.01 --weight_decay 0.01 --data_train_person_id  5 --data_test_person_id  5",
    
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