import subprocess

# 要執行的檔案及對應的參數
file = "main.py"
file_params = [
    "ctdet_gaze --exp_id gaze_resdcn18_ep70_all_test_p12 --arch resdcn_18 --dataset mpiifacegaze --num_epochs 70 --lr_step 45,60 --heat_map_debug --data_person_id 12",
    "ctdet_gaze --exp_id gaze_resdcn18_ep70_all_test_p11 --arch resdcn_18 --dataset mpiifacegaze --num_epochs 70 --lr_step 45,60 --heat_map_debug --data_person_id 11",
    "ctdet_gaze --exp_id gaze_resdcn18_ep70_all_test_p09 --arch resdcn_18 --dataset mpiifacegaze --num_epochs 70 --lr_step 45,60 --heat_map_debug --data_person_id  9",
    "ctdet_gaze --exp_id gaze_resdcn18_ep70_all_test_p08 --arch resdcn_18 --dataset mpiifacegaze --num_epochs 70 --lr_step 45,60 --heat_map_debug --data_person_id  8",
    "ctdet_gaze --exp_id gaze_resdcn18_ep70_all_test_p06 --arch resdcn_18 --dataset mpiifacegaze --num_epochs 70 --lr_step 45,60 --heat_map_debug --data_person_id  6",
    "ctdet_gaze --exp_id gaze_resdcn18_ep70_all_test_p05 --arch resdcn_18 --dataset mpiifacegaze --num_epochs 70 --lr_step 45,60 --heat_map_debug --data_person_id  5",
    "ctdet_gaze --exp_id gaze_resdcn18_ep70_all_test_p04 --arch resdcn_18 --dataset mpiifacegaze --num_epochs 70 --lr_step 45,60 --heat_map_debug --data_person_id  4",
    "ctdet_gaze --exp_id gaze_resdcn18_ep70_all_test_p03 --arch resdcn_18 --dataset mpiifacegaze --num_epochs 70 --lr_step 45,60 --heat_map_debug --data_person_id  3",
    "ctdet_gaze --exp_id gaze_resdcn18_ep70_all_test_p01 --arch resdcn_18 --dataset mpiifacegaze --num_epochs 70 --lr_step 45,60 --heat_map_debug --data_person_id  1",
    "ctdet_gaze --exp_id gaze_resdcn18_ep70_all_test_p00 --arch resdcn_18 --dataset mpiifacegaze --num_epochs 70 --lr_step 45,60 --heat_map_debug --data_person_id  0"
    
    
]


# 使用迴圈依次執行每個檔案及其參數
for params in file_params:
    # 在這裡加入你希望執行的程式碼，file代表檔案名稱，params代表參數
    print(f"執行檔案: {file}，參數: {params}")
    # 示範如何執行檔案並將參數傳遞給它
    subprocess.run(["python", file] + params.split())