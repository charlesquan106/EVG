import subprocess

# 要執行的檔案及對應的參數
file = "data2voc_MPIIFaceGaze.py"
file_params = [
    "-d train  -v  4",
    "-d train  -v  5",
    "-d train  -v  6",
    "-d train  -v  7",
    "-d train  -v  8",
    "-d train  -v  9",
    "-d train  -v 10",
    "-d train  -v 11",
    "-d train  -v 12",
    "-d train  -v 13",
    "-d train  -v 14"


]


    # "-d train  -v  0", "-d test   -v  0",
    # "-d train  -v  1", "-d test   -v  1",
    # "-d train  -v  3", "-d test   -v  3",
    # "-d train  -v  4", "-d test   -v  4",
    # "-d train  -v  5", "-d test   -v  5",
    # "-d train  -v  6", "-d test   -v  6",
    # "-d train  -v  8", "-d test   -v  8",
    # "-d train  -v  9", "-d test   -v  9",
    # "-d train  -v 11", "-d test   -v 11",
    # "-d train  -v 12", "-d test   -v 12",
    # "-d train  -v 13", "-d test   -v 13",
    # "-d train  -v 14", "-d test   -v 14",


# 使用迴圈依次執行每個檔案及其參數
for params in file_params:
    # 在這裡加入你希望執行的程式碼，file代表檔案名稱，params代表參數
    print(f"執行檔案: {file}，參數: {params}")
    # 示範如何執行檔案並將參數傳遞給它
    subprocess.run(["python", file] + params.split())