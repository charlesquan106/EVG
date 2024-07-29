import numpy as np
import cv2 
import os
import json

root = "/home/cyh/GazeDataset20200519/Original/GazeCapture"
out_root = "/home/cyh/GazeDataset20200519/GazePoint/GazeCapture"

def ImageProcessing_GazeCapture():
    persons = os.listdir(root)
    persons.sort()

    length = len(persons)

    for count, person in enumerate(persons):

        im_root = os.path.join(root, person)
        person_info = json.load(open(os.path.join(im_root, "info.json")))

        splited_set = person_info["Dataset"]
        devices = person_info["DeviceName"]

        
        im_outpath = os.path.join(out_root, "Image", person)
        label_outpath = os.path.join(out_root, "Label", splited_set, f"{person}.label")

        if not os.path.exists(os.path.join(im_outpath, 'face')):
            os.makedirs(os.path.join(im_outpath, 'face'))

        if not os.path.exists(os.path.join(im_outpath, 'left')):
            os.makedirs(os.path.join(im_outpath, 'left'))

        if not os.path.exists(os.path.join(im_outpath, 'right')):
            os.makedirs(os.path.join(im_outpath, 'right'))

        if not os.path.exists(os.path.join(im_outpath, 'grid')):
            os.makedirs(os.path.join(im_outpath, 'grid'))

        if not os.path.exists(os.path.join(out_root, "Label", splited_set)):
            os.makedirs(os.path.join(out_root, "Label", splited_set))

        progressbar = "".join(["\033[41m%s\033[0m" % '   '] * int(count/length * 20))
        progressbar = "\r" + progressbar + f" {count}|{length}, Prcessing {person}.."
        print(progressbar, end="", flush=True)

        ImageProcessing_Person(im_root, im_outpath, label_outpath, person, devices)


def ImageProcessing_Person(im_root, im_outpath, label_outpath, person, devices):
    # Read annotation files
    frames = json.load(open(os.path.join(im_root, "frames.json")))
    face_located = json.load(open(os.path.join(im_root, "appleFace.json")))
    left_located = json.load(open(os.path.join(im_root, "appleLeftEye.json")))
    right_located = json.load(open(os.path.join(im_root, "appleRightEye.json")))
    grid_info = json.load(open(os.path.join(im_root, "faceGrid.json")))
    gt_info = json.load(open(os.path.join(im_root, "dotInfo.json")))

    outfile = open(label_outpath, 'w')
    outfile.write("Face Left Right Grid Xcam,Ycam Xdot,Ydot Device\n")

    for index, frame in enumerate(frames):
        if not face_located["IsValid"][index]: continue
        if not left_located["IsValid"][index]: continue
        if not right_located["IsValid"][index]: continue
        if not grid_info["IsValid"][index]: continue

        im_path = os.path.join(im_root, "frames", frame)
        img = cv2.imread(im_path)


        face_img = CropImg(img, face_located["X"][index], face_located["Y"][index], 
                                face_located["W"][index], face_located["H"][index])
 
        left_img = CropImg(face_img, left_located["X"][index], left_located["Y"][index], 
                                left_located["W"][index], left_located["H"][index])

        right_img = CropImg(face_img,right_located["X"][index],right_located["Y"][index], 
                               right_located["W"][index],right_located["H"][index])

        grid = np.zeros((25, 25))
        X,Y = grid_info["X"][index], grid_info["Y"][index]
        W,H = grid_info["W"][index], grid_info["H"][index]
        grid[Y:(Y+H),X:(X+W)] = np.ones_like(grid[Y:(Y+H),X:(X+W)])

        centimeters_label = [str(gt_info["XCam"][index]), str(gt_info["YCam"][index])]
        pixel_label = [str(gt_info["XPts"][index]), str(gt_info["YPts"][index])]

        save_face_path = os.path.join(person, 'face', frame)
        save_left_path = os.path.join(person, 'left', frame)
        save_right_path = os.path.join(person, 'right', frame)
        save_grid_path = os.path.join(person, 'grid', frame)

        label = " ".join([save_face_path, save_left_path, save_right_path, save_grid_path, ",".join(centimeters_label), ",".join(pixel_label), devices.replace(" ", "")])
        outfile.write(label  + "\n")

        cv2.imwrite(os.path.join(im_outpath, 'face', frame), face_img)
        cv2.imwrite(os.path.join(im_outpath, 'left', frame), left_img)
        cv2.imwrite(os.path.join(im_outpath, 'right', frame), right_img)
        cv2.imwrite(os.path.join(im_outpath, 'grid', frame), grid)

    outfile.close()

def CropImg(img, X, Y, W, H):
    Y_lim, X_lim, _ = img.shape
    H =  min(H, Y_lim)
    W = min(W, X_lim)

    X, Y, W, H = list(map(int, [X, Y, W, H]))
    X = max(X, 0)
    Y = max(Y, 0)

    if X + W > X_lim:
        X = X_lim - W

    if Y + H > Y_lim:
        Y = Y_lim - H

    return img[Y:(Y+H),X:(X+W)]

if __name__ == "__main__":
    ImageProcessing_GazeCapture()
