import csv
import os
import cv2
import dlib

current_dir = os.getcwd()

csv_dir1 = current_dir + "/Manually_Annotated_filelist/training.csv"
csv_dir2 = current_dir + "/Manually_Annotated_filelist/validation.csv"
img_dir = current_dir + "/Manually_Annotated/Manually_Annotated_Images/"
resave_dir = current_dir + '/face/'
predicter_path = current_dir + '/shape_predictor_68_face_landmarks.dat'
face_dir = current_dir + "/manface/"
reface_dir = current_dir + "/face_expression/"
'''
csv_dir1 = current_dir + "/Manually_Annotated_filelist/training.csv"
csv_dir2 = current_dir + "/Manually_Annotated_filelist/validation.csv"
img_dir = current_dir + "/face_test/"
resave_dir = current_dir + '/face_test/'
predicter_path = current_dir + '/shape_predictor_68_face_landmarks.dat'
face_dir = current_dir + "/face_save/"
'''

def rename():
    csvFile1 = open(csv_dir1, "r")
    reader1 = csv.reader(csvFile1)
    csvFile2 = open(csv_dir2, "r")
    reader2 = csv.reader(csvFile2)

    result = {}
    for item in reader1:
        if reader1.line_num == 1:
            continue
        result[item[0]] = item[6]
    for item in reader2:
        if reader2.line_num == 1:
            continue
        result[item[0]] = item[6]

    csvFile1.close()
    csvFile2.close()

    num = 0
    for file_key in result.keys():
        img_path = img_dir + file_key
        img = cv2.imread(img_path)
        if img_path.endswith('.jpg'):
            rename = str(result[file_key]) + '_' + str(num) + '.jpg'
            num += 1
            cv2.imwrite(resave_dir + rename, img)
        else:
            os.remove(img_path)

def operate_face():
    file_dir = os.listdir(resave_dir)
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(predicter_path)
    #deny noisy and gray
    for file in file_dir:
        face_path = resave_dir + file
        bgr_img = cv2.imread(face_path)
        if bgr_img is None:
            print("Sorry")
            continue
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        dets = detector(rgb_img,1)
        num_faces = len(dets)
        if num_faces == 0:
            print("Sorry, we could not load an image".format(face_path))
            continue
        faces = dlib.full_object_detections()
        for det in dets:
            faces.append(sp(rgb_img,det))
        name = file.split('.')
        face_pa = os.path.join(face_dir,name[0])
        dlib.save_face_chips(rgb_img,faces,face_pa)
    file_dir = os.listdir(face_dir)
    detector = dlib.get_frontal_face_detector()
    #print(file_dir)
    for file in file_dir:
        if file == ".DS_Store":
            continue
        face_path = face_dir + file
        print(face_path)
        img = cv2.imread(face_path)
        if img is None:
            continue
        img = cv2.resize(img, (227, 227))
        #print(img.shape)
        #print("shuai")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dets = detector(img,1)
        for index, face in enumerate(dets):
            #print("shuai")
            left = face.left()
            top = face.top()
            right = face.right()
            bottom = face.bottom()
            height = bottom - top
            weight = right - left
            num = max(height, weight)
            face = img[top:top + num, left:left + num]
            #if face.shape[0] < 48 or face.shape[1] < 48:
            #   continue
            print(face.shape)
            img = cv2.resize(face, (227, 227), interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(reface_dir+file,img)
            print(img.shape)


if __name__ == '__main__':
    #rename()
    operate_face()


