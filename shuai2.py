# -*- coding:utf-8 -*-
import csv
import os
import cv2
import numpy as np
import random
from PIL import Image, ImageEnhance, ImageOps, ImageFile


SIZE_FACE = 227
train_dir = '/home/fuhongshuai/face_expression/train/'
img_dir = '/home/fuhongshuai/face_expression/train_face/'


def cut_image(img, ops):
    image = img.copy()
    if str(ops) == 'orgin':
        image = image[0:SIZE_FACE, 0:SIZE_FACE]
    elif str(ops) == 'turn':
        image = cv2.flip(image, 1)
    elif str(ops) == 'leftup':
        image = image[0:int(SIZE_FACE/2), 0:int(SIZE_FACE/2)]
    elif str(ops) == 'leftdown':
        image = image[int(SIZE_FACE/2):SIZE_FACE, 0:int(SIZE_FACE/2)]
    elif str(ops) == 'rightdown':
        image = image[int(SIZE_FACE/2):SIZE_FACE, int(SIZE_FACE/2):SIZE_FACE]
    elif str(ops) == 'rightup':
        image = image[0:int(SIZE_FACE/2), int(SIZE_FACE/2):SIZE_FACE]
    elif str(ops) == 'up':
        image = image[int(SIZE_FACE/2):SIZE_FACE, int(SIZE_FACE/4):int(SIZE_FACE*0.75)]
    elif str(ops) == 'down':
        image = image[0:int(SIZE_FACE/2), int(SIZE_FACE/4):int(SIZE_FACE*0.75)]
    elif str(ops) == 'focus':
        image = image[int(SIZE_FACE * 0.25):int(SIZE_FACE * 0.75),
                int(SIZE_FACE * 0.25):int(SIZE_FACE * 0.75)]
    return image


if __name__ == '__main__':
    for file in os.listdir(train_dir):
        file_path = train_dir + file
        label1 = file.split('_')
        name1 = int(label1[0])
        name = file.split('.')
        img = cv2.imread(file_path)
        orgin_img = cut_image(img, 'orgin')
        orgin_img = cv2.resize(orgin_img, (227, 227))
        rename = name[0] + '_' + '1' + '.jpg'
        cv2.imwrite(img_dir + rename, orgin_img)

        turn_img = cut_image(img, 'turn')
        turn_img = cv2.resize(turn_img, (227, 227))
        rename = name[0] + '_' + '1' + '.jpg'
        cv2.imwrite(img_dir + rename, turn_img)

        leftup_img = cut_image(img, 'leftup')
        leftup_img = cv2.resize(leftup_img, (227, 227))
        rename = name[0] + '_' + '1' + '.jpg'
        cv2.imwrite(img_dir + rename, leftup_img)

        leftdown_img = cut_image(img, 'leftdown')
        leftdown_img = cv2.resize(leftdown_img, (227, 227))
        rename = name[0] + '_' + '1' + '.jpg'
        cv2.imwrite(img_dir + rename, leftdown_img)

        rightdown_img = cut_image(img, 'rightdown')
        rightdown_img = cv2.resize(rightdown_img, (227, 227))
        rename = name[0] + '_' + '1' + '.jpg'
        cv2.imwrite(img_dir + rename, rightdown_img)

        rightup_img = cut_image(img, 'rightup')
        rightup_img = cv2.resize(rightup_img, (227, 227))
        rename = name[0] + '_' + '1' + '.jpg'
        cv2.imwrite(img_dir + rename, rightup_img)

        up_img = cut_image(img, 'up')
        up_img = cv2.resize(up_img, (227, 227))
        rename = name[0] + '_' + '1' + '.jpg'
        cv2.imwrite(img_dir + rename, up_img)

        down_img = cut_image(img, 'down')
        down_img = cv2.resize(down_img, (227, 227))
        rename = name[0] + '_' + '1' + '.jpg'
        cv2.imwrite(img_dir + rename, down_img)

        focus_img = cut_image(img, 'focus')
        focus_img = cv2.resize(focus_img, (227, 227))
        rename = name[0] + '_' + '1' + '.jpg'
        cv2.imwrite(img_dir + rename, focus_img)


