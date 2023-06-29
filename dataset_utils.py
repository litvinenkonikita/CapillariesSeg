
import argparse
import cv2 #as cv
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import shutil
import torch


def test_cuda():
    print(f'torch.cuda.is_available(): {torch.cuda.is_available()}')
    print(f'torch.cuda.device_count(): {torch.cuda.device_count()}')
    print(f'torch.cuda.current_device(): {torch.cuda.current_device()}')
    print(f'torch.cuda.get_device_name(): {torch.cuda.get_device_name()}')
    print(f'torch.cuda.get_device_properties(): {torch.cuda.get_device_properties(torch.cuda.current_device())}')
    print(f'torch.cuda.get_device_capability(): {torch.cuda.get_device_capability()}')


def get_train_val_test_from_ratio(df_indices, TRAIN_VAL_TEST_RATIO):
    
    TRAIN_DF_SIZE, VAL_DF_SIZE, TEST_DF_SIZE = get_train_val_test_sizes(df_indices, TRAIN_VAL_TEST_RATIO)
    train_indices = set(random.sample(list(df_indices), TRAIN_DF_SIZE))
    val_test_indices = set(df_indices) - train_indices
    val_indices = set(random.sample(val_test_indices, VAL_DF_SIZE))
    test_indices = val_test_indices - val_indices
    
    return train_indices, val_indices, test_indices


def get_train_val_test_indices(male_df_indices, female_df_indices, TRAIN_VAL_TEST_MALES, TRAIN_VAL_TEST_FEMALES):
    train_male_indices = set(random.sample(list(male_df_indices), TRAIN_VAL_TEST_MALES[0]))
    train_female_indices = set(random.sample(list(female_df_indices), TRAIN_VAL_TEST_FEMALES[0]))
    all_train_indices = train_male_indices.union(train_female_indices)

    val_test_male_indices = set(male_df_indices) - train_male_indices
    val_test_female_indices = set(female_df_indices) - train_female_indices

    val_male_indices = set(random.sample(val_test_male_indices, TRAIN_VAL_TEST_MALES[1]))
    val_female_indices = set(random.sample(val_test_female_indices, TRAIN_VAL_TEST_FEMALES[1]))
    all_val_indices = val_male_indices.union(val_female_indices)

    test_male_indices = val_test_male_indices - val_male_indices
    test_female_indices = val_test_female_indices - val_female_indices
    all_test_indices = test_male_indices.union(test_female_indices)
    
    return all_train_indices, all_val_indices, all_test_indices


def get_train_val_test_sizes(df_index, TRAIN_VAL_TEST_RATIO):
    VAL_DF_SIZE = len(df_index) * TRAIN_VAL_TEST_RATIO[1] // sum(TRAIN_VAL_TEST_RATIO)
    TEST_DF_SIZE = len(df_index) * TRAIN_VAL_TEST_RATIO[2] // sum(TRAIN_VAL_TEST_RATIO)
    TRAIN_SIZE = len(df_index) - (VAL_DF_SIZE + TEST_DF_SIZE)
    return TRAIN_SIZE, VAL_DF_SIZE, TEST_DF_SIZE


def plot_hist(df, tit, dest='train'):
    dfT = df.loc[(df["is_val_or_test_set"]==dest)]
    plt.hist(np.array(dfT[tit]))
    plt.title(tit + ' : ' +str(dest))
    print ('min', dfT[tit].min(), 'median', dfT[tit].median(), 'max', dfT[tit].max())

    
def clean_dir(dir_path):
    if os.path.exists(dir_path) and os.path.isdir(dir_path) and len(os.listdir(dir_path)) > 0 :
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

                
def checkRectsIntersected(rect1, rect2):
    x = max(rect1[0], rect2[0])
    y = max(rect1[1], rect2[1])
    w = min(rect1[0] + rect1[2], rect2[0] + rect2[2]) - x
    h = min(rect1[1] + rect1[3], rect2[1] + rect2[3]) - y

    foundIntersect = True
    if w < 0 or h < 0:
        foundIntersect = False

    return foundIntersect, [x, y, w, h]


def mergeRects(rect1, rect2):
    ##print('mergeRects():')
    min_x = min(rect1[0], rect2[0])
    min_y = min(rect1[1], rect2[1])
    merged_rect = (min_x,
                   min_y,
                   max(rect1[0] + rect1[2], rect2[0] + rect2[2]) - min_x,
                   max(rect1[1] + rect1[3], rect2[1] + rect2[3]) - min_y)
    ##print(f'merged_rect: {merged_rect}')
    return merged_rect


def intersection(rectA, rectB): # check if rect A & B intersect
    a, b = rectA, rectB
    startX = max(min(a[0], a[2]), min(b[0], b[2]))
    startY = max(min(a[1], a[3]), min(b[1], b[3]))
    endX = min(max(a[0], a[2]), max(b[0], b[2]))
    endY = min(max(a[1], a[3]), max(b[1], b[3]))
    if startX < endX and startY < endY:
        return True
    else:
        return False


#def rect_distance((x1, y1, x1b, y1b), (x2, y2, x2b, y2b)):
def rect_distance(rect1, rect2):

    x1 = rect1[0]
    y1 = rect1[1]
    x1b = rect1[2]
    y1b = rect1[3]

    x2 = rect2[0]
    y2 = rect2[1]
    x2b = rect2[2]
    y2b = rect2[3]

    left = x2b < x1
    right = x1b < x2
    bottom = y2b < y1
    top = y1b < y2

    if top and left:
        return math.dist((x1, y1b), (x2b, y2))
    elif left and bottom:
        return math.dist((x1, y1), (x2b, y2b))
    elif bottom and right:
        return math.dist((x1b, y1), (x2, y2b))
    elif right and top:
        return math.dist((x1b, y1b), (x2, y2))
    elif left:
        return x1 - x2b
    elif right:
        return x2 - x1b
    elif bottom:
        return y1 - y2b
    elif top:
        return y2 - y1b
    else:             # rectangles intersect
        return 0


def in_vicinity(rectA, rectB, vicinity): # check if rect A & B intersect
    a = rectA
    b = rectB
    if rect_distance(a, b) <= vicinity:
        return True
    else:
        return False


def combine_rects(rectA, rectB): # create bounding box for rect A & B
    a, b = rectA, rectB
    startX = min(a[0], b[0])
    startY = min(a[1], b[1])
    endX = max(a[2], b[2])
    endY = max(a[3], b[3])
    return startX, startY, endX, endY


def merge_intersected_rects(rects):
    #print('merge_intersected_rects():')
    if rects is None:
        return None
    mainRects = rects
    noIntersect = False
    while noIntersect == False and len(mainRects) > 1:
        mainRects = list(set(mainRects))
        # get the unique list of rect, or the noIntersect will be
        # always true if there are same rect in mainRects
        newRectsArray = []
        for rectA, rectB in itertools.combinations(mainRects, 2):
            newRect = []
            if intersection(rectA, rectB):
                newRect = combine_rects(rectA, rectB)
                newRectsArray.append(newRect)
                noIntersect = False
                # delete the used rect from mainRects
                if rectA in mainRects:
                    mainRects.remove(rectA)
                if rectB in mainRects:
                    mainRects.remove(rectB)
        if len(newRectsArray) == 0:
            # if no newRect is created = no rect in mainRect intersect
            noIntersect = True
        else:
            # loop again the combined rect and those remaining rect in mainRects
            mainRects = mainRects + newRectsArray
    return mainRects


def merge_in_vicinity_rects(rects, vicinity=3):
    #print('merge_in_vicinity_rects():')
    if rects is None:
        return None
    mainRects = rects
    not_in_vicinity = False
    while not_in_vicinity == False and len(mainRects) > 1:
        mainRects = list(set(mainRects))
        # get the unique list of rect, or the not_in_vicinity will be
        # always true if there are same rect in mainRects
        newRectsArray = []
        for rectA, rectB in itertools.combinations(mainRects, 2):
            newRect = []
            if in_vicinity(rectA, rectB, vicinity):
                newRect = combine_rects(rectA, rectB)
                newRectsArray.append(newRect)
                not_in_vicinity = False
                # delete the used rect from mainRects
                if rectA in mainRects:
                    mainRects.remove(rectA)
                if rectB in mainRects:
                    mainRects.remove(rectB)
        if len(newRectsArray) == 0:
            # if no newRect is created = no rect in mainRect in vicinity
            not_in_vicinity = True
        else:
            # loop again the combined rect and those remaining rect in mainRects
            mainRects = mainRects + newRectsArray
    return mainRects


def remove_small_rects(rects, threshold_size=5):
    #print('remove_small_rects():')
    if rects is None:
        return None
    mainRects = []
    for rect in rects:
        if rect[2]-rect[0] >= threshold_size or rect[3]-rect[1] >= threshold_size:
            mainRects.append(rect)
    return mainRects


def extend_rects(rects, img_size=(1000, 1000), margins_size = 5):
    #print('extend_rects():')
    if rects is None:
        return None
    extended_rects = []
    for rect in rects:
        x1 = rect[0] - margins_size
        y1 = rect[1] - margins_size
        x2 = rect[2] + margins_size
        y2 = rect[3] + margins_size

        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        if x2 > img_size[0]-1:
            x2 = img_size[0]-1
        if y2 > img_size[1]-1:
            y2 = img_size[1]-1

        extended_rect = (x1, y1, x2, y2)
        extended_rects.append(extended_rect)
    return extended_rects


def getRects_Contours(src_gray, threshold, threshold_max):
    canny_output = cv2.Canny(src_gray, threshold, threshold_max)
    contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    #contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f'len(contours) : {len(contours)}')
    print(f'len(hierarchy[0]) : {len(hierarchy[0])}')
    outer_contours = []
    outer_boundRects = []
    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)

    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        if hierarchy[0, i, 3] == -1:
            outer_contours.append(contours_poly[i])
            outer_boundRects.append(boundRect[i])

    print(f'len(outer_contours) : {len(outer_contours)}')
    print(f'len(outer_boundRects) : {len(outer_boundRects)}')

    only_outer_boundRects = []
    only_outer_contours = []
    root_boundRect = None
    root_contour = None
    for i in range(len(outer_boundRects)):
        root_boundRect = outer_boundRects[i]
        root_contour = outer_contours[i]
        #print(f'root_boundRect : {root_boundRect}')
        for j in range(len(outer_boundRects)):
            if i != j and outer_boundRects[j][0] <= root_boundRect[0] and outer_boundRects[j][1] <= root_boundRect[1] \
                    and root_boundRect[2]+root_boundRect[0] <= outer_boundRects[j][2]+outer_boundRects[j][0] \
                    and root_boundRect[3]+root_boundRect[1] <= outer_boundRects[j][3]+outer_boundRects[j][1]:
                root_boundRect = outer_boundRects[j]
                root_contour = outer_contours[j]

        #only_outer_boundRects.append(root_boundRect)
        only_outer_boundRects.append((root_boundRect[0], root_boundRect[1],
                                      root_boundRect[0] + root_boundRect[2],
                                      root_boundRect[1] + root_boundRect[3]))
        only_outer_contours.append(root_contour)

    return only_outer_boundRects, only_outer_contours


def thresh_callback(val):
    #threshold = val
    only_outer_boundRects, only_outer_contours = getRects_Contours(src_gray, thresh, max_thresh)
    print(f'len(only_outer_boundRects) : {len(only_outer_boundRects)}')
    print(f'len(only_outer_contours) : {len(only_outer_contours)}')
    mergedRects = merge_intersected_rects(only_outer_boundRects)
    print(f'len(mergedRects) : {len(mergedRects)}')
    mergedRects = merge_in_vicinity_rects(mergedRects, vicinity=2)
    print(f'len(mergedRects) : {len(mergedRects)}')
    rects = remove_small_rects(mergedRects, threshold_size=5)
    print(f'len(rects) : {len(rects)}')
    extended_rects = extend_rects(rects, img_size=src_gray.shape[:2], margins_size=5)
    print(f'len(extended_rects) : {len(extended_rects)}')

    canny_output = cv2.Canny(src_gray, 0, 255)

    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1]), dtype=np.uint8)
    thickness = 1
    color = (255, 255, 255)

    for i in range(len(extended_rects)):
        cv2.rectangle(drawing, (int(extended_rects[i][0]), int(extended_rects[i][1])),
                                (int(extended_rects[i][2]), int(extended_rects[i][3])), color, thickness)

    cv2.imshow('Contours', drawing)