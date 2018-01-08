import numpy as np
import cv2, os
import math, time
from rtree import index as rt

point_radius = 5
distance_scale = 10.0
#distance_baseline = 2
distance_baseline = 2
num_darts = 400000 # 15.81s
#num_darts = 2000 # debug


def im2double(im):
    return im.astype(np.float) / np.iinfo(im.dtype).max


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299 / 255., 0.587 / 255., 0.114 / 255.])


dir_path = os.path.dirname(os.path.realpath(__file__))
filename = "circles"
img = cv2.imread('%s/%s.png' % (dir_path, filename))
(height, width, _) = img.shape

min_ac_samples = num_darts

# scales = [2.81, 2.63, 2.39, 2.09]
# total ac samples:  2178
# 52.16026711463928
# total ac samples:  2036
# 49.768271923065186
# total ac samples:  1983
# 51.95157217979431
# total ac samples:  2093
# 50.33324646949768
# scales = [2.83, 2.621, 2.378, 2.082]
# total ac samples:  2120
# 50.910128355026245
# total ac samples:  2085
# 50.291696071624756
# total ac samples:  2020
# 49.75225329399109
# total ac samples:  2102
# 49.512478828430176
# min 2020
# scales = [2.88, 2.64, 2.38, 2.10]
# total ac samples:  2054
# 52.03005075454712
# total ac samples:  2023
# 48.901695013046265
# total ac samples:  2016
# 48.43964719772339
# total ac samples:  2061
# 48.28515982627869
# min 2016
# scales = [2.92, 2.7, 2.4, 2.2]
# total ac samples:  2027
# 66.21746754646301
# total ac samples:  1938
# 65.96248316764832
# total ac samples:  2005
# 64.40946054458618
# total ac samples:  1911
# 66.96147918701172
# min 1911
# total ac samples:  1990
# 65.22026062011719
# total ac samples:  2005
# 67.50421524047852
# total ac samples:  2005
# 68.69789862632751
# total ac samples:  2022
# 67.32213377952576
# min 1990

scales = [2.0, 2.94, 2.65, 2.395, 2.13, 4.24, 3.9, 3.46, 3.1, 6.0, 5.5, 4.9, 4.4]

# total ac samples:  3150
# 66.52730226516724
# total ac samples:  2928
# 66.57245588302612
# total ac samples:  2676
# 65.69244265556335
# total ac samples:  2574
# 66.59098982810974
# total ac samples:  1564
# 64.29867577552795
# total ac samples:  1404
# 61.02004623413086
# total ac samples:  1335
# 63.225895166397095
# total ac samples:  1252
# 62.611377000808716
# total ac samples:  805
# 60.35903286933899
# total ac samples:  723
# 56.351922035217285
# total ac samples:  689
# 61.894041538238525
# total ac samples:  642


arr = next(os.walk('./MasksV2'))[2]
for fid, file in enumerate(arr):
    filename = file[:-4]
    mask = rgb2gray(cv2.imread('%s/Masks/%s.png' % (dir_path, filename)))
    frameBufferSize = int(filename[15])
    if frameBufferSize == 1:
        circleSize = point_radius
    elif frameBufferSize == 2:
        circleSize = point_radius * 2
    elif frameBufferSize == 3:
        circleSize = point_radius * 4
    if frameBufferSize != 1:
        break
    circleSize = point_radius
    # scale = 1
    # if frameBufferSize == 2:
    #     scale = 1.4*1.4
    # elif frameBufferSize == 3:
    #     scale = 2.0*2.0
    scale = scales[fid]
    res = np.zeros((height, width, 3), np.float)
    pt = np.floor(np.random.random(2) * width).astype(int)
    tree = rt.Index()
    id = 0
    tree.insert(id, (pt[0], pt[1], pt[0], pt[1]))
    plist = np.zeros([num_darts, 2], np.int)
    plist[0] = pt
    col = img[pt[0], pt[1]]
    #print(col, np.int_(col), col.astype(int))
    clr = (int(col[0]), int(col[1]), int(col[2]))
    cv2.circle(res, (pt[0], pt[1]), circleSize, clr, -1)

    start = time.time()
    for iterations in range(num_darts):
        pt = np.floor(np.random.random(2) * width).astype(int)
        nid = list(tree.nearest((pt[0], pt[1], pt[0], pt[1])))[0]
        dist = np.linalg.norm(plist[nid] - pt)
        radius = (mask[pt[0]][pt[1]] * distance_scale + distance_baseline) * scale
        if dist > radius:
            id += 1
            tree.insert(id, (pt[0], pt[1], pt[0], pt[1]))
            plist[id] = pt
            col = img[pt[0], pt[1]]
            clr = (int(col[0]), int(col[1]), int(col[2]))
            cv2.circle(res, (pt[0], pt[1]), circleSize, clr, -1)
            # if id > 654:
            #     break
    if (id + 1) < min_ac_samples:
        min_ac_samples = id + 1
    centralPoint = (height >> 1, width >> 1)
    # cv2.circle(res, centralPoint, int(width / 2 / 5), (0, 104, 55), 5)
    # cv2.circle(res, centralPoint, int(width / 2 * 2 / 5), (49, 163, 84), 5)
    # cv2.circle(res, centralPoint, int(width / 2 * 3 / 5), (120, 198, 121), 5)
    # cv2.circle(res, centralPoint, int(width / 2 * 4 / 5), (173, 221, 142), 5)
    # cv2.circle(res, centralPoint, int(width / 2), (217, 240, 163), 5)
    cv2.imwrite("%s/ResultsV7/%s.png" % (dir_path, file[:-4]), res)
    print("total ac samples: ", id + 1)
    print(time.time() - start)
print("min", min_ac_samples)