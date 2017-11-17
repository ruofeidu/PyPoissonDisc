import numpy as np
import cv2, os
import math, time
from rtree import index as rt

point_radius = 5
distance_scale = 50.0
distance_baseline = 0.1
num_darts = 100000 #15.81s


def im2double(im):
    return im.astype(np.float) / np.iinfo(im.dtype).max


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299 / 255., 0.587 / 255., 0.114 / 255.])


dir_path = os.path.dirname(os.path.realpath(__file__))
filename = "circles"
img = cv2.imread('%s/%s.png' % (dir_path, filename))
filename = "Sampling_kernel4_sigma4"
(height, width, _) = img.shape
mask = rgb2gray(cv2.imread('%s/%s.png' % (dir_path, filename)))
res = np.zeros((height, width, 3), np.float)
pt = np.floor(np.random.random(2) * width).astype(int)
tree = rt.Index()
id = 0
tree.insert(0, (pt[0], pt[1], pt[0], pt[1]))
plist = np.zeros([num_darts, 2], np.int)
plist[0] = pt
col = img[pt[0], pt[1]]
#print(col, np.int_(col), col.astype(int))
clr = (int(col[0]), int(col[1]), int(col[2]))
cv2.circle(res, (pt[0], pt[1]), point_radius, clr, -1)

start = time.time()
for iterations in range(num_darts):
    pt = np.floor(np.random.random(2) * width).astype(int)
    nid = list(tree.nearest((pt[0], pt[1], pt[0], pt[1])))[0]
    dist = np.linalg.norm(plist[nid] - pt)
    radius = mask[pt[0]][pt[1]] * distance_scale + distance_baseline
    if dist > radius:
        id += 1
        tree.insert(id, (pt[0], pt[1], pt[0], pt[1]))
        plist[id] = pt
        col = img[pt[0], pt[1]]
        clr = (int(col[0]), int(col[1]), int(col[2]))
        cv2.circle(res, (pt[0], pt[1]), point_radius, clr, -1)


filename += "_res"
cv2.imwrite("%s/%s.png" % (dir_path, filename), res)

print(time.time() - start)

