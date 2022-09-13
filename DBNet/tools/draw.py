# import cv2
# from PIL import Image
# import numpy as np
#
#
# point = [56, 52, 67, 75, 103, 51, 127, 51, 163, 71, 170, 71, 179, 56, 148, 36, 115, 29, 85, 35]
# point = np.array(point, np.int32)
# point = point.reshape((-1,1,2))
# print(point)
# img = cv2.imread('../demo_images/demo.jpg')
# cv2.fillPoly(img, point,(255,255,0), 100)
# cv2.imshow('show', img)
# cv2.waitKey(100000)
#
#
# # test = [[56, 52, 67, 75, 103, 51, 127, 51, 163, 71, 170, 71, 179, 56, 148, 36, 115, 29, 85, 35], [51, 141, 70, 159, 110, 174, 137, 174, 180, 147, 170, 129, 129, 151, 100, 149, 67, 125, 58, 125], [98, 206, 98, 221, 127, 221, 127, 208, 120, 208, 119, 206], [133, 206, 133, 221, 164, 221, 164, 206], [27, 83, 27, 112, 210, 112, 206, 81], [99, 126, 99, 139, 137, 139, 137, 122, 135, 120, 105, 120]]

"""Crop a polygonal selection from an image."""
import numpy as np
from PIL import Image
from shapely.geometry import Point
from shapely.geometry import Polygon


im = Image.open('demo.jpg').convert('RGB')
pixels = np.array(im)
im_copy = np.array(im)

region = Polygon([(229, 454), (211, 431), (216, 402), (250, 380), (272, 343), (262, 299), (222, 270), (212, 248), (215, 232), (241, 216), (302, 269), (319, 310), (319, 360), (293, 419)])

for index, pixel in np.ndenumerate(pixels):
  # Unpack the index.
  row, col, channel = index
  # We only need to look at spatial pixel data for one of the four channels.
  if channel != 0:
    continue
  point = Point(row, col)
  if not region.contains(point):
    im_copy[(row, col, 0)] = 255
    im_copy[(row, col, 1)] = 255
    im_copy[(row, col, 2)] = 255
    # im_copy[(row, col, 3)] = 0

cut_image = Image.fromarray(im_copy)
cut_image.save('BOARD.png')
# import numpy as np
# b = []
# a = np.array([[1, 2],[3, 4]])
# print(type(a))
# for i in a:
#     b.append(tuple(i))
#
# print(b)