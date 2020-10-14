#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/4/13 20:50
# @Author  : Zisha Zhong
# @Email   : zhongzisha@outlook.com
# @File    : test_angles.py

import numpy as np
import cv2

from functools import partial
import copy

def polygonToRotRectangle_batch(bbox, with_module=True):
    """
    :param bbox: The polygon stored in format [x1, y1, x2, y2, x3, y3, x4, y4]
            shape [num_boxes, 8]
    :return: Rotated Rectangle in format [cx, cy, w, h, theta]
            shape [num_rot_recs, 5]
    """
    # print('bbox: ', bbox)
    bbox = np.array(bbox,dtype=np.float32)
    bbox = np.reshape(bbox,newshape=(-1, 2, 4),order='F')
    # angle = math.atan2(-(bbox[0,1]-bbox[0,0]),bbox[1,1]-bbox[1,0])
    # print('bbox: ', bbox)
    angle = np.arctan2(-(bbox[:, 0,1]-bbox[:, 0,0]),bbox[:, 1,1]-bbox[:, 1,0])
    # angle = np.arctan2(-(bbox[:, 0,1]-bbox[:, 0,0]),bbox[:, 1,1]-bbox[:, 1,0])
    # center = [[0],[0]] ## shape [2, 1]
    # print('angle: ', angle)
    center = np.zeros((bbox.shape[0], 2, 1))
    for i in range(4):
        center[:, 0, 0] += bbox[:, 0,i]
        center[:, 1, 0] += bbox[:, 1,i]

    center = np.array(center,dtype=np.float32)/4.0

    # R = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], dtype=np.float32)
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float32)

    normalized = np.matmul(R.transpose((2, 1, 0)),bbox-center)


    xmin = np.min(normalized[:, 0, :], axis=1)
    # print('diff: ', (xmin - normalized[:, 0, 3]))
    # assert sum((abs(xmin - normalized[:, 0, 3])) > eps) == 0
    xmax = np.max(normalized[:, 0, :], axis=1)
    # assert sum(abs(xmax - normalized[:, 0, 1]) > eps) == 0
    # print('diff2: ', xmax - normalized[:, 0, 1])
    ymin = np.min(normalized[:, 1, :], axis=1)
    # assert sum(abs(ymin - normalized[:, 1, 3]) > eps) == 0
    # print('diff3: ', ymin - normalized[:, 1, 3])
    ymax = np.max(normalized[:, 1, :], axis=1)
    # assert sum(abs(ymax - normalized[:, 1, 1]) > eps) == 0
    # print('diff4: ', ymax - normalized[:, 1, 1])

    w = xmax - xmin + 1
    h = ymax - ymin + 1

    w = w[:, np.newaxis]
    h = h[:, np.newaxis]
    # TODO: check it
    if with_module:
        angle = angle[:, np.newaxis] % ( 2 * np.pi)
    else:
        angle = angle[:, np.newaxis]
    dboxes = np.concatenate((center[:, 0].astype(np.float), center[:, 1].astype(np.float), w, h, angle), axis=1)
    return dboxes


def polygonToRotRectangle_batch_360(quad_boxes):
    rotated_boxes = []
    for box in quad_boxes:
        box = np.array(box).reshape(4, 2)
        # box: [4x2],0:top-left,1:top-right,2:bottom-right,3:bottom-left
        p1x = (box[0, 0] + box[1, 0]) / 2
        p1y = (box[0, 1] + box[1, 1]) / 2
        p2x = (box[2, 0] + box[3, 0]) / 2
        p2y = (box[2, 1] + box[3, 1]) / 2

        if p1x>p2x:
          angle1 = np.arctan(np.abs(p1y-p2y)/np.abs(p2x-p1x))
          if p1y==p2y:
            angle1 = 0
          elif p1y>p2y:
            # 4
            angle1 = 2*np.pi - angle1
          elif p1y<p2y:
            # 1
            angle1 = angle1
        elif p1x<p2x:
          angle1 = np.arctan(np.abs(p1y-p2y)/np.abs(p2x-p1x))
          if p1y==p2y:
            angle1 = np.pi
          elif p1y>p2y:
            # 3
            angle1 = np.pi + angle1
          elif p1y<p2y:
            # 2
            angle1 = np.pi - angle1
        else:
          if p1y>p2y:
            angle1 = 1.5*np.pi
          else:
            angle1 = np.pi/2
        # rotate the four points
        rx, ry = np.mean(box, axis=0)
        rect1 = cv2.minAreaRect(box)
        x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]
        box1 = cv2.boxPoints(((x,y),(w,h),theta))  # theta in degrees
        box1 = box1.reshape([4,2])
        indexes = np.array([[0,1,2,3],[3,0,1,2],[2,3,0,1],[1,2,3,0]],dtype=np.int)
        dist = np.zeros((4,),dtype=np.float32)
        for i in range(4):
          dist[i] = np.sum(np.sqrt(np.sum(np.square(box-box1[indexes[i]]),axis=0)))
        mini = np.argmin(dist)
        box1 = box1[indexes[mini]]
        w = np.sqrt(np.sum(np.square(box1[0,:]-box1[1,:])))
        h = np.sqrt(np.sum(np.square(box1[1,:]-box1[2,:])))
        w = int(w)
        h = int(h)
        rotated_boxes.append([rx,ry,w,h,angle1])
    return np.array(rotated_boxes,dtype=np.float32).reshape(-1, 5)    # angle1 is [-np.pi, np.pi] in radians


def rotate_poly_single(h, w, new_h, new_w, rotate_matrix_T, poly):
    poly[::2] = poly[::2] - (w - 1) * 0.5
    poly[1::2] = poly[1::2] - (h - 1) * 0.5
    coords = poly.reshape(4, 2)
    new_coords = np.matmul(coords,  rotate_matrix_T) + np.array([(new_w - 1) * 0.5, (new_h - 1) * 0.5])
    rotated_polys = new_coords.reshape(-1, ).tolist()

    return rotated_polys

# TODO: refactor the single - map to whole numpy computation
def rotate_poly(h, w, new_h, new_w, rotate_matrix_T, polys):
    rotate_poly_fn = partial(rotate_poly_single, h, w, new_h, new_w, rotate_matrix_T)
    rotated_polys = list(map(rotate_poly_fn, polys))

    return rotated_polys

#
# im = np.ones((800, 800, 3), dtype=np.uint8) * 255
# quad_boxes = np.array([[0,3,1,1,4,2,3,4],
#                        [1,1,4,2,3,4,0,3],
#                        [4,2,3,4,0,3,1,1],
#                        [3,4,0,3,1,1,4,2]], dtype=np.float32) * 100
#
# rboxes1 = polygonToRotRectangle_batch(quad_boxes, with_module=False)
# rboxes2 = polygonToRotRectangle_batch_360(quad_boxes)
#
# print('rboxes1', rboxes1)
# print('rboxes2', rboxes2)
#
# for quad_box in quad_boxes:
#     rect = quad_box.astype(np.int32).reshape([-1, 1, 2])
#     xc, yc = np.mean(rect, axis=0).flatten()
#     cv2.drawContours(im, [rect], -1, color=(0, 255, 0), thickness=2)
#     cv2.circle(im, center=(rect[0, 0, 0], rect[0, 0, 1]), radius=3, color=(0, 255, 0),
#                thickness=3)
#     xc1, yc1 = (rect[0, 0, 0]+rect[1, 0, 0])//2, (rect[0, 0, 1]+rect[1, 0, 1])//2
#     cv2.line(im, (int(xc), int(yc)), (int(xc1), int(yc1)), color=(0, 255, 0))
#
#     # # rotate the four points
#     # box = quad_box.astype(np.int32).reshape([4, 2])
#     # rx, ry = np.mean(box, axis=0)
#     # rect1 = cv2.minAreaRect(box)
#     # x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]
#     # rect1 = cv2.boxPoints(((x, y), (w, h), theta))  # theta in degrees
#     # rect1 = rect1.reshape([4, 1, 2]).astype(np.int32)
#     # print(rect1)
#     # rect2 = cv2.minAreaRect(rect1)
#     # print(rect2)
#     # cv2.drawContours(im, [rect1], -1, color=(0, 0, 255), thickness=2)
#     # cv2.circle(im, center=(rect1[0, 0, 0], rect1[0, 0, 1]), radius=3, color=(0, 0, 255),
#     #            thickness=3)
#     break
#
#
#
# cv2.imwrite('test.png', im)

if False:
  quad_boxes = np.array([[0,3,1,1,4,2,3,4]], dtype=np.float32) * 100
  for angle in [90, 180, 270]:
      im = np.ones((800, 800, 3), dtype=np.uint8) * 255
      # randomly rotate the image and quad_boxes
      scale = 1.0
      # angle = 270 #np.random.rand() * 360
      h, w = im.shape[:2]
      center = (w//2, h//2)
      rotated_matrix = cv2.getRotationMatrix2D(center, -angle, scale)
      rotated_matrix_T = copy.deepcopy(rotated_matrix[:2, :2]).T

      cos = np.abs(rotated_matrix[0, 0])
      sin = np.abs(rotated_matrix[0, 1])
      new_w = h * sin + w * cos
      new_h = h * cos + w * sin
      rotated_matrix[0, 2] += (new_w - w) * 0.5
      rotated_matrix[1, 2] += (new_h - h) * 0.5
      w = int(np.round(new_w))
      h = int(np.round(new_h))

      rotated_im = cv2.warpAffine(im, rotated_matrix, (w, h), borderValue=0)

      rotated_quad_boxes = rotate_poly(im.shape[0], im.shape[1], h, w, rotated_matrix_T, quad_boxes)
      print('rotated_quad_boxes', rotated_quad_boxes)
      for quad_box in np.array(rotated_quad_boxes):
          rect = quad_box.astype(np.int32).reshape([-1, 1, 2])
          xc, yc = np.mean(rect, axis=0).flatten()
          cv2.drawContours(rotated_im, [rect], -1, color=(0, 255, 0), thickness=2)
          cv2.circle(rotated_im, center=(rect[0, 0, 0], rect[0, 0, 1]), radius=3, color=(0, 255, 0),
                     thickness=3)
          xc1, yc1 = (rect[0, 0, 0] + rect[1, 0, 0]) // 2, (rect[0, 0, 1] + rect[1, 0, 1]) // 2
          cv2.line(rotated_im, (int(xc), int(yc)), (int(xc1), int(yc1)), color=(0, 255, 0))

      cv2.imwrite('test-rot_%.3f.png'%angle, rotated_im)

im = np.ones((600, 800, 3), dtype=np.uint8) * 255
xc, yc = (100, 100)
if True:
    cv2.putText(im,
                'Text',
                (int(xc), int(yc)),
                fontFace=4,
                fontScale=2,
                color=(255, 0, 0))
if True:
  quad_boxes = np.array([[0, 3, 1, 1, 4, 2, 3, 4]], dtype=np.float32) * 100
  for angle in [0, 90, 180, 270]:
    # randomly rotate the image and quad_boxes
    scale = 1.0
    # angle = 270 #np.random.rand() * 360
    h, w = im.shape[:2]
    center = (w // 2, h // 2)
    cx, cy = w // 2, h // 2
    rotated_matrix = cv2.getRotationMatrix2D(center, -angle, scale)
    rotated_matrix_T = copy.deepcopy(rotated_matrix[:2, :2]).T

    cos = rotated_matrix[0, 0]
    sin = rotated_matrix[0, 1]
    new_w = h * sin + w * cos
    new_h = h * cos + w * sin
    rotated_matrix[0, 2] += (new_w - w) * 0.5
    rotated_matrix[1, 2] += (new_h - h) * 0.5
    w = int(np.round(new_w))
    h = int(np.round(new_h))

    rotated_im = cv2.warpAffine(im, rotated_matrix, (w, h), borderValue=0)

    print('rotated_matrix:', rotated_matrix)
    theta = np.radians(angle)
    scale = 1.0
    x0, y0 = w//2, h//2
    rotated_quad_boxes = []
    for box in quad_boxes:
      box = box.reshape(4, 2)
      box1 = []
      for bi in range(4):
        x = box[bi, 0]
        y = box[bi, 1]
        x1 = ((x - x0) * np.cos(theta)) - ((y - y0) * np.sin(theta)) + x0
        y1 = ((x - x0) * np.sin(theta)) + ((y - y0) * np.cos(theta)) + y0
        box1.append([x1, y1])
      rotated_quad_boxes.append(box1)
    rotated_quad_boxes = np.array(rotated_quad_boxes).reshape((-1, 8))

    print('rotated_quad_boxes', rotated_quad_boxes)
    for quad_box in np.array(rotated_quad_boxes):
      rect = quad_box.astype(np.int32).reshape([-1, 1, 2])
      xc, yc = np.mean(rect, axis=0).flatten()
      cv2.drawContours(rotated_im, [rect], -1, color=(0, 255, 0), thickness=2)
      cv2.circle(rotated_im, center=(rect[0, 0, 0], rect[0, 0, 1]), radius=3, color=(0, 255, 0),
                 thickness=3)
      xc1, yc1 = (rect[0, 0, 0] + rect[1, 0, 0]) // 2, (rect[0, 0, 1] + rect[1, 0, 1]) // 2
      cv2.line(rotated_im, (int(xc), int(yc)), (int(xc1), int(yc1)), color=(0, 255, 0))

    cv2.imwrite('test-rot_%.3f.png' % angle, rotated_im)

