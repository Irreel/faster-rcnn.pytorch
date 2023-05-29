import cv2
import numpy as np


def vis_proposals(im, boxes, thresh=0.8):
    """Visual proposal box from RPN"""
    for i in range(np.minimum(10, boxes.shape[0])):
        bbox = tuple(int(np.round(x)) for x in boxes[i, :4])
        cv2.rectangle(im, bbox[0:2], bbox[2:4], (255, 165, 0), 1)
            # cv2.putText(im, '%s: %.3f' % (class_name, score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
            #             1.0, (0, 0, 255), thickness=1)
    return im