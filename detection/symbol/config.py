import numpy as np
from easydict import EasyDict as edict

config = edict()

# image processing config
config.EPS = 1e-14
config.PIXEL_MEANS = np.array([[[123.68, 116.779, 103.939]]])
config.SCALES = (600, )  # single scale training and testing
config.MAX_SIZE = 1000

config.USE_GPU_NMS = True
config.GPU_ID = 0
config.END2END = 0

# R-CNN testing
config.TEST = edict()
config.TEST.HAS_RPN = False
config.TEST.BATCH_IMAGES = 1
config.TEST.NMS = 0.3
config.TEST.DEDUP_BOXES = 1. / 16.

# RPN proposal
config.TEST.RPN_NMS_THRESH = 0.7
config.TEST.RPN_PRE_NMS_TOP_N = 6000
config.TEST.RPN_POST_NMS_TOP_N = 300
config.TEST.RPN_MIN_SIZE = 16
