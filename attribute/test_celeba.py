#!/usr/bin/env python

import argparse
import logging
import os
import cv2
import mxnet as mx
import numpy as np
from lightened_moon import lightened_moon_feature

ctx = mx.gpu(0)

def main():
    _, model_args, model_auxs = mx.model.load_checkpoint(args.model_prefix, args.epoch)
    symbol = lightened_moon_feature()
    cnt = correct_cnt = 0

    with open(args.test_list, 'r') as f:
        label = np.ones(40)
        pred = np.ones(40)
        for line in f.readlines():
            # get img and label
            line_list = line.strip('\n').split('\t')
            img_name = line_list[-1]
            logging.info("processing {}, the {}th image".format(img_name, int(cnt)))
            for label_idx in range(1,41):
                label[label_idx-1] = int(line_list[label_idx])
            img = cv2.imread(os.path.join(args.root, img_name), 0)/255.0
            img = cv2.resize(img, (args.size, args.size))
            img = np.expand_dims(np.expand_dims(img, axis=0), axis=0)
            # get pred
            model_args['data'] = mx.nd.array(img, ctx)
            exector = symbol.bind(ctx, model_args ,args_grad=None, grad_req="null", aux_states=model_auxs)
            exector.forward(is_train=False)
            exector.outputs[0].wait_to_read()
            output = exector.outputs[0].asnumpy()
            for i in range(40):
                if output[0][i] < 0:
                    pred[i] = -1
                else:
                    pred[i] = 1
            correct_cnt += (label == pred).sum()
            cnt += 1.0
        logging.info("acc is:{}".format(correct_cnt / (cnt*40.0)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/home/work/data/Face/CelebA/Img/img_celeba_cropped/img_celeba',
                        help='the root dir of celeba cropped image')
    parser.add_argument('--test-list', type=str, default='/home/work/data/Face/CelebA/Img/img_celeba_cropped/celeba_test.lst',
                        help='the test list file of CelebA cropped image, the image number is : 19667 vs 19962, for the \n'
                             'failure of face detection, Similarly, train and val is also a little small than original')
    parser.add_argument('--size', type=int, default=128,
                        help='the image size of lfw aligned image, only support squre size')
    parser.add_argument('--model-prefix', default='../model/lightened_moon/lightened_moon_fuse',
                        help='The trained model to get feature')
    parser.add_argument('--epoch', type=int, default=82,
                        help='The epoch number of model')
    args = parser.parse_args()
    logging.info(args)
    main()