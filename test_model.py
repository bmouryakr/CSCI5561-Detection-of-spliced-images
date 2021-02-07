from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import xml.etree.ElementTree as ET
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from lib.config import config as cfg
from lib.utils.nms_wrapper import nms
from lib.utils.test import im_detect
#from nets.resnet_v1 import resnetv1
from lib.nets.vgg16 import vgg16
from lib.utils.timer import Timer

CLASSES = ('__background__',
           'tampered')

# PLEASE specify weight files dir for vgg16
NETS = {'vgg16': ('vgg16_faster_rcnn_iter_40000.ckpt',)}
DATASETS = {'pascal_voc': ('voc_2007_trainval',), 'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}





def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        im = im[:, :, (2, 1, 0)]
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(im, aspect='equal')
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                 fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    #plt.savefig("/home/iaa/agraw208/Image_manipulation_detection-master/output.jpg")


def demo(sess, net, image_name):

    def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou

    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join('/panfs/roc/groups/11/iaa/agraw208/Img2/Img2/data/DIY_dataset/VOC2007/JPEGImages/Test', image_name)
    im = cv2.imread(im_file)
    xml_name=image_name.replace(".jpg",".xml")
    tamp=[]
    annot = os.path.join('/panfs/roc/groups/11/iaa/agraw208/Img2/Img2/data/DIY_dataset/VOC2007/Annotations/Test', xml_name)
    tree = ET.parse(annot)
    obj = tree.find('object')   
    bbox = obj.find('bndbox')
    tamp.append(int(bbox.find('xmin').text))
    tamp.append(int(bbox.find('ymin').text))
    tamp.append(int(bbox.find('xmax').text))
    tamp.append(int(bbox.find('ymax').text))




    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.1
    NMS_THRESH = 0.1
    metric_list=[]
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        if len(inds) == 0:
            metric_list=[0,0,1,0]
        else:
            iou=[]
            for i in inds:
                bb = dets[i, :4]
                iou.append(bb_intersection_over_union(tamp,bb))
            iou=np.array(iou)
            iou_mean=np.mean(iou)
            metric_list.append(iou_mean) #appending iou
            metric_list.append(np.sum((iou<0.5).astype(int))) #appending false positive
            if (np.sum(iou)==0):
                metric_list.append(1) # appending false_negative
            else:
                metric_list.append(0)
            metric_list.append(np.sum((iou>=0.5).astype(int))) #appending true positive

        vis_detections(im, cls, dets, thresh=CONF_THRESH)
    return metric_list


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    tfmodel = os.path.join('default', 'DIY_dataset', 'default', NETS[demonet][0])

    # if not os.path.isfile(tfmodel + '.meta'):
    #     print(tfmodel)
    #     raise IOError(('{:s} not found.\nDid you download the proper networks from '
    #                    'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True



    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16(batch_size=1)
    # elif demonet == 'res101':
        # net = resnetv1(batch_size=1, num_layers=101)
    else:
        raise NotImplementedError
    net.create_architecture(sess, "TEST", 2,
                            tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    iou = []
    fp = 0 
    tp = 0
    fn = 0
    poor=0
    good=0
    excel=0
    print('Loaded network {:s}'.format(tfmodel))
    i=0
    for file in os.listdir("/panfs/roc/groups/11/iaa/agraw208/Img2/Img2/data/DIY_dataset/VOC2007/JPEGImages/Test"):
        i=i+1
        if file.endswith(".jpg") or file.endswith(".png"):
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print('Demo for lib/layer_utils/{}'.format(file))
            metric_list = demo(sess, net, file)
            plt.savefig("output/"+file)
            iou.append(metric_list[0])
            if(metric_list[0]<=0.4):
                poor=poor+1
            elif(metric_list[0]<=0.8):
                good=good+1
            else:
                excel=excel+1
            fp+=metric_list[1]
            fn+=metric_list[2]
            tp+=metric_list[3]
    f1 = tp/(tp+0.5*(fp+fn))
    iou = np.array(iou)
    iou_mean = np.mean(iou)
    print('avg_IOU = ' + str(iou_mean))
    print('Poor: ',poor,' Good: ',good,' Excellent: ',excel)
    print('f1 ='+ str(f1))
