# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick, Sean Bell and Xinlei Chen
# Modified by Jianwei Zhang
# --------------------------------------------------------
import numpy as np


def generate_anchors_pre(height, width, feat_stride, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
    """ A wrapper function to generate anchors given different scales
      Also return the number of anchors in variable 'length'

    Parameters
    ----------
    height: Height of the feature map
    width: Width of the feature map
    feat_stride: feature stride
    anchor_scales: default is (8, 16, 32) which means (8^2, 16^2, 32^2)
    anchor_ratios: default is (0.5, 1, 2)

    Returns
    -------
    anchors: a float 2D array with shape of [K x A, 4], where K = w x h,
                A = len(anchor_ratio) x len(anchor_scale)
    length: length of the all anchors, value = (w x h) x (3 x 3)
    """
    anchors = generate_anchors(ratios=np.array(
        anchor_ratios), scales=np.array(anchor_scales))
    A = anchors.shape[0]    # A = 3
    shift_x = np.arange(0, width) * feat_stride
    shift_y = np.arange(0, height) * feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    # ravel(): Return a flattened array.
    # shift length of [x1, y1, x2, y2]
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()
    K = shifts.shape[0]  # w * h
    # width changes faster, so here it is H, W, C
    anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    anchors = anchors.reshape((K * A, 4)).astype(np.float32, copy=False)
    length = np.int32(anchors.shape[0])     # K*A = (w*h) * 3

    return anchors, length


# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

# array([[ -83.,  -39.,  100.,   56.],
#       [-175.,  -87.,  192.,  104.],
#       [-359., -183.,  376.,  200.],
#       [ -55.,  -55.,   72.,   72.],
#       [-119., -119.,  136.,  136.],
#       [-247., -247.,  264.,  264.],
#       [ -35.,  -79.,   52.,   96.],
#       [ -79., -167.,   96.,  184.],
#       [-167., -343.,  184.,  360.]])

def generate_anchors(base_size=16,
                     ratios=np.array([0.5, 1, 2]),
                     scales=2 ** np.arange(3, 6)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """

    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    #                                 h : w
    # [[ -3.5   2.   18.5  13. ]    0.5 : 1
    #  [  0.    0.   15.   15. ]      1 : 1
    #  [  2.5  -3.   12.5  18. ]]     2 : 1

    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in range(ratio_anchors.shape[0])])
    # h * scales, w * scales
    #                               ratio     scales      size     area 16*16*
    # [[ -84.  -40.   99.   55.]    0.5:1       8      [ 96, 184]    8**2
    #  [-176.  -88.  191.  103.]    0.5:1      16      [192, 368]   16**2
    #  [-360. -184.  375.  199.]]   0.5:1      32      [384, 736]   32**2
    # [[ -56.  -56.   71.   71.]      1:1       8      [128, 128]    8**2
    #  [-120. -120.  135.  135.]      1:1      16      [256, 256]   16**2
    #  [-248. -248.  263.  263.]]     1:1      32      [512, 512]   32**2
    # [[ -36.  -80.   51.   95.]      2:1       8      [176,  88]    8**2
    #  [ -80. -168.   95.  183.]      2:1      16      [352, 176]   16**2
    #  [-168. -344.  183.  359.]]     2:1      32      [704, 352]   32**2

    return anchors


# width height centers
def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    #     16,16, 7.5,   7.5
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors


# ratio enumerate
def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h  # 256 = 16*16
    size_ratios = size / ratios  # 256 / [0.5, 1, 2] = [512, 256, 128]
    ws = np.round(np.sqrt(size_ratios))  # [23., 16., 11.]
    hs = np.round(ws * ratios)  # [12., 16., 22.]
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors
