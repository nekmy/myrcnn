import colorsys
import warnings
import random

import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt
from matplotlib import patches
from skimage import transform
from skimage.measure import find_contours
from matplotlib.patches import Polygon
from matplotlib import animation
import scipy
import cv2

def demold_box(box, shape):
    # box: 正規化されたbox
    h, w, _ = shape
    scale = (h - 1, w - 1, h - 1, w - 1)
    shift = (0, 0, 1, 1)
    return (box * scale + shift).astype(np.int32)

def window_image(image):
    h, w, _ = image.shape
    image = image

def unmold_mask(mask, box, shape):
    h, w, _ = shape
    y1, x1, y2, x2 = box
    mask = resize(mask, (y2 - y1, x2 - x1))
    mask = np.where(mask>=0.5, 1, 0).astype(np.bool)
    full_mask = np.zeros((h, w))
    full_mask[y1:y2, x1:x2] = mask
    return full_mask

def resize(image, shape, order=1, mode="constant", cval=0, clip=True,
           preserve_range=False, anti_aliasing=False, anti_aliasing_sigma=None):
    return transform.resize(
            image, shape[:2],
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range, anti_aliasing=anti_aliasing,
            anti_aliasing_sigma=anti_aliasing_sigma)
    #return transform.resize(image, shape[:2], mode="constant")

def compose_image_metas(image_id, original_image_shape, molded_image_shape, window, scale, activate_class_ids):
    # image_id: inferenceなら0
    # original_image_shape: 入力時のshape(H, W, C)
    # molded_image_shape: moldしたあとのshape(2*pad_h+H, 2*pad_w+W, C)
    # sclae: moldした際のscale
    # activate_class_ids
    image_metas = np.array(
        [image_id]+
        list(original_image_shape)+
        list(molded_image_shape)+
        list(window)+
        [scale]+
        list(activate_class_ids)
    )
    return image_metas

def metas_converter(metas):
    return {
        "image_id": metas[:, 0],
        "original_shape": metas[:, 1:4],
        "molded_shape": metas[:, 4:7],
        "window": metas[:, 7:11],
        "scale": metas[:, 11],
        "activate_class_ids": metas[:, 12:],
    }

def dictize_metas(image_metas):
    metas = {
        "image_shape": image_metas[0, :],
        "window": image_metas[1, :]
    }
    return metas

def mold_images(images, max_dim, min_dim, config):
    molded_images = []
    image_metas = []
    windows = []
    pads = []
    scales = []
    for image in images:
        molded_image, image_meta, window, pad, scale = mold_image(image, max_dim, min_dim, config)
        molded_image = molded_image - config.MEAN_PIXEL
        molded_images.append(molded_image)
        image_metas.append(image_meta)
        windows.append(window)
        pads.append(pad)
        scales.append(scale)
    image_metas = np.stack(image_metas)
    return np.array(molded_images), image_metas, np.array(windows), pads, scales

def mold_image(image, max_dim, min_dim, config):
    # imageを一辺の長さがmax_dimの正方形の画像に変える.
    # 画像は長辺がmax_dim以下, 短辺がmin_dim以上になるようにする.
    # なお, 両条件が同時に成立しない場合は前者を優先する.
    # アスペクト比は保持される.
    # 空白部分は0とする.
    dtype = image.dtype
    original_image_shape = h, w, _ = image.shape
    # 各辺が少なくともmin_dim以上のサイズになるようにする.
    scale = max(1, min_dim / min(h, w))
    # 各辺が少なくともmax_dim以下のサイズになるようにする.
    max_shape = max(h, w)
    if round(max_shape * scale) > max_dim:
        scale = max_dim / max_shape
    
    image = transform.resize(image, (round(h * scale), round(w * scale)), preserve_range=True, anti_aliasing=False, mode="constant")
    #resize後のサイズ
    h, w, _ = image.shape

    top_pad = (max_dim - h) // 2
    bottom_pad = max_dim - top_pad - h
    left_pad = (max_dim - w) // 2
    right_pad = max_dim - left_pad - w
    pad = ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0))
    image = np.pad(image, pad, mode="constant", constant_values=0)
    # moldedの中のどの位置にoriginal_imageがあるか
    window = (top_pad, left_pad, h + top_pad, w + left_pad)
    meta = compose_image_metas(0, original_image_shape, image.shape, window, scale, np.ones((config.NUM_CLASSES,)))

    return image.astype(dtype), meta, window, pad, scale

# backbone_strideはbackbone_shape等とともに選ばれる.
def make_anchors_with_single_scale(scale, ratios, backbone_shape, backbone_stride):
    # backbone_shapeはアンカーの数
    shift_x = np.arange(0, backbone_shape[0], 1) * backbone_stride
    shift_y = np.arange(0, backbone_shape[1], 1) * backbone_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    heights = scale / np.sqrt(ratios)
    widths = scale * np.sqrt(ratios)
    # グリッド上の各点, 各幅毎に処理を行う.
    heights, center_y = np.meshgrid(heights, shift_y)
    widths, center_x = np.meshgrid(widths, shift_x)
    center_box = np.stack([center_y, center_x], axis=2).reshape(-1, 2)
    delta_box = np.stack([heights, widths], axis=2).reshape(-1, 2)
    boxes = np.concatenate([center_box - 0.5 * delta_box, center_box + 0.5 * delta_box], axis=1)
    return boxes

def make_anchors(scales, ratios, image_shape, backbone_strides):
    # anchorsをまとめて作る.
    backbone_shape = [(np.math.ceil(image_shape[0] / stride), 
                      np.math.ceil(image_shape[1] / stride)) for stride in backbone_strides]
    anchors = []
    for i in range(len(scales)):
        scale = scales[i]
        backbone_stride = backbone_strides[i]
        anchors.append(make_anchors_with_single_scale(scale, ratios, backbone_shape[i], backbone_stride))
    anchors = np.concatenate(anchors, axis=0)
    anchors = norm_boxes(anchors, image_shape[:2])
    return anchors

def make_anchors_nomold(scales, ratios, image_shape, backbone_strides):
    # anchorsをまとめて作る.
    backbone_shape = [(np.math.ceil(image_shape[0] / stride), 
                      np.math.ceil(image_shape[1] / stride)) for stride in backbone_strides]
    anchors = []
    for i in range(len(scales)):
        scale = scales[i]
        backbone_stride = backbone_strides[i]
        anchors.append(make_anchors_with_single_scale(scale, ratios, backbone_shape[i], backbone_stride))
    anchors = np.concatenate(anchors, axis=0)
    #anchors = norm_boxes(anchors, image_shape[:2])
    return anchors

def norm_boxes(boxes, shape):
    # 引数のboxexはピクセルの端を意味する.
    # これを対応するピクセルの番地として表したものに変換する(shift).
    h, w = shape
    scale = (h - 1, w - 1, h - 1, w - 1)
    shift = (0, 0, -1, -1)
    return ((boxes + shift) / scale).astype(np.float32)

def denorm_boxes(boxes, shape):
    # 引数のboxexはピクセルの端を意味する.
    # これを対応するピクセルの番地として表したものに変換する(shift).
    h, w = shape
    scale = (h - 1, w - 1, h - 1, w - 1)
    shift = (0, 0, 1, 1)
    return boxes * scale + shift

def sift_boxes(boxes, sift):
    y1, x1, y2, x2 = np.split(boxes, 4, axis=1)
    dy, dx = sift
    y1 = y1 + dy
    x1 = x1 + dx
    y2 = y2 + dy
    x2 = x2 + dx
    return np.concatenate([y1, x1, y2, x2], axis=1)

def norm_boxes_tf(boxes, shape):
    # 引数のboxexはピクセルの端を意味する.
    # これを対応するピクセルの番地として表したものに変換する(shift).
    h, w = shape[:, 0], shape[:, 1]
    scale = tf.concat((h - 1, w - 1, h - 1, w - 1), axis=-1)
    shift = tf.concat(tf.constant([0., 0., -1., -1.]), axis=-1)
    return tf.cast(((boxes + shift) / scale), tf.float32)

def intersection_over_union(box1, box2):
    y1_1 = box1[:, 0]
    x1_1 = box1[:, 1]
    y1_2 = box1[:, 2]
    x1_2 = box1[:, 3]
    y1_1, x1_1, y1_2, x1_2 = [tf.reshape(c, (-1, 1)) for c in [y1_1, x1_1, y1_2, x1_2]]
    y2_1 = box2[:, 0]
    x2_1 = box2[:, 1]
    y2_2 = box2[:, 2]
    x2_2 = box2[:, 3]
    y2_1, x2_1, y2_2, x2_2 = [tf.reshape(c, (1, -1)) for c in [y2_1, x2_1, y2_2, x2_2]]
    y1 = tf.maximum(y1_1, y2_1)
    x1 = tf.maximum(x1_1, x2_1)
    y2 = tf.minimum(y1_2, y2_2)
    x2 = tf.minimum(x1_2, x2_2)
    areas_of_union = tf.maximum(y2 - y1, 0) * tf.maximum(x2 - x1, 0)
    areas1 = (y1_2 - y1_1) * (x1_2 - x1_1)
    areas2 = (y2_2 - y2_1) * (x2_2 - x2_1)
    areas_of_overlap = areas1 + areas2 - areas_of_union
    iou = areas_of_union / areas_of_overlap
    return iou

def iou_np(box1, box2):
    y1_1 = box1[:, 0]
    x1_1 = box1[:, 1]
    y1_2 = box1[:, 2]
    x1_2 = box1[:, 3]
    y1_1, x1_1, y1_2, x1_2 = [np.reshape(c, (-1, 1)) for c in [y1_1, x1_1, y1_2, x1_2]]
    y2_1 = box2[:, 0]
    x2_1 = box2[:, 1]
    y2_2 = box2[:, 2]
    x2_2 = box2[:, 3]
    y2_1, x2_1, y2_2, x2_2 = [np.reshape(c, (1, -1)) for c in [y2_1, x2_1, y2_2, x2_2]]
    y1 = np.maximum(y1_1, y2_1)
    x1 = np.maximum(x1_1, x2_1)
    y2 = np.minimum(y1_2, y2_2)
    x2 = np.minimum(x1_2, x2_2)
    areas_of_union = np.maximum(y2 - y1, 0) * np.maximum(x2 - x1, 0)
    areas1 = (y1_2 - y1_1) * (x1_2 - x1_1)
    areas2 = (y2_2 - y2_1) * (x2_2 - x2_1)
    areas_of_overlap = areas1 + areas2 - areas_of_union
    iou = areas_of_union / areas_of_overlap
    return iou

def apply_deltas(anchors, deltas):
    # anchors[N, h_a*w_a*k, 4]
    # x = x_a + dx
    # y = y_a + dy
    # h = h_a * exp(dh)
    # w = w_a * exp(dw)
    y1_a = anchors[:, 0]
    x1_a = anchors[:, 1]
    y2_a = anchors[:, 2]
    x2_a = anchors[:, 3]
    dy = deltas[:, 0]
    dx = deltas[:, 1]
    dh = deltas[:, 2]
    dw = deltas[:, 3]
    y_a = (y1_a + y2_a) * 0.5
    x_a = (x1_a + x2_a) * 0.5
    h_a = (y2_a - y1_a)
    w_a = (x2_a - x1_a)
    y = y_a + h_a * dy
    x = x_a + w_a * dx
    h = h_a * tf.exp(dh)
    w = w_a * tf.exp(dw)
    y1 = y - h * 0.5
    x1 = x - w * 0.5
    y2 = y + h * 0.5
    x2 = x + w * 0.5
    box = tf.stack([y1, x1, y2, x2], axis=1)
    return box

def apply_deltas_np(anchors, deltas):
    # anchors[N, h_a*w_a*k, 4]
    # x = x_a + dx
    # y = y_a + dy
    # h = h_a * exp(dh)
    # w = w_a * exp(dw)
    y1_a = anchors[:, 0]
    x1_a = anchors[:, 1]
    y2_a = anchors[:, 2]
    x2_a = anchors[:, 3]
    dy = deltas[:, 0]
    dx = deltas[:, 1]
    dh = deltas[:, 2]
    dw = deltas[:, 3]
    y_a = (y1_a + y2_a) * 0.5
    x_a = (x1_a + x2_a) * 0.5
    h_a = (y2_a - y1_a)
    w_a = (x2_a - x1_a)
    y = y_a + h_a * dy
    x = x_a + w_a * dx
    h = h_a * np.exp(dh)
    w = w_a * np.exp(dw)
    y1 = y - h * 0.5
    x1 = x - w * 0.5
    y2 = y + h * 0.5
    x2 = x + w * 0.5
    box = np.stack([y1, x1, y2, x2], axis=1)
    return box

def clip_box(boxes, window):
    y1 = tf.maximum(boxes[:, 0], window[0])
    x1 = tf.maximum(boxes[:, 1], window[1])
    y2 = tf.minimum(boxes[:, 2], window[2])
    x2 = tf.minimum(boxes[:, 3], window[3])
    boxes = tf.stack([y1, x1, y2, x2], axis=1)
    return boxes

def clip_box_np(boxes, window):
    y1 = np.maximum(boxes[:, 0], window[0])
    x1 = np.maximum(boxes[:, 1], window[1])
    y2 = np.minimum(boxes[:, 2], window[2])
    x2 = np.minimum(boxes[:, 3], window[3])
    boxes = np.stack([y1, x1, y2, x2], axis=1)
    return boxes

def non_maximum_suppression_old(boxes, score, max_k=None, iou_min=0.5, sorted=False):
    # boxがかぶっているもののうちscoreが最大のもののインデックスを返す.
    overlaps = intersection_over_union(boxes, boxes)
    overlaps_bools = overlaps > iou_min
    if type(score) == np.ndarray:
        # iが入力, jが出力
        # 自分(j列)よりもscoreが大きい, かつ被っている相手(i行)がいるかどうか.
        covers = overlaps_bools * (score.reshape(-1, 1) > score.reshape(1, -1))
        # 自分よりもscoreが大きいかつ被っている相手がいればTrue.
        top = np.sum(covers, axis=0) == False
        pre_top = np.ones((score.shape[0],), dtype=np.bool)
        # 更新しても変わらなくなれば終わり.
        while top != pre_top:
            pre_top = top
            top = np.dot(top.reshape(1, -1), covers)
        indices = np.where(top)[0]
        score = score[indices]
        # scoreの降順
        argindex = np.argsort(score)[::-1]
        # scoreの大きいindices
        indices = indices[argindex][:max_k]
        if not sorted:
            # indicesの昇順に戻す.
            indices = np.sort(indices)
    elif type(score) == tf.Tensor:
        larger_than_j = (tf.reshape(score, (-1, 1)) > tf.reshape(score, (1, -1)))
        covers = tf_logical((overlaps_bools, larger_than_j), fn=tf.multiply)
        top = tf.logical_not(tf_logical((covers,), {"axis": 1}, fn=tf.reduce_sum))
        pre_top = tf.ones((score.shape[0],), dtype=tf.bool)
        while top != pre_top:
            pre_top = top
            #top = tf.matmul(top.reshape(1, -1), covers)
            top = tf_logical((tf.reshape(top, (1, -1)), covers), fn=tf.matmul)
        indices = tf.where(top)[0]
        if max_k is None:
            max_k = len(score)
        score = tf.gather(score, indices)
        argindex = tf.nn.top_k(score).indices
        indices = tf.gather(indices, argindex)[:max_k]
        if not sorted:
            indices = tf.nn.top_k(indices)[::-1]
    return indices
#pre_nms_box, scores, self.proposal_count, self.nms_threshold, name="rpn_non_max_suppression"
def non_max_suppression(boxes, score, max_k=None, iou_min=0.5, sorted=False, name=None):
    # boxがかぶっているもののうちscoreが最大のもののインデックスを返す.
    overlaps = intersection_over_union(boxes, boxes)
    overlaps_bools = overlaps > iou_min
    if type(score) == np.ndarray:
        # iが入力, jが出力
        # 自分(j列)よりもscoreが大きい, かつ被っている相手(i行)がいるかどうか.
        covers = overlaps_bools * (score.reshape(-1, 1) > score.reshape(1, -1))
        # 自分よりもscoreが大きいかつ被っている相手がいればTrue.
        top = np.sum(covers, axis=0) == False
        pre_top = np.ones((score.shape[0],), dtype=np.bool)
        # 更新しても変わらなくなれば終わり.
        def body():
            pre_top = top
            top = np.dot(top.reshape(1, -1), covers)
        cond = lambda : top != pre_top
        tf.while_loop(cond, body, ())
            
        indices = np.where(top)[0]
        score = score[indices]
        # scoreの降順
        argindex = np.argsort(score)[::-1]
        # scoreの大きいindices
        indices = indices[argindex][:max_k]
        if not sorted:
            # indicesの昇順に戻す.
            indices = np.sort(indices)
    elif type(score) == tf.Tensor:
        larger_than_j = tf.greater(tf.reshape(score, (-1, 1)), tf.reshape(score, (1, -1)))
        covers = tf_logical((overlaps_bools, larger_than_j), fn=tf.multiply)
        top = tf.logical_not(tf_logical((covers,), {"axis": 0}, fn=tf.reduce_sum))
        pre_top = tf.ones((tf.shape(score)[0],), dtype=tf.bool)
        def body(top, pre_top):
            pre_top = top
            top = tf.logical_not(tf_logical((tf.reshape(top, (1, -1)), covers), fn=tf.matmul))
            top = tf.reshape(top, (-1,))
            #top = tf.multiply(tf.reshape(top, (1, -1)), covers)
            return top, pre_top
        #cond = lambda i: top != pre_top
        def cond(top, pre_top):
            not_eq = tf.not_equal(top, pre_top)
            b = tf.reduce_sum(tf.cast(not_eq, tf.int16))
            return tf.cast(b, tf.bool)

        top, pre_top = tf.while_loop(cond, body, (top, pre_top))
        #indices = tf.reshape(tf.where(top), (-1,))
        indices = tf.where(top)[:, 0]
        if max_k is None:
            max_k = len(indices)

        max_k = tf.minimum(max_k, tf.shape(indices)[0])
        score = tf.gather(score, indices)
        argindex = tf.nn.top_k(score, k=max_k).indices
        indices = tf.gather(indices, argindex)[:max_k]
        if False:
            indices = tf.nn.top_k(indices, k=max_k)[::-1]
    return indices#, top#, covers, overlaps, overlaps_bools

def tf_logical(args=(), kwargs={}, fn=None):
    # bool型同士の計算を0, 1にして行う. boolで返す.
    args = [tf.cast(x, tf.int32) for x in args]
    outputs = fn(*args, **kwargs)
    return tf.cast(outputs, tf.bool)

class NMSLayer(tf.keras.layers.Layer):
    def __init__(self):
        self.out = 1
        pass
    
    def call(self):
        return self.out
    


def batch_slice(inputs, batch_size, function, names=None):
    # バッチ毎に行わなければならない処理を行う.
    # バッチサイズが分かっていなければこの処理は行えない.
    outputs_slices = []
    #length = len(inputs)
    for i in range(batch_size):
        # inputsのうちi番目のbatchのリストを取り出す.[]
        inputs_slices = [input_elem[i] for input_elem in inputs]
        output = function(*inputs_slices)
        if not type(output) in (tuple, list):
            output = [output]
        outputs_slices.append(output)
    
    length = len(outputs_slices[0])
    # outputs = [(a_1, b_1), (a_2, b_2), (a_3, b_3)] => [(a_1, a_2, a_3), (b_1, b_2, b_3)]
    outputs_slices = zip(*outputs_slices)
    if names == None:
        names = (None,) * length
    outputs = ([tf.stack(o, axis=0, name=n) for o, n in zip(outputs_slices, names)])
    if len(outputs) == 1:
        outputs = outputs[0]
    return outputs



def make_deltas(rois, boxes):
    y1_1, x1_1, y1_2, x1_2 = tf.split(rois, 4, axis=1)
    y2_1, x2_1, y2_2, x2_2 = tf.split(boxes, 4, axis=1)
    y1 = (y1_1 + y1_2) * 0.5
    x1 = (x1_1 + x1_2) * 0.5
    y2 = (y2_1 + y2_2) * 0.5
    x2 = (x2_1 + x2_2) * 0.5
    h1 = (y1_2 - y1_1)
    w1 = (x1_2 - x1_1)
    h2 = (y2_2 - y2_1)
    w2 = (x2_2 - x2_1)
    dy = (y2 - y1) / h1
    dx = (x2 - x1) / w1
    dh = tf.log(h2 / h1)
    dw = tf.log(w2 / w1)
    deltas = tf.concat([dy, dx, dh, dw], axis=1)
    return deltas

def random_colors(N):
    brightness = 1.0
    hsvs = [(i/3.3, 1, 1) for i in range(N)]

    rgbs = [colorsys.hsv_to_rgb(*hsv) for hsv in hsvs]
    return rgbs

def display(ax, image, boxes, cls_ids, scores, masks, figsize, cls_names, colors=None, save_name="", show=True):
    #ax = fig.add_subplot(1, 1, 1)
    ax.cla()
    masked_image = image.copy().astype(np.uint32)
    assert boxes.shape[0] == cls_ids.shape[0] == scores.shape[0] == masks.shape[0]
    N = boxes.shape[0]
    num_color = len(cls_names)
    colors = colors if colors else random_colors(num_color)
    for i in range(N):
        y1, x1, y2, x2 = boxes[i]
        """
        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor=colors[i], linewidth=20,
            alpha=0.7, linestyle="dashed", facecolor='none')
        """
        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor=colors[cls_ids[i]],
            alpha=0.7, linestyle="dashed", facecolor='none')
        ax.add_patch(p)

        caption = "{}:{:.3f}".format(cls_names[cls_ids[i]], scores[i])
        #ax.text(x1, y1+8, caption, fontsize=100)
        ax.text(x1, y1+8, caption, fontsize=10, color=(1.0, 1.0, 1.0))
        # ここでマスクを塗る.
        mask = masks[i]
        padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for vers in contours:
            c = np.fliplr(vers) - 1
            #p = Polygon(c, facecolor="none", edgecolor=colors[i], linewidth=20)
            p = Polygon(c, facecolor="none", edgecolor=(1.0, 1.0, 1.0))
            ax.add_patch(p)

        masked_image = apply_mask(masked_image, masks[i], colors[cls_ids[i]], alpha=0.7)
    if show:
        plt.imshow(masked_image.astype(np.uint8))
    if save_name:
        plt.savefig(save_name)
    plt.pause(0.01)

def display_CV2(image, boxes, cls_ids, scores, masks, figsize, cls_names, colors=None):
    masked_image = image.copy().astype(np.uint8)
    assert boxes.shape[0] == cls_ids.shape[0] == scores.shape[0] == masks.shape[0]
    N = boxes.shape[0]
    colors = colors if colors else random_colors(N)
    for i in range(N):
        y1, x1, y2, x2 = boxes[i]
        cv2.rectangle(masked_image, (2*x1, 2*y1), (2*x2, 2*y2), colors[i], lineType=cv2.LINE_AA, shift=1)
        caption = "{}:{:.3f}".format(cls_names[cls_ids[i]], scores[i])
        #ax.text(x1, y1+8, caption, fontsize=100)
        cv2.putText(masked_image, caption, (x1, y1+8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255,255))
        # ここでマスクを塗る.
        mask = masks[i]
        padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        masked_image = apply_mask(masked_image, masks[i], colors[i], alpha=0.5)
        contours, _ = cv2.findContours(padded_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        epsilon = 0.9 * cv2.arcLength(contours[0], True)
        app = cv2.approxPolyDP(contours[0], epsilon, True)
        masked_image = cv2.polylines(masked_image, [app], True, colors[i], thickness=10)
        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor=colors[i],
            alpha=0.7, linestyle="dashed", facecolor='none')

    cv2.imshow("predict", masked_image)
    cv2.waitKey(1)

def apply_mask(image, mask, color, alpha):
    for c in range(len(color)):
        image[:, :, c] = np.where(mask==1,
                                image[:, :, c] * (1 - alpha) + color[c] * alpha * 255,
                                image[:, :, c])
    return image

def make_gt_match_old(anchors, gt_boxes):
    iou = iou_np(anchors, gt_boxes)
    indices = [np.arange(np.shape(anchors)[0]), np.argmax(iou, axis=1)]
    # anchorごとの最大のiou
    # ide: どのanchorか
    # indices: どのgt_boxか
    maxiou = iou[indices]
    positive_ids = np.where(maxiou >= 0.7)[0]
    negaitive_ids = np.where(maxiou < 0.3)[0]
    rpn_gt_match = np.zeros((np.shape(anchors)[0]))
    rpn_gt_match[positive_ids] = 1
    rpn_gt_match[negative_ids] = -1
    # positiveなanchor毎に最も近いgt_boxのインデックスを用意する.
    positive_indices = indices[positive_ids]
    rpn_gt_boxes = gt_boxes[positive_indices]
    rpn_anchors = anchors[positive_ids]
    gt_y = (gt_boxes[:, 2] + gt_boxes[:, 0]) / 2
    gt_x = (gt_boxes[:, 3] + gt_boxes[:, 1]) / 2
    gt_h = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_w = gt_boxes[:, 3] - gt_boxes[:, 1]
    a_y = (rpn_anchors[:, 2] + rpn_anchors[:, 0]) / 2
    a_x = (rpn_anchors[:, 3] + rpn_anchors[:, 1]) / 2
    a_h = rpn_anchors[:, 2] - rpn_anchors[:, 0]
    a_w = rpn_anchors[:, 3] - rpn_anchors[:, 1]
    dy = gt_y - a_y
    dx = gt_x - a_x
    dh = np.log(gt_h / a_h)
    dw = np.log(gt_w / a_w)
    rpn_gt_deltas = np.concatenate([dy, dx, dh, dw], axis=1)
    gap = np.shape(anchors)[0] - np.shape(rpn_anchors)[0]
    rpn_gt_deltas = np.pad(rpn_gt_deltas, ((0, gap), (0, 0)))

    return rpn_gt_match, rpn_gt_deltas

def L1_smooth(y1, y2):
    abs_diff = tf.abs(y2 - y1)
    mask_liner = tf.cast(abs_diff > 1, tf.float32)
    loss = 0.5 * (1 - mask_liner) * abs_diff ** 2 + mask_liner * (abs_diff - 0.5)
    return loss

def pack_gt_deltas(x, row, batch_size):
    pack = []
    for i in range(batch_size):
        pack.append(x[i, :row[i]])
    return tf.concat(pack, axis=0)

def make_matchs(anchors, gt_boxes, config):
    matchs = np.zeros((anchors.shape[0],), dtype=np.int)
    deltas = np.zeros_like(anchors)
    if len(gt_boxes):
        iou = iou_np(anchors, gt_boxes)
        iou_argmax = np.argmax(iou, axis=1)
        iou_max = np.max(iou, axis=1)
        positive_idx = np.where(iou_max > 0.7)
        negative_idx = np.where(iou_max < 0.3)
        gt_max_idx = np.argwhere(iou==np.max(iou, axis=0))[:, 0]
        matchs[gt_max_idx] = 1
        matchs[positive_idx] = 1
        matchs[negative_idx] = -1
        """
        import pickle
        with open("rpn_matchs_origin.pkl", "rb") as f:
            d = pickle.load(f)
        """
        idx = np.where(matchs==-1)[0]
        extra = len(idx) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE - np.sum(matchs==1))
        if extra:
            extra = np.min([extra, len(np.where(matchs==-1)[0])])
            ix = np.random.choice(np.where(matchs==-1)[0], extra, replace=False)
            matchs[ix] = 0
        #matchs = d["rpn_match"]
        target_boxes = gt_boxes[iou_argmax]
        y1_a, x1_a, y2_a, x2_a = np.split(anchors, 4, axis=1)
        y_a = (y1_a + y2_a) * 0.5
        x_a = (x1_a + x2_a) * 0.5
        h_a = y2_a - y1_a
        w_a = x2_a - x1_a
        y1_gt, x1_gt, y2_gt, x2_gt = np.split(target_boxes, 4, axis=1)
        y_gt = (y1_gt + y2_gt) * 0.5
        x_gt = (x1_gt + x2_gt) * 0.5
        h_gt = y2_gt - y1_gt
        w_gt = x2_gt - x1_gt
        dy = (y_gt - y_a) / h_a
        dx = (x_gt - x_a) / w_a
        dh = np.log(h_gt / h_a)
        dw = np.log(w_gt / w_a)
        deltas = np.concatenate([dy, dx, dh, dw], axis=1) / config.RPN_BBOX_STD_DEV
    else:
        matchs = np.ones_like(matchs) * -1
    # 追加
    ix = np.where(matchs==1)[0]
    deltas = deltas[ix]
    return matchs, deltas

# 流用
def build_rpn_targets(image_shape, anchors, gt_class_ids, gt_boxes, config):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.

    anchors: [num_anchors, (y1, x1, y2, x2)]
    gt_class_ids: [num_gt_boxes] Integer class IDs.
    gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]

    Returns:
    rpn_match: [N] (int32) matches between anchors and GT boxes.
               1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    """
    # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    # RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
    rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    crowd_ix = np.where(gt_class_ids < 0)[0]
    if crowd_ix.shape[0] > 0:
        # Filter out crowds from ground truth class IDs and boxes
        non_crowd_ix = np.where(gt_class_ids > 0)[0]
        crowd_boxes = gt_boxes[crowd_ix]
        gt_class_ids = gt_class_ids[non_crowd_ix]
        gt_boxes = gt_boxes[non_crowd_ix]
        # Compute overlaps with crowd boxes [anchors, crowds]
        crowd_overlaps = iou_np(anchors, crowd_boxes)
        crowd_iou_max = np.amax(crowd_overlaps, axis=1)
        no_crowd_bool = (crowd_iou_max < 0.001)
    else:
        # All anchors don't intersect a crowd
        no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)

    # Compute overlaps [num_anchors, num_gt_boxes]
    overlaps = iou_np(anchors, gt_boxes)

    # Match anchors to GT Boxes
    # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
    # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
    # Neutral anchors are those that don't match the conditions above,
    # and they don't influence the loss function.
    # However, don't keep any GT box unmatched (rare, but happens). Instead,
    # match it to the closest anchor (even if its max IoU is < 0.3).
    #
    # 1. Set negative anchors first. They get overwritten below if a GT box is
    # matched to them. Skip boxes in crowd areas.
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1
    # 2. Set an anchor for each GT box (regardless of IoU value).
    # If multiple anchors have the same IoU match all of them
    gt_iou_argmax = np.argwhere(overlaps == np.max(overlaps, axis=0))[:,0]
    rpn_match[gt_iou_argmax] = 1
    # 3. Set anchors with high overlap as positive.
    rpn_match[anchor_iou_max >= 0.7] = 1

    # Subsample to balance positive and negative anchors
    # Don't let positives be more than half the anchors
    ids = np.where(rpn_match == 1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        # Reset the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    # Same for negative proposals
    ids = np.where(rpn_match == -1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE -
                        np.sum(rpn_match == 1))
    if extra > 0:
        # Rest the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    # For positive anchors, compute shift and scale needed to transform them
    # to match the corresponding GT boxes.
    ids = np.where(rpn_match == 1)[0]
    ix = 0  # index into rpn_bbox
    # TODO: use box_refinement() rather than duplicating the code here
    for i, a in zip(ids, anchors[ids]):
        # Closest gt box (it might have IoU < 0.7)
        gt = gt_boxes[anchor_iou_argmax[i]]

        # Convert coordinates to center plus width/height.
        # GT Box
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        # Anchor
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w

        # Compute the bbox refinement that the RPN should predict.
        rpn_bbox[ix] = [
            (gt_center_y - a_center_y) / a_h,
            (gt_center_x - a_center_x) / a_w,
            np.log(gt_h / a_h),
            np.log(gt_w / a_w),
        ]
        # Normalize
        rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV
        ix += 1

    return rpn_match, rpn_bbox

def pack_on(materials, len_min=1):
    # 最大のサイズに合わせてゼロで固める.
    ndim = materials[0].ndim
    lens = [m.shape[0] for m in materials]
    max_len = max(max(lens), len_min)
    pads = [((0, max_len - l),) + ((0, 0),) * (ndim - 1) for l in lens]
    materials = [np.pad(m, p, mode="constant", constant_values=0) for m, p in zip(materials, pads)]
    return np.stack(materials, axis=0)

def mold_mask(masks, scale, pad):
    n, h, w = masks.shape
    if n:
        #(n, h, w) -> (h, w, n)
        masks = np.transpose(masks, (1, 2, 0))
        masks = transform.resize(masks, (round(h * scale), round(w * scale)), preserve_range=True, anti_aliasing=False, mode="constant")
        masks = np.transpose(masks, (2, 0, 1))
    else:
        masks = np.reshape(masks, (0, round(h * scale), round(w * scale)))
    masks = np.pad(masks, ((0, 0),) + pad[:2], mode="constant", constant_values=0)
    return masks
                

def to_mini_mask(masks, boxes, crop_size, sess):
    mini_masks = []
    for i in range(len(masks)):
        mask = np.reshape(masks[i], (1,) + masks[i].shape + (1,))
        box = np.reshape(boxes[i], (1, -1))
        mini_mask = tf.image.crop_and_resize(mask, box, [0], crop_size)
        mini_mask = sess.run(mini_mask)
        mini_mask = np.reshape(mini_mask, mini_mask.shape[1:3])
        mini_masks.append(mini_mask)
    return mini_masks

def to_minimask(masks, boxes, crop_size):
    mini_masks = []
    for i in range(len(masks)):
        #mask = np.reshape(masks[i], (1,) + masks[i].shape + (1,))
        mask = masks[i]
        #box = np.reshape(boxes[i], (1, -1)).copy()
        box = boxes[i]
        h, w = mask.shape
        box = np.round(box * np.array([h, w, h, w])).astype(np.int)
        if box[2] <= box[0]: box[2] = box[0] + 1
        if box[3] <= box[1]: box[3] = box[1] + 1
        mini_mask = mask[box[0]:box[2], box[1]:box[3]]
        mini_mask = transform.resize(mini_mask, crop_size)
        #mini_mask = np.reshape(mini_mask, mini_mask.shape[1:3])
        mini_masks.append(mini_mask)
    if len(mini_masks):
        mini_masks = np.array(mini_masks)
    else:
        mini_masks = np.empty((0,) + crop_size)
    return mini_masks

def pack_gt_boxes(gt_boxes, mold_scales, windows, molded_shape):
    # rescale
    gt_boxes = [b * s for b, s in zip(gt_boxes, mold_scales)]
    # shift
    gt_boxes = [sift_boxes(b, w[:2]) for b, w in zip(gt_boxes, windows)]
    return gt_boxes

def mold_gt_boxes(gt_boxes, mold_scales, windows, molded_shape):
    # normalize
    gt_boxes = [norm_boxes(b, molded_shape[1:3]) for b in gt_boxes]
    return gt_boxes

def pack_gt_masks(gt_masks, scales, pads, gt_boxes, config, sess, ):
    # rescale and pad
    
    gt_masks = [mold_mask(m, s, p) for m, s, p in zip(gt_masks, scales, pads)]
    # crop and resize to mini_mask
    gt_masks = [to_mini_mask(m, b, config.MINI_MASK_SIZE, sess) for m, b in zip(gt_masks, gt_boxes)]
    # round to bool
    gt_masks = [np.round(m).astype(np.bool) for m in gt_masks]
    return gt_masks

def pack_gt_minimasks(gt_masks, scales, pads, gt_boxes, config, ):
    # rescale and pad
    
    gt_masks = [mold_mask(m, s, p) for m, s, p in zip(gt_masks, scales, pads)]
    # crop and resize to mini_mask
    gt_masks = [to_minimask(m, b, config.MINI_MASK_SIZE) for m, b in zip(gt_masks, gt_boxes)]
    # round to bool
    gt_masks = [np.round(m).astype(np.bool) for m in gt_masks]
    return gt_masks

def mold_gt_masks(gt_masks, scales, pads):
    # rescale and pad
    
    gt_masks = [mold_mask(m, s, p) for m, s, p in zip(gt_masks, scales, pads)]
    return gt_masks

def pack_gt_minimasks_already_molded(gt_masks, gt_boxes, config):
    # crop and resize to mini_mask
    gt_masks = [to_minimask(m, b, config.MINI_MASK_SIZE) for m, b in zip(gt_masks, gt_boxes)]
    # round to bool
    gt_masks = [np.round(m).astype(np.bool) for m in gt_masks]
    return gt_masks


import dataset
def make_gt_boxes_from_mask(gt_masks):
    gt_boxes = [[dataset.make_gt_box_from_mask(m) for m in gt_mask] for gt_mask in gt_masks]
    gt_boxes = [np.array(b) for b in gt_boxes]
    gt_boxes = [b if b.size else np.empty((0, 4)) for b in gt_boxes]
    return gt_boxes

def resize_mask(mask, scale, padding, crop=None):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the mask, are resized consistently.

    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (0, 0)]
    """
    # Suppress warning from scipy 0.13.0, the output shape of zoom() is
    # calculated with round() instead of int()
    mask = np.expand_dims(mask, axis=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
    if crop is not None:
        y, x, h, w = crop
        mask = mask[y:y + h, x:x + w]
    else:
        mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return np.squeeze(mask, axis=-1).astype(np.bool)

def minimize_mask(bbox, mask, mini_shape):
    """Resize masks to a smaller version to reduce memory load.
    Mini-masks can be resized back to image scale using expand_masks()

    See inspect_data.ipynb notebook for more details.
    """
    mini_mask = np.zeros((mask.shape[0],) + mini_shape, dtype=bool)
    for i in range(mask.shape[0]):
        # Pick slice and cast to bool in case load_mask() returned wrong dtype
        m = mask[i, :, :].astype(bool)
        y1, x1, y2, x2 = bbox[i][:4]
        m = m[y1:y2, x1:x2]
        if m.size == 0:
            raise Exception("Invalid bounding box with area of zero")
        # Resize with bilinear interpolation
        m = resize(m, mini_shape)
        mini_mask[i, :, :] = np.around(m).astype(np.bool)
    return mini_mask

def usable_image(detections, mrcnn_masks, window, original_shape, molded_shape):
    # detection_box(config.NUM_MAX_ROIS, 4)
    # cls_id(config.NUM_MAX_ROIS)
    # score(config.NUM_MAX_ROIS)
    # mrcnn_mask(config.NUM_MAX_ROIS, 14, 14, config.NUM_CLASS)
    detection_box = detections[:, :4]
    cls_id = detections[:, 4]
    score = detections[:, 5]
    i_zero = np.where(cls_id == 0)[0]
    # cls_idが背景を示すもののインデックス
    # cls_idが空ならば変更後の長さは変わらない.
    N = i_zero[0] if len(i_zero) else detection_box.shape[0]
    detection_box = detection_box[:N]
    cls_id = cls_id[:N].astype(np.int32)
    score = score[:N]
    # 背景でないものかつ, cls_idの示すもの(N, 7, 7)
    masks = mrcnn_masks[np.arange(N), :, :, cls_id]
    # moldedに合わせる
    molded_window = norm_boxes(window, molded_shape[1:3])
    
    wy1, wx1, wy2, wx2 = molded_window
    wh = wy2 - wy1
    ww = wx2 - wx1
    # 端に寄せる
    box = detection_box - (wy1, wx1, wy1, wx1)
    # windowサイズで(1, 1)になるようにする.
    box = np.divide(box, (wh, ww, wh, ww))
    # 元の画像サイズに合わせる.
    box = demold_box(box, original_shape)
    exclude_id = np.where(((box[:, 2] - box[:, 0]) <= 0) + ((box[:, 3] - box[:, 1]) <= 0))[0]
    box = np.delete(box, exclude_id, axis=0)
    cls_id = np.delete(cls_id, exclude_id, axis=0)
    score = np.delete(score, exclude_id, axis=0)
    masks = np.delete(masks, exclude_id, axis=0)
    full_masks = []
    for i, mask in enumerate(masks):
        full_mask = unmold_mask(mask, box[i], original_shape)
        full_masks.append(full_mask)
    full_masks = np.stack(full_masks, axis=0) if full_masks else np.empty((0,)+original_shape[:2])
    return box, cls_id, score, full_masks

if __name__ == "__main__":
    pass
