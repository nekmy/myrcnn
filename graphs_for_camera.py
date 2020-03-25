import tensorflow as tf
import numpy as np

import utils

def log2_graph(x):
    return tf.log(x) / tf.log(2.0)

def conv_block(inputs, num_filters, kernel_size, stage, block, strides):
    base_name = "res" + str(stage) + block + "_branch"
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    nf1, nf2, nf3 = num_filters

    x = tf.layers.conv2d(inputs, filters=nf1, kernel_size=1, strides=strides,
        padding="VALID", name=base_name+"2a")
    x = tf.layers.batch_normalization(x, name=bn_name_base + '2a')
    x = tf.nn.relu(x)

    x = tf.layers.conv2d(x, filters=nf2, kernel_size=kernel_size, strides=(1, 1),
        padding="SAME", name=base_name+"2b")
    x = tf.layers.batch_normalization(x, name=bn_name_base + '2b')
    x = tf.nn.relu(x)

    x = tf.layers.conv2d(x, filters=nf3, kernel_size=1, strides=(1, 1),
        padding="SAME", name=base_name+"2c")
    x = tf.layers.batch_normalization(x, name=bn_name_base + '2c')
    shortcut = tf.layers.conv2d(inputs, filters=nf3, kernel_size=1, strides=strides,
        padding="VALID", name=base_name+"1")
    shortcut = tf.layers.batch_normalization(shortcut, name=bn_name_base + '1')
    x = tf.add(x, shortcut)
    x = tf.nn.relu(x)
    return x

def identity_block(inputs, num_filters, kernel_size, stage, block):
    base_name = "res" + str(stage) + block + "_branch"
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    nf1, nf2, nf3 = num_filters

    x = tf.layers.conv2d(inputs, filters=nf1, kernel_size=1, strides=(1, 1),
        padding="SAME", name=base_name+"2a")
    x = tf.layers.batch_normalization(x, name=bn_name_base + '2a')
    x = tf.nn.relu(x)

    x = tf.layers.conv2d(x, filters=nf2, kernel_size=kernel_size, strides=(1, 1),
        padding="SAME", name=base_name+"2b")
    x = tf.layers.batch_normalization(x, name=bn_name_base + '2b')
    x = tf.nn.relu(x)
    
    x = tf.layers.conv2d(x, filters=nf3, kernel_size=1, strides=(1, 1),
        padding="SAME", name=base_name+"2c")
    x = tf.layers.batch_normalization(x, name=bn_name_base + '2c')
    
    x = tf.add(x, inputs)
    x = tf.nn.relu(x)
    return x

def resnet(images):
    # imagesをCNNにかけることで特徴量を得る.
    x = tf.pad(images, ((0, 0), (3, 3), (3, 3), (0, 0)), mode="CONSTANT", constant_values=0.0)
    conv1 = x = tf.layers.conv2d(x, filters=64, kernel_size=7, strides=(2, 2), padding="VALID", name="conv1")
    bn = x = tf.layers.batch_normalization(x, name='bn_conv1')
    x = tf.nn.relu(x)
    C1 = x = tf.nn.max_pool(x, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding="SAME")
    x = conv_block(x, num_filters=[64, 64, 256], kernel_size=3, stage=2, block="a", strides=(1, 1))
    x = identity_block(x, num_filters=[64, 64, 256], kernel_size=3, stage=2, block="b")
    C2 = x = identity_block(x, num_filters=[64, 64, 256], kernel_size=3, stage=2, block="c")
    x = conv_block(x, num_filters=[128, 128, 512], kernel_size=3, stage=3, block="a", strides=(2, 2))
    x = identity_block(x, num_filters=[128, 128, 512], kernel_size=3, stage=3, block="b")
    x = identity_block(x, num_filters=[128, 128, 512], kernel_size=3, stage=3, block="c")
    C3 = x = identity_block(x, num_filters=[128, 128, 512], kernel_size=3, stage=3, block="d")
    x = conv_block(x, num_filters=[256, 256, 1024], kernel_size=3, stage=4, block="a", strides=(2, 2))
    for idx in range(22):
        x = identity_block(x, num_filters=[256, 256, 1024], kernel_size=3, stage=4, block=chr(98+idx))
    C4 = x
    x = conv_block(x, num_filters=[512, 512, 2048], kernel_size=3, stage=5, block="a", strides=(2, 2))
    x = identity_block(x, num_filters=[512, 512, 2048], kernel_size=3, stage=5, block="b")
    C5 = x = identity_block(x, num_filters=[512, 512, 2048], kernel_size=3, stage=5, block="c")
    return C1, C2, C3, C4, C5

def refine_detection_graph(rois, cls_prob, deltas, window, config):
    # cls_probが大きいもの（deltaごとに１つだけ）
    cls_ids = tf.cast(tf.argmax(cls_prob, axis=1), tf.int32)
    # ((0, i0), (1, i1), (2, i2), ...)
    indices = tf.stack([tf.range(tf.shape(cls_ids)[0]), cls_ids], axis=1)
    cls_scores = tf.gather_nd(cls_prob, indices)
    # 最もcls_probの大きいclsのdeltasを用いる.
    cls_deltas = tf.gather_nd(deltas, indices)
    # roiは提案されたもの. boxはdeltaを適用したもの.
    boxes = utils.apply_deltas(rois, cls_deltas*config.BBOX_STD_DEV)
    boxes = utils.clip_box(boxes, window)
    # back_groundではないもの
    keep = tf.where(cls_ids > 0)[:, 0]
    if config.MIN_SCORE:
        keep_2 = tf.where(cls_scores > config.MIN_SCORE)[:, 0]
        keep = tf.sets.set_intersection(tf.expand_dims(keep, 0), tf.expand_dims(keep_2, 0))
        keep = tf.sparse_tensor_to_dense(keep)[0]
    pre_nms_cls_ids = tf.gather(cls_ids, keep)
    pre_nms_scores = tf.gather(cls_scores, keep)
    pre_nms_boxes = tf.gather(boxes, keep)
    unique_class_ids = tf.unique(pre_nms_cls_ids)[0]
    
    def nms_cls(cls_id):
        #ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))
        # pre_nmsのなかで何番目か
        ixs = tf.where(tf.equal(pre_nms_cls_ids, cls_id))[:, 0]
        # ixsの中で何番目か
        cls_keep = tf.image.non_max_suppression(
            tf.gather(pre_nms_boxes, ixs),
            tf.gather(pre_nms_scores, ixs),
            max_output_size=config.MAX_NUM_ROIS,
            iou_threshold=config.DETECTION_NMS_THRESHOLD)
        cls_keep = tf.gather(ixs, cls_keep)
        cls_keep = tf.gather(keep, cls_keep)
        gap = config.MAX_NUM_ROIS - tf.shape(cls_keep)[0]
        cls_keep = tf.pad(cls_keep, [(0, gap)], mode="CONSTANT", constant_values=-1)
        cls_keep.set_shape([config.MAX_NUM_ROIS])
        return cls_keep
    nms_keep = tf.map_fn(nms_cls, unique_class_ids, dtype=tf.int64)
    nms_keep = tf.reshape(nms_keep, (-1,))
    nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])
    keep = tf.sets.set_intersection(tf.expand_dims(nms_keep, 0), tf.expand_dims(keep, 0))
    keep = tf.sparse_tensor_to_dense(keep)[0]
    roi_cound = config.MAX_NUM_ROIS
    class_score_keep = tf.gather(cls_scores, keep)
    num_keep = tf.minimum(roi_cound, tf.shape(class_score_keep)[0])
    top_ids = tf.nn.top_k(class_score_keep, num_keep, sorted=True)[1]
    keep = tf.gather(keep, top_ids)
    detected_boxes = tf.gather(boxes, keep)
    detected_cls_ids = tf.gather(cls_ids, keep)
    detected_cls_scores = tf.gather(cls_scores, keep)
    detections = tf.concat([detected_boxes,
        tf.to_float(detected_cls_ids)[..., tf.newaxis],
        detected_cls_scores[..., tf.newaxis]], axis=1)

    gap = roi_cound - tf.shape(detections)[0]
    detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")
    return detections

def detection_graph(config, mrcnn_rois, mrcnn_class_prob, mrcnn_deltas, image_meta):
    metas = utils.metas_converter(image_meta)
    molded_shape = metas["molded_shape"]
    windows = utils.norm_boxes_tf(metas["window"], molded_shape[:2])
    detections_batch = utils.batch_slice([mrcnn_rois, mrcnn_class_prob, mrcnn_deltas, windows],
        config.BATCH_SIZE_INFERENCE, lambda x, y, z, w: refine_detection_graph(x, y, z, w, config))
    detections_batch = tf.reshape(detections_batch, (config.BATCH_SIZE_INFERENCE, config.MAX_NUM_ROIS, 6))
    return detections_batch

def pyramid_roi_align(pool_shape, rois, feature_map, image_metas, **kwargs):
    # rois(num_batch, num_rois, 4)
    # y1(num_batch, num_rois, 1)
    y1, x1, y2, x2 = tf.split(rois, 4, axis=2)
    h = y2 - y1
    w = x2 - x1
    image_shape = utils.metas_converter(image_metas)["molded_shape"][0]
    area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
    # どの特徴量マップを使うか
    rois_level = log2_graph(tf.sqrt(h * w) / (224.0 / tf.sqrt(area)))
    rois_level = tf.minimum(5, tf.maximum(2, 4 + tf.cast(tf.round(rois_level), tf.int32)))
    rois_level = tf.squeeze(rois_level, axis=2)

    pooled = []
    rois_to_level = []
    for i, level in enumerate(range(2, 6)):
        ix = tf.cast(tf.where(tf.equal(rois_level, level)), tf.int32)
        level_rois = tf.gather_nd(rois, ix)
        batch_indices = ix[:, 0]
        rois_to_level.append(ix)
        level_rois = tf.stop_gradient(level_rois)
        batch_indices = tf.stop_gradient(batch_indices)
        pooled.append(tf.image.crop_and_resize(feature_map[i], level_rois, batch_indices, pool_shape, method="bilinear"))
    
    origin = pooled = tf.concat(pooled, axis=0)
    rois_to_level = tf.concat(rois_to_level, axis=0)
    level_indices = tf.reshape(tf.range(0, tf.shape(rois_to_level)[0]), (-1, 1))
    rois_to_level = tf.concat([rois_to_level, level_indices], axis=1)
    rois_indices = rois_to_level[:, 0] * 10000 + rois_to_level[:, 1]
    sort_indices = tf.nn.top_k(rois_indices, tf.shape(rois_indices)[0], sorted=True).indices[::-1]

    # level順に並べられたpooledをbatch, rois順に並べなおす.
    ix = tf.gather(rois_to_level[:, 2], sort_indices)
    # => pooled(num_batch*num_rois, 7, 7, C)
    pooled = tf.gather(pooled, ix)
    # => shape = (num_batch, num_rois, 7, 7, C)
    shape = tf.concat([tf.shape(rois)[:2], tf.shape(pooled)[1:]], axis=0)
    pooled = tf.reshape(pooled, shape)
    return pooled

def rpn_graph(feature_map, anchors_per_pixel, anchor_stride):
    # k:num_anchors_per_pixel
    # 512, 3, 3
    """
    x = tf.layers.conv2d(feature_map, filters=512, kernel_size=1, strides=anchor_stride,
        padding="SAME", name="rpn_conv_shared")
    shared = tf.nn.relu(x)
    # k*2, 1, 1
    # ピクセルごとに全結合
    x = tf.layers.conv2d(shared, filters=anchors_per_pixel*2, kernel_size=1,
        strides=(1, 1), name="rpn_class_raw")
    rpn_cls_logit = tf.keras.layers.Lambda(lambda t: tf.reshape(t, (tf.shape(t)[0], -1, 2)))(x)
    rpn_cls_prob = tf.nn.softmax(rpn_cls_logit)
    rpn_cls_prob = tf.keras.layers.Lambda(lambda t: t)(rpn_cls_prob)
    # k*4, 1, 1
    x = tf.layers.conv2d(shared, filters=anchors_per_pixel*4, kernel_size=1,
        strides=(1, 1), name="rpn_bbox_pred")
    # N, H, W, k, 4
    rpn_deltas = tf.keras.layers.Lambda(lambda t: tf.reshape(t, (tf.shape(t)[0], -1, 4)))(x)
    rpn_deltas = tf.keras.layers.Lambda(lambda t: t)(rpn_deltas)
    return [rpn_cls_logit, rpn_cls_prob, rpn_deltas]
    """
    shared = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu',
                       strides=anchor_stride,
                       name='rpn_conv_shared')(feature_map)

    # Anchor Score. [batch, height, width, anchors per location * 2].
    x = tf.keras.layers.Conv2D(2 * anchors_per_pixel, (1, 1), padding='valid',
                  activation='linear', name='rpn_class_raw')(shared)

    # Reshape to [batch, anchors, 2]
    rpn_cls_logit = tf.keras.layers.Lambda(
        lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)

    # Softmax on last dimension of BG/FG.
    rpn_cls_prob = tf.keras.layers.Activation(
        "softmax", name="rpn_class_xxx")(rpn_cls_logit)

    # Bounding box refinement. [batch, H, W, anchors per location * depth]
    # where depth is [x, y, log(w), log(h)]
    x = tf.keras.layers.Conv2D(anchors_per_pixel * 4, (1, 1), padding="valid",
                  activation='linear', name='rpn_bbox_pred')(shared)

    # Reshape to [batch, anchors, 4]
    rpn_deltas = tf.keras.layers.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)
    return [rpn_cls_logit, rpn_cls_prob, rpn_deltas]

def build_rpn_model(anchor_stride, anchors_per_location, depth):
    input_feature_map = tf.keras.layers.Input(shape=[None, None, depth],
                                 name="input_rpn_feature_map")
    outputs = rpn_graph(input_feature_map, anchors_per_location, anchor_stride)
    return tf.keras.Model([input_feature_map], outputs, name="rpn_model")

def build_rpn_graph(amchor_strude, anchors_per_pixel, depth, config, batch_size):
    feature_map = tf.placeholder(tf.float32, (batch_size, None, None, depth), name="input_rpn_feature_map")
    x = tf.layers.conv2d(feature_map, filters=512, kernel_size=3, strides=(1, 1),
        padding="SAME", name="rpn_conv_shared")
    shared = tf.nn.relu(x)
    x = tf.layers.conv2d(shared, filters=anchors_per_pixel*2, kernel_size=1,
        strides=(1, 1), name="rpn_class_raw")
    # rpn_cls_logit(num_batch, H*W*A/pix, 2)
    # delta, logitは各anchorと対になるように生成される.
    # したがってanchorの数はH*Wと同じ
    # 厳密にはanchorの数に合わせてHWを定める(特徴量マップを定める).
    rpn_cls_logit = tf.reshape(x, (x.shape[0], -1, 2))
    rpn_cls_prob = tf.nn.softmax(rpn_cls_logit)
    # k*4, 1, 1
    x = tf.layers.conv2d(shared, filters=anchors_per_pixel*4, kernel_size=1,
        strides=(1, 1), name="rpn_bbox_pred")
    # N, H*W*k, 4
    rpn_deltas = tf.reshape(x, (x.shape[0], -1, 4))
    outputs = [rpn_cls_logit, rpn_cls_prob, rpn_deltas]
    return feature_map, outputs

def region_proposal_network(self, feature_map, anchors_per_pixel, stage):
    # k:num_anchors_per_pixel
    # 512, 3, 3
    N = feature_map.shape[0]
    x = tf.layers.conv2d(feature_map, filters=512, kernel_size=1, strides=(3, 3),
        padding="SAME", name="rpn_conv_shared")
    shared = tf.nn.relu(x)
    # k*2, 1, 1
    # ピクセルごとに全結合
    x = tf.layers.conv2d(shared, filters=anchors_per_pixel*2, kernel_size=1,
        strides=(1, 1), name="rpn_class_raw")
    rpn_cls_logit = tf.reshape(x, (N, -1, 2))
    rpn_cls_prob = tf.nn.softmax(rpn_cls_logit)
    # k*4, 1, 1
    x = tf.layers.conv2d(shared, filters=anchors_per_pixel*4, kernel_size=1,
        strides=(1, 1), name="rpn_bbox_pred")
    # N, H, W, k, 4
    rpn_deltas = tf.reshape(x, (N, -1, 4))
    return [rpn_cls_logit, rpn_cls_prob, rpn_deltas]

def proposal_graph(proposal_count, nms_threshold, config, batch_size, cls_prob, deltas, anchors):
    # (num_batch, N, 4)
    deltas = deltas * tf.reshape(tf.constant(config.RPN_BBOX_STD_DEV, tf.float32), (1, 1, 4))
    scores = cls_prob[:, :, 1]
    indices = tf.nn.top_k(scores, k=tf.minimum(config.PRE_NMS_PROPOSALS_INFERENCE, tf.shape(anchors)[0]), sorted=True).indices
    # 以下indicesのものだけ.
    scores = utils.batch_slice((scores, indices), batch_size, tf.gather)
    deltas = utils.batch_slice((deltas, indices), batch_size, tf.gather)
    anchors = utils.batch_slice((indices,), batch_size, lambda x:tf.gather(anchors, x))
    #pre_nms_box = utils.apply_deltas(anchors, deltas)
    pre_nms_boxes = utils.batch_slice((anchors, deltas), batch_size, utils.apply_deltas)
    windows = np.array([0, 0, 1, 1], dtype=np.float32)
    pre_nms_boxes = utils.batch_slice((pre_nms_boxes,), batch_size, lambda x: utils.clip_box(x, windows))
    tf.function()
    #nms_indices = tf.image.non_max_suppression(pre_nms_box, scores, max_k, iou_threshold=0.5)
    def nms(pre_nms_box, scores):
        #indices = utils.non_maximum_suppression(pre_nms_box, scores, proposal_count, iou_min=0.5, sorted=True)
        #indices = tf.image.non_max_suppression(pre_nms_box, scores, proposal_count, nms_threshold, name="rpn_non_max_suppression")
        indices = utils.non_max_suppression(pre_nms_box, scores, proposal_count, nms_threshold, name="rpn_non_max_suppression")
        proposals = tf.gather(pre_nms_box, indices)
        num_pad = tf.maximum(proposal_count - tf.shape(proposals)[0], 0)
        proposals = tf.pad(proposals, [(0, num_pad), (0, 0)])
        proposals = tf.gather(proposals, tf.range(proposal_count))
        return proposals
    proposals = utils.batch_slice((pre_nms_boxes, scores), batch_size, nms)
    return proposals, pre_nms_boxes

def fpn_classifer_graph(feature_map, rois, image_metas, poolsize, num_class, fc_layer_size=1024, training=False):
    #x = PyramidROIAlign((poolsize, poolsize))([rois, feature_map, image_metas])
    x = pyramid_roi_align((poolsize, poolsize), rois, feature_map, image_metas)
    # (N, num_rois, 7, 7, C)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(fc_layer_size, (poolsize, poolsize), padding="valid"),
                        name="mrcnn_class_conv1")(x)
    x = tf.keras.layers.TimeDistributed(tf.layers.BatchNormalization(name='mrcnn_class_bn1'))(x, training=training)
    x = tf.nn.relu(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(fc_layer_size, (1, 1)),
                        name="mrcnn_class_conv2")(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization(), name='mrcnn_class_bn2')(x, training=training)
    x = tf.nn.relu(x)

    shared = tf.squeeze(x, [2, 3])

    # Classifier head
    mrcnn_class_logits = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_class),
                                            name='mrcnn_class_logits')(shared)
    mrcnn_probs = tf.keras.layers.TimeDistributed(tf.keras.layers.Activation("softmax"),
                                    name="mrcnn_class")(mrcnn_class_logits)

    # BBox head
    # [batch, num_rois, NUM_CLASSES * (dy, dx, log(dh), log(dw))]
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_class * 4, activation='linear'),
                        name='mrcnn_bbox_fc')(shared)
    # Reshape to [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
    mrcnn_deltas = tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], num_class, 4))
    return mrcnn_class_logits, mrcnn_probs, mrcnn_deltas

def fpn_mask_graph(rois, feature_map, image_metas, pool_size, num_classes):
    # (num_batch, num_rois, 7, 7, C)
    #x = PyramidROIAlign((pool_size, pool_size))([rois, feature_map, image_metas])
    x = pyramid_roi_align((pool_size, pool_size), rois, feature_map, image_metas)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv1")(x)
    x = tf.keras.layers.TimeDistributed(tf.layers.BatchNormalization(name="mrcnn_mask_bn1"))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Activation("relu"), name="mrcnn_mask_relu1")(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv2")(x)
    x = tf.keras.layers.TimeDistributed(tf.layers.BatchNormalization(name="mrcnn_mask_bn2"))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Activation("relu"), name="mrcnn_mask_relu2")(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv3")(x)
    x = tf.keras.layers.TimeDistributed(tf.layers.BatchNormalization(name="mrcnn_mask_bn3"))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Activation("relu"), name="mrcnn_mask_relu3")(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv4")(x)
    x = tf.keras.layers.TimeDistributed(tf.layers.BatchNormalization(name="mrcnn_mask_bn4"))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Activation("relu"), name="mrcnn_mask_relu4")(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=2, activation="relu"), name="mrcnn_mask_deconv")(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid"), name="mrcnn_mask")(x)
    return x

def rpn_cls_loss_func(rpn_logits, gt_rpn_matchs):
    # 学習対象のanchorのインデックス(-1or1)
    rpn_logits_shape = tf.shape(rpn_logits)
    # (batch, anchors, (negative, positive)) => (batch*anchors, (negative, positive))
    rpn_logits = tf.reshape(rpn_logits, (-1, rpn_logits_shape[2]))
    gt_rpn_matchs = tf.reshape(gt_rpn_matchs, (-1,))
    target_idx = tf.where(tf.not_equal(gt_rpn_matchs, 0))[:, 0]
    # 1は1それ以外は0
    target_gt_rpn_cls_id = tf.cast(gt_rpn_matchs>0, tf.int64)
    target_gt_rpn_cls_id = tf.gather(target_gt_rpn_cls_id, target_idx)
    # target_gt_rpn_cls = tf.one_hot(target_gt_rpn_cls_id, 2)
    target_rpn_logits = tf.gather(rpn_logits, target_idx)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=target_gt_rpn_cls_id, logits=target_rpn_logits)
    loss = tf.cond(tf.not_equal(tf.size(loss), 0),
        true_fn=lambda: tf.reduce_mean(loss),
        false_fn=lambda: tf.constant(0.0))
    #loss = tf.reduce_mean(loss)

    return loss

def rpn_delta_loss_func(rpn_deltas, gt_rpn_deltas, gt_rpn_matchs, config):
    # 一致しているid(N, A)
    positive_ids = tf.where(tf.equal(gt_rpn_matchs, 1))
    # target_rpn_deltas(num_target, 4)
    target_rpn_deltas = tf.gather_nd(rpn_deltas, positive_ids)
    # gt_rpn_matchs(N, num_anchors)
    num_row = tf.math.reduce_sum(tf.cast(tf.equal(gt_rpn_matchs, 1), tf.int32), axis=1)
    target_gt_rpn_deltas = utils.pack_gt_deltas(gt_rpn_deltas, num_row, config.BATCH_SIZE_TRAIN)
    loss = utils.L1_smooth(target_rpn_deltas, target_gt_rpn_deltas)
    #loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    loss = tf.cond(tf.not_equal(tf.size(loss), 0),
        true_fn=lambda: tf.reduce_mean(loss),
        false_fn=lambda: tf.constant(0.0))
    #loss = tf.reduce_mean(loss)

    return loss, target_gt_rpn_deltas, target_rpn_deltas

def detection_target(self, proposals, gt_boxes, gt_cls_ids, gt_masks):
    overlaps = utils.intersection_over_union(proposals, gt_boxes)
    positive_count = self.NUM_TARGET * self.RATIO
    positive_indices = tf.where(overlaps>=0.5)[0]
    negative_indices = tf.where(overlaps<0.5)[0]
    positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
    negative_count = self.NUM_TARGET - positive_count
    negative_indices = tf.random_shuffle(negative_indices)[:negative_count]
    positive_proposals = tf.gather(proposals, positive_indices)
    positive_gt_boxes = tf.gather(gt_boxes, positive_indices)
    positive_gt_deltas = utils.make_deltas(positive_proposals, positive_gt_boxes)
    return

def trim_non_zero(boxes):
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros)
    return boxes, non_zeros

def detection_target_graph(proposals, gt_boxes, gt_cls_ids, gt_masks, config, positive_indices_master=None, negative_indices_master=None):
    # batchごと
    proposals, _ = trim_non_zero(proposals)
    gt_boxes, non_zeros = trim_non_zero(gt_boxes)
    indices = tf.where(non_zeros)[:, 0]
    gt_cls_ids = tf.gather(gt_cls_ids, indices)
    gt_masks = tf.gather(gt_masks, indices)
    # ポジティブではなく, crowdでもない要素をネガティブとする(gt_cls_id=0)
    # cls学習のときにのみproposalとセットで使われる.
    # crowdを排除するために使う.
    crowd_ixs = tf.where(gt_cls_ids < 0)[:, 0]
    crowd_gt_boxes = tf.gather(gt_boxes, crowd_ixs)
    non_crowd_ixs = tf.where(gt_cls_ids > 0)[:, 0]
    base_gt_cls_ids = gt_cls_ids = tf.gather(gt_cls_ids, non_crowd_ixs)
    base_gt_boxes = gt_boxes = tf.gather(gt_boxes, non_crowd_ixs)
    base_gt_masks = gt_masks = tf.gather(gt_masks, non_crowd_ixs)
    crowd_overlaps = utils.intersection_over_union(proposals, crowd_gt_boxes)
    crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
    iou_max = tf.reduce_max(utils.intersection_over_union(proposals, gt_boxes), axis=1)
    # >0.7, <0.3等ではどうか
    positive_indices = tf.where(iou_max>=0.5)[:, 0]
    negative_indices = tf.where(tf.logical_and(iou_max<0.5, crowd_iou_max<0.001))[:, 0]

    # 比率を維持できるように個数を決める.
    #positive_count = tf.cast(tf.reduce_max([int(config.TRAIN_ROIS_NUM * config.POSITIVE_RATIO), tf.shape(positive_indices)[0]]), tf.int64)
    positive_count = tf.cast(int(config.TRAIN_ROIS_NUM * config.POSITIVE_RATIO), tf.int64)
    # negative_count = int(float(positive_count) / config.POSITIVE_RATIO) - positive_count
    negative_count = tf.cast(tf.cast(positive_count, tf.float32) / config.POSITIVE_RATIO, tf.int64) - positive_count
    # ploposalsを選ぶ.
    positive_indices = tf.random_shuffle(positive_indices)[:positive_count]

    #positive_indices = positive_indices_master

    positive_rois = tf.gather(proposals, positive_indices)
    negative_indices = tf.random_shuffle(negative_indices)[:negative_count]

    #negative_indices = negative_indices_master

    negative_rois = tf.gather(proposals, negative_indices)
    rois = tf.concat([positive_rois, negative_rois], axis=0)

    # 最も被っているgtを選ぶ(これに近づける)
    positive_iou = utils.intersection_over_union(positive_rois, gt_boxes)
    arg_positive_iou_max = tf.cond(
        tf.not_equal(tf.shape(positive_iou)[0], 0),
        true_fn=lambda: tf.argmax(positive_iou, axis=1),
        false_fn=lambda: tf.constant([], dtype=tf.int64, shape=(0,)))
    gt_cls_ids = tf.gather(gt_cls_ids, arg_positive_iou_max)
    gt_boxes = tf.gather(gt_boxes, arg_positive_iou_max)
    gt_deltas = utils.make_deltas(positive_rois, gt_boxes) / config.BBOX_STD_DEV
    #transposed_masks = tf.expand_dims(tf.transpose(gt_masks, [2, 0, 1]), axis=-1)
    transposed_masks = tf.expand_dims(gt_masks, axis=-1)
    # roi毎にgt_maskを選ぶ.
    gt_masks = tf.gather(transposed_masks, arg_positive_iou_max)
    # Cが必要
    # gt_masks = tf.expand_dims(gt_masks, axis=-1)
    box_ind = tf.range(0, tf.shape(gt_masks)[0])

    # mask 変換
    # gt_maskはmask部だけで構成されている.
    gt_dh = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_dw = gt_boxes[:, 3] - gt_boxes[:, 1]
    y1 = (positive_rois[:, 0] - gt_boxes[:, 0]) / gt_dh
    x1 = (positive_rois[:, 1] - gt_boxes[:, 1]) / gt_dw
    y2 = (positive_rois[:, 2] - gt_boxes[:, 0]) / gt_dh
    x2 = (positive_rois[:, 3] - gt_boxes[:, 1]) / gt_dw
    boxes = tf.stack([y1, x1, y2, x2], axis=1)
    # gt_masksのうちboxesの箇所を切り取って（枠外は0になる）28*28に変形する.
    gt_masks = tf.image.crop_and_resize(gt_masks, boxes, box_ind=box_ind,
                                     crop_size=config.MASK_SHAPE)
    gt_masks = tf.squeeze(gt_masks, axis=3)
    gt_masks = tf.round(gt_masks)
    N = tf.shape(negative_rois)[0]
    P = tf.reduce_max([[config.TRAIN_ROIS_NUM], [tf.shape(rois)[0]]]) - tf.shape(rois)[0]
    proposals = tf.pad(rois, ((0, P), (0, 0)))
    gt_cls_ids = tf.pad(gt_cls_ids, ((0, N + P),))
    gt_deltas = tf.pad(gt_deltas, ((0, N + P), (0, 0)))
    gt_masks = tf.pad(gt_masks, ((0, N + P), (0, 0), (0, 0)))

    return proposals, gt_cls_ids, gt_deltas, gt_masks

def mrcnn_cls_loss_func(cls_logits, target_cls_ids, active_ids):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tf.cast(target_cls_ids, tf.int64), logits=cls_logits
    )
    pred_ids = tf.arg_max(cls_logits, dimension=2)
    # active_ids: activeなidは1, それ以外は0
    active_pred = tf.gather(active_ids[0], pred_ids)
    loss *= active_pred
    # activeなものの平均
    loss = tf.cond(tf.not_equal(tf.size(active_pred), 0),
        true_fn=lambda: tf.reduce_sum(loss) / tf.reduce_sum(active_pred),
        false_fn=lambda: tf.constant(0.0))
    #loss = tf.reduce_sum(loss) / tf.reduce_sum(active_pred)
    return loss

def mrcnn_delta_loss_func(target_cls_ids, target_deltas, pred_deltas):
    # target_cls_ids[batch, rois] 0埋めを含む
    # target_deltas[batch, rois, 4]
    # pred_deltas[batch, rois, cls, 4]
    target_cls_ids = tf.reshape(target_cls_ids, (-1,))
    target_deltas = tf.reshape(target_deltas, (-1, 4))
    pred_deltas = tf.reshape(pred_deltas, (-1, tf.shape(pred_deltas)[2], 4))
    target_ix = tf.where(target_cls_ids>0)[:, 0]
    pred_cls_ids = tf.gather(target_cls_ids, target_ix)
    indices = tf.stack([target_ix, tf.cast(pred_cls_ids, tf.int64)], axis=1)
    target_deltas = tf.gather(target_deltas, target_ix)
    pred_deltas = tf.gather_nd(pred_deltas, indices)
    loss = utils.L1_smooth(target_deltas, pred_deltas)
    loss = tf.cond(tf.not_equal(tf.size(loss), 0),
        true_fn=lambda: tf.reduce_mean(loss),
        false_fn=lambda: tf.constant(0.0))
    #loss = tf.reduce_mean(loss)
    return loss

def mrcnn_mask_loss_func(pred_cls_ids, pred_mask, target_mask):
    # pred_mask(batch, rois, H, W, cls)
    pred_cls_ids = tf.reshape(pred_cls_ids, (-1,))
    pred_mask_shape = tf.shape(pred_mask)
    pred_mask = tf.reshape(pred_mask, (-1, pred_mask_shape[2], pred_mask_shape[3], pred_mask_shape[4]))
    target_mask_shape = tf.shape(target_mask)
    target_mask = tf.reshape(target_mask, (-1, target_mask_shape[2], target_mask_shape[3]))
    positive_idx = tf.where(pred_cls_ids>0)[:, 0]
    pred_cls_ids = tf.gather(pred_cls_ids, positive_idx)
    indices = tf.stack([positive_idx, pred_cls_ids], axis=1)
    # (batch*rois, H, W, cls) -> (batch*rois, cls, H, W)
    pred_mask = tf.transpose(pred_mask, (0, 3, 1, 2))
    pred_mask = tf.gather_nd(pred_mask, indices)
    target_mask = tf.gather(target_mask, positive_idx)

    loss = tf.keras.backend.binary_crossentropy(target_mask, pred_mask)
    loss = tf.cond(tf.not_equal(tf.size(loss), 0),
        true_fn=lambda: tf.reduce_mean(loss),
        false_fn=lambda: tf.constant(0.0))
    #loss = tf.reduce_mean(loss)
    return loss
