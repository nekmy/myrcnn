from functools import reduce
import pickle

import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt

import utils
import visualize
import cal_mAP

scope_dict = {
    "resnet50": ["conv1", "bn_conv1", "res2", "bn2", "res3", "bn3", "res4", "bn4", "res5", "bn5"],
    "restorer": ["conv1", "bn_conv1", "res2", "bn2"],
    "res3": ["res3", "bn3", "res4", "bn4", "res5", "bn5", "fpn", "rpn", "mrcnn"],
    "res4": ["res4", "bn4", "res5", "bn5", "fpn", "rpn", "mrcnn"],
    "res5": ["res5", "bn5", "fpn", "rpn", "mrcnn"],
    "head": ["fpn", "rpn", "mrcnn"],
    "all": [None]
}

def make_training_op(loss, keys, lr, config):
    if not type(keys) in [tuple, list]:
        keys = [keys]
    scopes = [scope_dict[key] for key in keys]
    scopes = reduce(lambda x, y: x + y, scopes)
    trainable_var_list = [tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope) for scope in scopes]
    trainable_var_list = reduce(lambda x, y: x + y, trainable_var_list)
    trainable_var_list = list(set(trainable_var_list))
    #optimizer = tf.train.AdamOptimizer(lr)
    #optimizer = tf.train.GradientDescentOptimizer(lr)
    reg_losses = [
            tf.keras.regularizers.l2(config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
            for w in trainable_var_list
            if 'gamma' not in w.name and 'beta' not in w.name]
    reg_losses = tf.reduce_sum(reg_losses)
    loss = loss + reg_losses
    optimizer = tf.train.MomentumOptimizer(lr, config.LEARNING_MOMENTUM)
    # lossの関係するvariableがすべて返される.
    # trainableを限定するにはここで選別する必要がある.
    gradients, valiables = zip(*optimizer.compute_gradients(loss, var_list=trainable_var_list))
    # gradients, valiables = list(zip(*[(g, v) for g, v in zip(gradients, valiables) if v.name in trainable_var_list]))
    # gradients, valiables = list(zip(*[(g, v) for g, v in zip(gradients, valiables) if v in trainable_var_list]))
    gradients, _ = tf.clip_by_global_norm(gradients, config.CLIP_NORM)
    # 各variableのgradientにgradientsを適用する.
    training_op = optimizer.apply_gradients(zip(gradients, valiables))
    # training_op = optimizer.minimize(loss, var_list=trainable_var_list)
    return training_op

def make_saver(keys):
    if not type(keys) in [tuple, list]:
        keys = [keys]
    scopes = [scope_dict[key] for key in keys]
    scopes = reduce(lambda x, y: x + y, scopes)
    var_list = [tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope) for scope in scopes]
    var_list = reduce(lambda x, y: x + y, var_list)
    var_list = list(set(var_list))
    return tf.train.Saver(var_list)

def train_batch(dataset, inputs, config, num_epoch, sess):
    input_images, input_metas, input_anchors, input_rpn_gt_deltas, input_rpn_gt_matchs, input_gt_cls_ids, input_gt_box, input_gt_mask = inputs
    for i in range(num_epoch):
        scales = config.SCALES
        ratios = config.RATIOS
        images, gt_boxes, gt_masks, gt_class_ids = dataset.make_batch(config.BATCH_SIZE_TRAIN, img_ids=[139, 139])#139, 285, 632
        # batch毎にndarrayにする.
        molded_images, image_metas, windows, pads, mold_scales = utils.mold_images(images, max_dim=1024, min_dim=800, config=config)
        # mini_maskでないならばmaskもmoldedされる(paddingを出して)
        assert not molded_images.dtype == np.dtype("O"), "image shape is not same"
        molded_shape = molded_images.shape
        # gt_boxをresize(scale)->シフト(window)->正規化(molded_shape)を行う.
        gt_boxes = utils.pack_gt_boxes(gt_boxes, mold_scales, windows, molded_shape)
        # gt_maskをresize->pad->crop and resize
        gt_masks = utils.pack_gt_masks(gt_masks, mold_scales, pads, gt_boxes, config, sess)
        anchors = utils.make_anchors(scales, ratios, molded_images[0].shape, (4, 8, 16, 32, 64))
        rpn_gt_matchs, rpn_gt_deltas = zip(*[utils.make_matchs(anchors, b, config) for b in gt_boxes])

        # リストをndarray(batch, anchors)(batch, anchors, 4)にbatch方向にくっつける
        rpn_gt_matchs, rpn_gt_deltas = [np.stack(gts, axis=0) for gts in [rpn_gt_matchs, rpn_gt_deltas]]
        gt_boxes, gt_class_ids, gt_masks =\
                    [utils.pack_on(m) for m in (gt_boxes, gt_class_ids, gt_masks)]
        feed_dict = {
            input_images: molded_images,
            input_metas: image_metas,
            input_anchors: anchors,
            input_rpn_gt_matchs: rpn_gt_matchs,
            input_rpn_gt_deltas: rpn_gt_deltas,
            input_gt_cls_ids: gt_class_ids,
            input_gt_box: gt_boxes,
            input_gt_mask: gt_masks,
        }
        yield i, feed_dict, 

def train_generator(dataset, inputs, config, num_epoch, sess):
    input_images, input_metas, input_anchors, input_rpn_gt_deltas, input_rpn_gt_matchs,\
        input_gt_cls_ids, input_gt_box, input_gt_mask = inputs
    for i in range(num_epoch):
        scales = config.SCALES
        ratios = config.RATIOS
        if True:
            images, gt_boxes, gt_masks, gt_class_ids, img_ids = dataset.make_batch(config.BATCH_SIZE_TRAIN, img_ids=[284282, 109441])#139, 285, 32811
        else:
            images, gt_boxes, gt_masks, gt_class_ids, img_ids = dataset.make_batch(config.BATCH_SIZE_TRAIN, img_ids=None)# 226111, 58636,
        # batch毎にndarrayにする.
        molded_images, image_metas, windows, pads, mold_scales = utils.mold_images(images, max_dim=1024, min_dim=800, config=config)
        # mini_maskでないならばmaskもmoldedされる(paddingを出して)
        assert not molded_images.dtype == np.dtype("O"), "image shape is not same"
        molded_shape = molded_images.shape
        # gt_boxをresize(scale)->シフト(window)->正規化(molded_shape)を行う.
        #gt_boxes = utils.pack_gt_boxes(gt_boxes, mold_scales, windows, molded_shape)
        #gt_boxes_molded = utils.mold_gt_boxes(gt_boxes, mold_scales, windows, molded_shape)
        #gt_masks = utils.mold_gt_masks(gt_masks, mold_scales, pads, gt_boxes_molded, config)
        #gt_masks = utils.mold_gt_masks(gt_masks, mold_scales, pads)
        gt_masks = [[utils.resize_mask(m, s, p) for m in gt_mask] for gt_mask, s, p in zip(gt_masks, mold_scales, pads)]
        gt_boxes = utils.make_gt_boxes_from_mask(gt_masks)
        gt_boxes_molded = utils.mold_gt_boxes(gt_boxes, mold_scales, windows, molded_shape)
        gt_minimasks = [utils.minimize_mask(b, np.array(m), config.MINI_MASK_SIZE) for b, m in zip(gt_boxes, gt_masks)]
        #gt_minimasks = utils.pack_gt_minimasks_already_molded(gt_masks, gt_boxes_molded, config)
        # gt_maskをresize->pad->crop and resize
        #gt_masks = utils.pack_gt_minimasks(gt_masks, mold_scales, pads, gt_boxes_molded, config)
        anchors = utils.make_anchors_nomold(scales, ratios, molded_images[0].shape, (4, 8, 16, 32, 64))
        anchors_molded = utils.make_anchors(scales, ratios, molded_images[0].shape, (4, 8, 16, 32, 64))
        rpn_gt_matchs, rpn_gt_deltas = zip(*[utils.make_matchs(anchors, b, config) for b in gt_boxes])
        #rpn_gt_matchs_new, rpn_gt_deltas_new = zip(*[utils.build_rpn_targets(molded_shape, anchors, c, b, config) for b, c in zip(gt_boxes, gt_class_ids)])
        """
        import pickle
        with open("rpn.pkl", "rb") as f:
            d = pickle.load(f)
        with open("rpn_deltas.pkl", "rb") as f:
            dd = pickle.load(f)
        positive_indices_master = dd["positive_indices"][0]
        negative_indices_master = dd["negative_indices"][0]
        """
        #master_masks = np.transpose(np.squeeze(dd["batch_gt_masks"], axis=0), (2, 0, 1))
        
        #utils.apply_deltas_np(anchors, rpn_gt_deltas[0]*config.RPN_BBOX_STD_DEV)
        # リストをndarray(batch, anchors)(batch, anchors, 4)にbatch方向にくっつける

        ix = np.where(rpn_gt_matchs[0]==1)[0]
        rpn_rois = utils.apply_deltas_np(anchors[ix], rpn_gt_deltas[0]*config.RPN_BBOX_STD_DEV)
        #visualize.show_boxes_demold(images[0], gt_boxes[0], windows[0], molded_shape)
        #visualize.show_boxes_demold(images[0], rpn_rois, windows[0], molded_shape)

        rpn_gt_matchs, = [np.stack(gts, axis=0) for gts in [rpn_gt_matchs]]
        gt_boxes_molded, gt_class_ids, gt_minimasks, rpn_gt_deltas =\
                    [utils.pack_on(m) for m in (gt_boxes_molded, gt_class_ids, gt_minimasks, rpn_gt_deltas)]
        
        feed_dict = {
            input_images: molded_images,
            input_metas: image_metas,
            input_anchors: anchors_molded,
            input_rpn_gt_matchs: rpn_gt_matchs,
            input_rpn_gt_deltas: rpn_gt_deltas,
            input_gt_cls_ids: gt_class_ids,
            input_gt_box: gt_boxes_molded,
            input_gt_mask: gt_minimasks,
            #input_positive_indices_master: positive_indices_master,
            #input_negative_indices_master: negative_indices_master,
        }
        yield i, feed_dict, images, windows, molded_shape, anchors_molded[ix], img_ids

def train(keys, lr, sess, loss, dataset, inputs, config, num_epoch, ckpt_path, losses, restorer, saver):
    training_op = make_training_op(loss, keys, lr, config)
    init = tf.global_variables_initializer()
    generator = train_generator(dataset, inputs, config, num_epoch, sess)
    sess.run(init)
    restorer.restore(sess, ckpt_path)
    for i, train_feed_dict, images, windows, molded_shape, target_anchors, img_ids in generator:
        print("No.{}, ImgIds: {}".format(i, img_ids))
        if (i != 100000) and (i % 10 == 0):
            loss = sess.run(losses, feed_dict=train_feed_dict)
            print("n:{}, imgids: {}, loss: {}".format(i, img_ids, loss))
            #cal_mAP.compute_ap()
        sess.run(training_op, feed_dict=train_feed_dict)
        if (i != 0) and (i % 100 == 0):
            saver.save(sess, ckpt_path)
    