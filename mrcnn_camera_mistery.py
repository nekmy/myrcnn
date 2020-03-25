import asyncio
import time

import cv2
import tensorflow as tf
import numpy as np
from tensorflow.contrib import graph_editor
import matplotlib.pylab as plt
from matplotlib import animation
from PIL import Image

import utils
import graphs_for_camera
import trainer
from dataset import Dataset

tf.ConfigProto().gpu_options.allow_growth = True

class Mrcnn():
    def __init__(self, mode, ckpt_path, config):
        self.config = config
        if mode == "inference":
            self.input, self.output = self.mrcnn_graph(mode, config)
            self.sess = tf.Session()
            init = tf.global_variables_initializer()
            restorer = trainer.make_saver("all")
            self.sess.run(init)
            restorer.restore(self.sess, ckpt_path)

    def predict(self, images):
        input_images, input_image_metas, input_anchors = self.input
        molded_images, image_metas, windows, _, _ = utils.mold_images(images, max_dim=1024, min_dim=800, config=self.config)
        assert not molded_images.dtype == np.dtype("O"), "image shape is not same"
        molded_shape = molded_images.shape
        scales = self.config.SCALES
        ratios = self.config.RATIOS
        anchors = utils.make_anchors(scales, ratios, molded_images[0].shape, (4, 8, 16, 32, 64))
        detections, mrcnn_mask, rpn_rois, rpn_deltas = self.sess.run(self.output, feed_dict={
            input_images: molded_images,
            input_image_metas: image_metas,
            input_anchors: anchors,
        })
        results = []
        for i, image in enumerate(images):
            final_box, final_cls_id, final_score, final_mask =\
                utils.usable_image(detections[i], mrcnn_mask[i], windows[i], image.shape, molded_shape)
            results.append(
                {"boxes": final_box, 
                "cls_ids": final_cls_id,
                "scores": final_score,
                "masks": final_mask}
            )
        return results
    
    def train(self, ckpt_path, dataset, batch_size=1, lr=0.001, num_train=1):
        self.inputs, self.losses = self.mrcnn_graph("train", self.config)
        #self.feed_dict = load_dict(resnet=True)
        #self.feed_dict = load_dict()
        resnet = trainer.make_saver("resnet50")
        restorer = trainer.make_saver("all")
        saver = trainer.make_saver("all")
        input_images, image_metas, anchors,\
            input_rpn_gt_deltas, input_rpn_gt_matchs,\
            input_gt_cls_ids, input_gt_box, input_gt_mask = self.inputs
        rpn_cls_loss, rpn_delta_loss, mrcnn_cls_loss, mrcnn_delta_loss, mrcnn_mask_loss = self.losses
        mrcnn_loss = rpn_cls_loss + rpn_delta_loss + mrcnn_cls_loss + mrcnn_delta_loss + mrcnn_mask_loss
        losses = [rpn_cls_loss, rpn_delta_loss, mrcnn_cls_loss, mrcnn_delta_loss, mrcnn_mask_loss]
        scales = self.config.SCALES
        ratios = self.config.RATIOS
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            #sess.run(init, feed_dict=self.feed_dict)
            #saver.save(sess, ckpt_path)
            sess.run(init)
            resnet.restore(sess, ckpt_path)
            #restorer.restore(sess, ckpt_path)
            #saver.save(sess, ckpt_path)
            #sess.run(init)
            num_epoch = 40000

            trainer.train(
                "head", lr, sess, mrcnn_loss, dataset, self.inputs, self.config,
                num_epoch, ckpt_path, losses, restorer, saver)
            
            trainer.train(
                "res4", lr*0.1, sess, mrcnn_loss, dataset, self.inputs, self.config,
                num_epoch, ckpt_path, losses, restorer, saver)
            
            trainer.train(
                "all", lr*0.01, sess, mrcnn_loss, dataset, self.inputs, self.config,
                num_epoch, ckpt_path, losses, restorer, saver)
    
    def mrcnn_graph(self, mode, config):
        with tf.device('/cpu:0'):
            if mode == "inference":
                batch_size = config.BATCH_SIZE_INFERENCE
            if mode == "train":
                batch_size = config.BATCH_SIZE_TRAIN
            input_images = tf.placeholder(tf.float32, shape=[batch_size, 1024, 1024, config.IMAGE_SHAPE[2]], name="input_image")
            image_metas = tf.placeholder(tf.float32, shape=[batch_size, config.IMAGE_META_SIZE], name="metas")
            anchors = tf.placeholder(tf.float32, (None, 4), name="anchors")
            if mode == "train":
                input_rpn_gt_matchs = tf.placeholder(tf.int64, (batch_size, None), name="rpn_gt_match")
                input_rpn_gt_deltas = tf.placeholder(tf.float32, (batch_size, None, 4), name="gt_deltas")
                input_gt_box = tf.placeholder(tf.float32, (batch_size, None, 4), name="input_gt_box")
                input_gt_cls_ids = tf.placeholder(tf.int64, (batch_size, None,), name="gt_cls_ids")
                input_gt_mask = tf.placeholder(tf.int64, (batch_size, None) + config.MINI_MASK_SIZE, name="input_gt_mask")
            # imageをもとにfeature_mapを作る.
            C1, C2, C3, C4, C5 = graphs_for_camera.resnet(input_images)
            # feature_mapをもとにPを作る.
            psize = config.TOP_DOWN_PYRAMID_SIZE
            P5 = tf.layers.conv2d(C5, filters=psize, kernel_size=1, strides=(1, 1), name="fpn_c5p5")
            p = tf.keras.layers.UpSampling2D(size=(2, 2), name="fpn_P5_Upsampling")(P5)
            c = tf.layers.conv2d(C4, filters=psize, kernel_size=1, strides=(1, 1), name="fpn_c4p4")
            P4 = tf.add(p, c, "fpn_p4_Add")
            p = tf.keras.layers.UpSampling2D(size=(2, 2), name="fpn_P4_Upsampling")(P4)
            c = tf.layers.conv2d(C3, filters=psize, kernel_size=1, strides=(1, 1), name="fpn_c3p3")
            P3 = tf.add(p, c, "fpn_p3_Add")
            p = tf.keras.layers.UpSampling2D(size=(2, 2), name="fpn_P3_Upsampling")(P3)
            c = tf.layers.conv2d(C2, filters=psize, kernel_size=1, strides=(1, 1), name="fpn_c2p2")
            P2 = tf.add(p, c, "fpn_p2_Add")
            # Attach 3x3 conv to all P layers to get the final feature maps.
            P2 = tf.layers.conv2d(P2, filters=psize, kernel_size=3, strides=(1, 1), padding="SAME", name="fpn_p2")
            P3 = tf.layers.conv2d(P3, filters=psize, kernel_size=3, strides=(1, 1), padding="SAME", name="fpn_p3")
            P4 = tf.layers.conv2d(P4, filters=psize, kernel_size=3, strides=(1, 1), padding="SAME", name="fpn_p4")
            P5 = tf.layers.conv2d(P5, filters=psize, kernel_size=3, strides=(1, 1), padding="SAME", name="fpn_p5")
            # P6 is used for the 5th anchor scale in RPN. Generated by
            # subsampling from P5 with stride of 2.
            P6 = tf.nn.max_pool(P5, ksize=(1, 1, 1, 1), strides=(1, 2, 2, 1), padding="SAME")

            feature_map = [P2, P3, P4, P5, P6]
            mrcnn_map = [P2, P3, P4, P5]
            rpn_outputs = []
            # rpn = build_rpn_model(config.ANCHOR_STRIDE, config.ANCHOR_PER_LOCATION, config.TOP_DOWN_PYRAMID_SIZE)
            input_feature_map, output_rpn = graphs_for_camera.build_rpn_graph(config.ANCHOR_STRIDE, config.ANCHOR_PER_LOCATION, config.TOP_DOWN_PYRAMID_SIZE, config, batch_size)
            
            for P in feature_map:
                rpn_output = graph_editor.graph_replace(output_rpn, {input_feature_map: P})
                rpn_outputs.append(rpn_output)
            # [(cls_logit, cls_prob, deltas), (cls_logit, cls_prob, deltas), (cls_logit, cls_prob, deltas)]
            # =>[[cls_logit, cls_logit, ...], [cls_prob, cls_prob, ...], [deltas, deltas, ...]]
            rpn_outputs = zip(*rpn_outputs)
            # わざわざnameつける必要あるのか.
            names = ["rpn_cls_logit", "rpn_cls_prob", "rpn_deltas"]
            # (HW2+HW3+HW4+HW5+HW6*A/pix, 2or4)
            rpn_cls_logit, rpn_cls_prob, rpn_deltas =\
                [tf.concat(output, name=name, axis=1) for output, name in zip(rpn_outputs, names)]
            # nmsで対象roisを減らす
            
            rpn_rois, pre_nms_box = graphs_for_camera.proposal_graph(config.POST_NMS_PROPOSALS_INFERENCE,\
                config.NMS_THRESHOLD, self.config, batch_size, rpn_cls_prob, rpn_deltas, anchors)
            # 以降はboxの処理がroisがベースになる
            # rpn_roisの範囲の特徴量マップをRoiAlignで加工し, fpnにかける.
            # 各roiのクラス, 枠の補正を出力する.
            metas = utils.metas_converter(image_metas)

        if mode == "train":
            # 学習のtargetを決める
            # rpn_roisは条件を満たすものからランダムに選ばれる.
            rpn_rois, gt_cls_ids, gt_mrcnn_deltas, gt_mask = utils.batch_slice(
                (rpn_rois, input_gt_box, input_gt_cls_ids, input_gt_mask),
                self.config.BATCH_SIZE_TRAIN,
                lambda x, y, z, w: graphs_for_camera.detection_target_graph(x, y, z, w, config))
                #lambda x, y, z, w: graphs_for_camera.detection_target_graph(x, y, z, w, config, positive_indices_master, negative_indices_master))
            mrcnn_cls_logit, mrcnn_cls_prob, mrcnn_deltas = graphs_for_camera.fpn_classifer_graph(mrcnn_map, rpn_rois, image_metas,
                config.POOL_SIZE, config.NUM_CLASSES, config.FPN_CLASSIFER_FC_FILTER_SIZE, training=False)
            mrcnn_mask = graphs_for_camera.fpn_mask_graph(rpn_rois, mrcnn_map, image_metas, config.MASK_POOLSIZE, config.NUM_CLASSES)
            # 各anchorに物体が存在するかどうか.
            # rpnのlogitとmatchs(0 or 1)を比較.
            rpn_cls_loss = graphs_for_camera.rpn_cls_loss_func(rpn_cls_logit, input_rpn_gt_matchs)
            # 物体がある項目(match==1)について回帰を行う.
            rpn_delta_loss, target_gt_rpn_deltas, target_rpn_deltas = graphs_for_camera.rpn_delta_loss_func(rpn_deltas, input_rpn_gt_deltas, input_rpn_gt_matchs, config)
            # 
            active_ids = utils.metas_converter(image_metas)["activate_class_ids"]
            mrcnn_cls_loss = graphs_for_camera.mrcnn_cls_loss_func(mrcnn_cls_logit, gt_cls_ids, active_ids)
            mrcnn_delta_loss = graphs_for_camera.mrcnn_delta_loss_func(gt_cls_ids, gt_mrcnn_deltas, mrcnn_deltas)
            mrcnn_mask_loss = graphs_for_camera.mrcnn_mask_loss_func(gt_cls_ids, mrcnn_mask, gt_mask)
            losses = [rpn_cls_loss, rpn_delta_loss, mrcnn_cls_loss, mrcnn_delta_loss, mrcnn_mask_loss]
            inputs = [input_images, image_metas, anchors, input_rpn_gt_deltas, input_rpn_gt_matchs, input_gt_cls_ids, input_gt_box, input_gt_mask]
            return inputs, losses

        if mode == "inference":
            mrcnn_cls_logit, mrcnn_cls_prob, mrcnn_deltas =\
                graphs_for_camera.fpn_classifer_graph(mrcnn_map, rpn_rois, image_metas,
                config.POOL_SIZE, config.NUM_CLASSES, config.FPN_CLASSIFER_FC_FILTER_SIZE, training=False)
            detections = graphs_for_camera.detection_graph(config, rpn_rois, mrcnn_cls_prob, mrcnn_deltas, image_metas)
            detection_box, cls_id, score = detections[:, :, :4], detections[:, :, 4], detections[:, :, 5]
            mrcnn_mask = graphs_for_camera.fpn_mask_graph(detection_box, mrcnn_map, image_metas, config.MASK_POOLSIZE, config.NUM_CLASSES)
            inputs = [input_images, image_metas, anchors]
            outputs = [detections, mrcnn_mask, rpn_rois, rpn_deltas]
            return inputs, outputs


class MrcnnCamera():
    def __init__(self, ckpt_path, show=True):
        config = Config()
        self.mrcnn = Mrcnn(mode="inference", ckpt_path=ckpt_path, config=config)
        if show:
            self.set_camera()
        self.class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
        if show:
            self.predict_and_show((100, 100), 0.1)
        

    def make_jpg(self, image, figsize, save_name):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        result = self.mrcnn.predict([image])[0]
        utils.display(ax, image, result["boxes"], result["cls_ids"], result["scores"],
            result["masks"], (100, 100), self.class_names, save_name=save_name, show=True)
        plt.close(fig)

    def predict_and_show(self, figsize, delay_sec):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        #_, ax = plt.subplots(1, figsize=figsize)
        loop = asyncio.get_event_loop()
        future = asyncio.ensure_future(self.update_temporaly(ax, delay_sec))
        futures = (future,)
        #can = asyncio.ensure_future(self.key_interrupt_and_cancel(futures))
        futures = asyncio.gather(*futures)# + (can,))
        loop.run_until_complete(futures)
        print("end")
    
    @asyncio.coroutine
    def update_temporaly(self, ax, delay_sec):
        while True:
            try:
                yield from asyncio.gather(
                    self.update(ax),
                    asyncio.sleep(delay_sec)
                )
            except asyncio.CancelledError:
                break
    
    def set_camera(self, device_num=0):
        self.cap = cv2.VideoCapture(device_num)
        pass
    
    async def update(self, ax):
        ret, frame = self.cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.mrcnn.predict([image])[0]
        #utils.display_CV2(image, result["boxes"], result["cls_ids"], result["scores"], result["masks"], (100, 100), self.class_names)
        utils.display(ax, image, result["boxes"], result["cls_ids"], result["scores"], result["masks"], (100, 100), self.class_names)

        
    async def key_interrupt_and_cancel(self, futures):
        while True:
            #await asyncio.sleep(10.0)
            await input()
            self.cancel_futures(futures)
        return


    def cancel_futures(self, futures):
        for future in futures:
            future.cancel()

class Config:
    STD = 0.1
    IMAGE_SHAPE = (1024, 1024, 3)
    BATCH_SIZE_INFERENCE = 1
    BATCH_SIZE_TRAIN = 2
    TOP_DOWN_PYRAMID_SIZE = 256
    SCALES = [32, 64, 128, 256, 512]
    RATIOS = [0.5, 1.0, 2.0]
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])
    PRE_NMS_PROPOSALS_INFERENCE = 6000
    POST_NMS_PROPOSALS_INFERENCE = 2000
    NMS_THRESHOLD = 0.7
    POOL_SIZE = 7
    FPN_CLASSIFER_FC_FILTER_SIZE = 1024
    NUM_CLASSES = 81
    MIN_SCORE = 0.7
    MAX_NUM_ROIS = 100
    DETECTION_NMS_THRESHOLD = 0.3
    ANCHOR_STRIDE = (1, 1)
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    MASK_POOLSIZE = 14
    MASK_SHAPE = (28, 28)
    MINI_MASK_SIZE = (56, 56)
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256
    TRAIN_ROIS_NUM = 200
    POSITIVE_RATIO = 0.7
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0001
    CLIP_NORM = 1.0
    def __init__(self):
        self.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES
        self.ANCHOR_PER_LOCATION = len(self.RATIOS)

def train():
    ckpt_path = "my_mrcnn.ckpt"
    config = Config()
    mrcnn = Mrcnn("train", ckpt_path, config)
    dataset = Dataset()
    mrcnn.train(ckpt_path, dataset)

def main():
    ckpt_path = "my_mrcnn.ckpt"
    #mrcnn_camera = MrcnnCamera(ckpt_path)
    train()

def write(name="c1"):
    import glob
    import os
    ckpt_path = "my_mrcnn.ckpt"
    mrcnn_camera = MrcnnCamera(ckpt_path, show=False)
    img_paths = glob.glob("cap1/*.jpg")
    dir3 = "cap5"
    if not os.path.exists(dir3):
        os.mkdir(dir3)
    for n, img_path in enumerate(img_paths):
        image = np.array(Image.open(img_path))
        #image = np.transpose(image, (1, 0, 2))[:, ::-1, :]
        mrcnn_camera.make_jpg(image, (10, 10), "cap5/{}_{}.jpg".format(name, str(n).zfill(4)))

def write2(name="c1"):
    import glob
    import os
    ckpt_path = "my_mrcnn.ckpt"
    mrcnn_camera = MrcnnCamera(ckpt_path, show=False)
    img_path = glob.glob("images/*")[3]
    name = "mmu"
    image = np.array(Image.open(img_path))
    #image = np.transpose(image, (1, 0, 2))[:, ::-1, :]
    mrcnn_camera.make_jpg(image, (10, 10), "images/{}_momou.jpg".format(name))

if __name__ == "__main__":
    write2()