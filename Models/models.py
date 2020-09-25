import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.losses import binary_crossentropy, sparse_categorical_crossentropy
from .layers import *
from .model_config import *
from .utils import broadcast_iou

def darknetConv(x, filters, size, strides=1, batch_norm=True):
    if strides == 1: padding = 'same'
    else:
        x = zeroPadding2D(((1, 0), (0, 1)))(x)
        padding = 'valid'
    x = conv2D(filters, size, strides, padding, batch_norm)(x)
    if batch_norm:
        x = batchNormalization()(x)
        x = leakyRelu(alpha=0.1)(x)
    return x


def darknetResidual(x, filters):
    prev = x
    x = darknetConv(x, filters // 2, 1)
    x = darknetConv(x, filters, 3)
    return add()([prev, x])


def darknetBlock(x, filters, blocks):
    x = darknetConv(x, filters, 3, strides=2)
    for _ in range(blocks):
        x = darknetResidual(x, filters)
    return x


def darkNet(name=None):
    x = inputs = Input([None, None, 3])
    x = darknetConv(x, 32, 3)
    x = darknetBlock(x, 64, 1)
    x = darknetBlock(x, 128, 2)
    x = x_36 = darknetBlock(x, 256, 8)
    x = x_61 = darknetBlock(x, 512, 8)
    x = darknetBlock(x, 1024, 4)
    return Model(inputs=inputs, outputs=(x_36, x_61, x), name=name)


def yoloConv(filters, name=None):
    def yolo_conv(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:], Input(x_in[1].shape[1:]))
            x, x_skip = inputs

            x = darknetConv(x, filters, 1)
            x = upsampling2D(size=2)(x)
            x = concatenate()([x, x_skip])
        else: x = inputs = Input(x_in.shape[1:])

        x = darknetConv(x, filters, 1)
        x = darknetConv(x, filters * 2, 3)
        x = darknetConv(x, filters, 1)
        x = darknetConv(x, filters * 2, 3)
        x = darknetConv(x, filters, 1)
        return Model(inputs=inputs, outputs=x, name=name)(x_in)
    return yolo_conv


def yoloOutput(filters, anchors, classes, name=None):
    def yolo_output(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = darknetConv(x, filters * 2, 3)
        x = darknetConv(x, anchors * (classes + 5), 1, batch_norm=False)
        x = Lambda(lambda x: tf.reshape(x,
                                        (-1, tf.shape(x)[1], tf.shape(x)[2], anchors, classes + 5)))(x)
        return Model(inputs=inputs, outputs=x, name=name)(x_in)
    return yolo_output


def yoloBoxes(pred, anchors, classes):
    grid_size = tf.shape(pred)[1:3]
    box_xy, box_wh, objectness, class_probs = tf.split(pred, (2, 2, 1, classes), axis=-1)

    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)
    pred_box = tf.concat((box_xy, box_wh), axis=-1)

    grid = tf.meshgrid(tf.range(grid_size[1]), tf.range(grid_size[0]))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(box_wh) * anchors

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs, pred_box


def yoloNMS(outputs, anchors, masks, classes):
    # box, confidence, type
    b, c, t = [], [], []

    for o in outputs:
        b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

    bbox = tf.concat(b, axis=-1)
    confidence = tf.concat(c, axis=1)
    class_probs = tf.concat(t, axis=1)

    scores = confidence * class_probs
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(
            scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=yolo_max_boxes,
        max_total_size=yolo_max_boxes,
        iou_threshold=yolo_iou_threshold,
        score_threshold=yolo_score_threshold
    )

    return boxes, scores, classes, valid_detections


def yoloV3(size=None, channels=3, anchors=yolo_anchors,
           masks=yolo_anchor_masks, classes=80, training=False):
    x = inputs = Input([size, size, channels], name='input')

    x_36, x_61, x = darkNet(name='yolo_darknet')(x)

    x = yoloConv(filters=512, name='yolo_conv_0')(x)
    output_0 = yoloOutput(filters=512, anchors=len(masks[0]), classes=classes, name='yolo_output_0')(x)

    x = yoloConv(filters=256, name='yolo_conv_1')(x)
    output_1 = yoloOutput(filters=256, anchors=len(masks[1]), classes=classes, name='yolo_output_1')(x)

    x = yoloConv(filters=128, name='yolo_conv_2')(x)
    output_2 = yoloOutput(filters=128, anchors=len(masks[2]), classes=classes, name='yolo_output_2')(x)

    if training: return Model(inputs=inputs,
                              outputs=(output_0, output_1, output_2),
                              name='yolov3')

    boxes_0 = Lambda(lambda x: yoloBoxes(x, anchors[masks[0]], classes), name='yolo_boxes_0')(output_0)
    boxes_1 = Lambda(lambda x: yoloBoxes(x, anchors[masks[1]], classes), name='yolo_boxes_1')(output_1)
    boxes_2 = Lambda(lambda x: yoloBoxes(x, anchors[masks[2]], classes), name='yolo_boxes_2')(output_2)

    outputs = Lambda(lambda x: yoloNMS(x, anchors, masks, classes),
                     name='yolo_nms')((boxes_0[:3], boxes_1[:3], boxes_2[:3]))

    return Model(inputs=inputs, outputs=outputs, name='yolov3')


def yoloLoss(anchors, classes=80, ignore_thresh=0.5):
    def yolo_loss(y_true, y_pred):
        # y_pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
        pred_box, pred_obj, pred_class, pred_xywh = yoloBoxes(y_pred, anchors, classes)
        pred_xy = pred_xywh[..., 0:2]
        pred_wh = pred_xywh[..., 2:4]

        # PASCAL VOC -> COCO
        # xmin, ymin, xmax, ymax -> x, y, w, h
        # y_true: (batch_size, grid, grid, anchors, (x1, y1, x2, y2, obj, cls))
        true_box, true_obj, true_class_idx = tf.split(y_true, (4, 1, 1), axis=-1)
        true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
        true_wh = true_box[..., 2:4] - true_box[..., 0:2]

        # give higher weights to small boxes
        # true_wh[..., 0] * true_wh[..., 1] : box area
        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

        grid_size = tf.shape(y_true)[1]
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        true_xy = true_xy * tf.cast(grid_size, tf.float32) - tf.cast(grid, tf.float32)
        true_wh = tf.math.log(true_wh / anchors)
        true_wh = tf.where(tf.math.is_inf(true_wh), tf.zeros_like(true_wh), true_wh)

        obj_mask = tf.squeeze(true_obj, -1)
        best_iou = tf.map_fn(
            lambda x: tf.reduce_max(broadcast_iou(x[0], tf.boolean_mask(
                x[1], tf.cast(x[2], tf.bool))), axis=-1),
            (pred_box, true_box, obj_mask), tf.float32)
        ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)

        xy_loss = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        wh_loss = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
        obj_loss = binary_crossentropy(true_obj, pred_obj)
        obj_loss = obj_mask * obj_loss + (1 - obj_mask) * ignore_mask * obj_loss
        class_loss = obj_mask * sparse_categorical_crossentropy(true_class_idx, pred_class)

        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

        return xy_loss + wh_loss + obj_loss + class_loss
    return yolo_loss


if __name__ == '__main__':
    import cv2
    import numpy as np
    from .utils import transform_images

    IMG_PATH = f'example.jpg'
    img = tf.image.decode_image(open(IMG_PATH, 'rb').read(), channels=3)
    img = tf.expand_dims(img, 0)
    img = transform_images(img, size=416)

    model = yoloV3(classes=80)
    print(model.summary())
    boxes, scores, classes, num = model(img)
    print(f'boxes : {boxes.shape}')
    print(f'scores : {scores.shape}')
    print(f'classes : {classes.shape}')
    print(f'num : {num.shape}')

