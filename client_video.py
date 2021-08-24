import argparse
import numpy as np
import sys
import cv2
import time
import os

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

from processing import preprocess, postprocess
from render import render_box, render_filled_box, get_text_size, render_text, RAND_COLORS
from labels import COCOLabels


def client(frame, ip, port=':8001'):
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--model',
                        type=str,
                        required=False,
                        default='yolov5s',
                        help='Inference model name, default yolov5s')
    parser.add_argument('-c',
                        '--confidence',
                        type=float,
                        required=False,
                        default=0.5,
                        help='Confidence threshold for detected objects, default 0.5')
    parser.add_argument('-n',
                        '--nms',
                        type=float,
                        required=False,
                        default=0.5,
                        help='Non-maximum suppression threshold for filtering raw boxes, default 0.5')
    parser.add_argument('-y',
                        '--mask_y',
                        type=int,
                        default=150,
                        help='y coordinate of mask')

    FLAGS = parser.parse_args()

    # Create server context
    url = ip + port
    try:
        triton_client = grpcclient.InferenceServerClient(
            url=url,
            verbose=False,
            ssl=False,
            root_certificates=None,
            private_key=None,
            certificate_chain=None)
    except Exception as e:
        print("context creation failed: " + str(e))
        sys.exit()

    # Health check
    if not triton_client.is_server_live():
        print("FAILED : is_server_live")
        sys.exit(1)

    if not triton_client.is_server_ready():
        print("FAILED : is_server_ready")
        sys.exit(1)

    if not triton_client.is_model_ready(FLAGS.model):
        print("FAILED : is_model_ready")
        sys.exit(1)

    t0 = time.time()

    inputs = []
    outputs = []
    inputs.append(grpcclient.InferInput('data', [1, 3, 640, 640], "FP32"))
    outputs.append(grpcclient.InferRequestedOutput('prob'))

    input_image_buffer = preprocess(frame, FLAGS.mask_y)
    input_image_buffer = np.expand_dims(input_image_buffer, axis=0)
    inputs[0].set_data_from_numpy(input_image_buffer)

    results = triton_client.infer(model_name=FLAGS.model,
                                  inputs=inputs,
                                  outputs=outputs,
                                  client_timeout=None)

    result = results.as_numpy('prob')
    detected_objects = postprocess(result, frame.shape[1], frame.shape[0], FLAGS.confidence, FLAGS.nms)

    for box in detected_objects:
        # 绘制边界框
        frame = render_box(frame, box.box(), color=tuple(RAND_COLORS[box.classID]))
        # 绘制文本框并打印类别信息
        size = get_text_size(frame, f"{COCOLabels(box.classID).name}", normalised_scaling=0.6)
        frame = render_filled_box(frame, (box.x1 - 3, box.y1 - 3, box.x1 + size[0], box.y1 + size[1]),
                                  color=tuple(RAND_COLORS[box.classID]))
        frame = render_text(frame, f"{COCOLabels(box.classID).name}", (box.x1, box.y1), color=(255, 255, 255),
                            normalised_scaling=0.5)

    # 计时
    t1 = time.time()
    print("time:%.2fs/frame" % (t1 - t0))

    return frame


if __name__ == '__main__':
    cap = cv2.VideoCapture('./daolu1.avi')
    while True:
        ret, frame = cap.read()
        print("######################")
        if ret:
            frame = client(frame, ip='192.168.2.100')
            cv2.imshow('frame', frame)
            cv2.waitKey(1)
        else:
            break
    cap.release()
    # cv2.destroyAllWindows()
