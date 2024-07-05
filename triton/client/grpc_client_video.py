import sys
import time
import argparse
import numpy as np
import cv2

import tritonclient.grpc as grpcclient

from config import *
from preprocess import preprocess
from postprocess import postprocess

parser = argparse.ArgumentParser()
parser.add_argument('-m',
                    '--model',
                    type=str,
                    required=False,
                    default='yolov8s',
                    help='Inference model name, default yolov8s')
parser.add_argument('-v',
                    '--video',
                    type=str,
                    required=False,
                    # default=r'D:\projects\self_test\HRNet-Semantic-Segmentation-HRNet-OCR\images\01.jpeg',
                    default=r'videos/street_01.mp4',
                    help='Source image path')

FLAGS = parser.parse_args()

# Create server context
ip = '192.168.0.230'
port = '32803'
url = ip + ":" + port
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


def draw_image(detected_list, image, line_color=(255, 0, 255), label_color=(255, 255, 255), line_thickness=2):
    for x1, y1, x2, y2, conf, class_id in detected_list:
        c1, c2 = (x1, y1), (x2, y2)
        cv2.rectangle(image, c1, c2, line_color, thickness=line_thickness, lineType=cv2.LINE_AA)

        label = f"{class_name_list[class_id]} {conf:.2f}"
        # label = class_name_list[class_id]
        t_size = cv2.getTextSize(label, 0, fontScale=line_thickness / 3, thickness=line_thickness)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, line_color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, label, (c1[0], c1[1] - 2), 0, line_thickness / 3, label_color, thickness=line_thickness,
                    lineType=cv2.LINE_AA)


def client(image_input):
    t0 = time.time()

    inputs = []
    outputs = []
    inputs.append(grpcclient.InferInput('images', [1, 3, kInputH, kInputW], "FP32"))
    outputs.append(grpcclient.InferRequestedOutput('output0'))

    input_image_buffer = preprocess(image_input, kInputH, kInputW).astype(np.float32)
    input_image_buffer = np.expand_dims(input_image_buffer, axis=0)
    inputs[0].set_data_from_numpy(input_image_buffer)

    results = triton_client.infer(model_name=FLAGS.model,
                                  inputs=inputs,
                                  outputs=outputs,
                                  client_timeout=None)

    result = results.as_numpy('output0')
    result = result.reshape((4 + kNumClass, -1))
    detect_res = postprocess(image_input, result, kConfThresh, kNmsThresh, kInputH, kInputW)

    t1 = time.time()
    print("time: %.2f ms/frame" % ((t1 - t0) * 1000))

    draw_image(detect_res, image_input)


if __name__ == '__main__':
    cap = cv2.VideoCapture(FLAGS.video)
    while True:
        ret, frame = cap.read()
        print("=" * 50)
        if ret:
            client(frame)
            cv2.imshow('frame', frame)
            cv2.waitKey(1)
        else:
            break
    cap.release()
