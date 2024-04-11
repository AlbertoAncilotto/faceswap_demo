import argparse
import cv2
from camera import Camera
import numpy as np
import onnxruntime
onnxruntime.set_default_logger_severity(3)
from utils.align_face import dealign, align_img
from utils.prepare_data import LandmarkModel
import time

cv2.namedWindow('out', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('out',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

def video_test():
    landmarkModel = LandmarkModel(name='landmarks')
    landmarkModel.prepare(ctx_id= 0, det_thresh=0.1, det_size=(128,128))

    ort_session = onnxruntime.InferenceSession("demo_images/nc.jpg.onnx", providers=['CPUExecutionProvider'])

    cap = Camera(width=480, height=320, unzoom=2)
    while True:
        ts_frame_start = time.time()
        frame = cap.get_frame()
        landmark = landmarkModel.get(frame)
        if landmark is not None:
            att_img, back_matrix = align_img(frame, landmark)
            att_img = np.transpose(cv2.cvtColor(att_img, cv2.COLOR_BGR2RGB), (2, 0, 1)).astype(np.float32)/ 255.0
            ts_swap_start = time.time()
            ort_inputs = {ort_session.get_inputs()[0].name: att_img[None,...]}
            [res, mask] = ort_session.run(None, ort_inputs)
            ts_postprocess_start = time.time()
            res = cv2.cvtColor(np.transpose(res[0]*255, (1, 2, 0)), cv2.COLOR_RGB2BGR)
            mask = np.transpose(mask[0], (1, 2, 0))
            res = dealign(res, frame, back_matrix, mask)
            frame = res
            print('Preprocess:', ts_swap_start-ts_frame_start, 'Swapping', ts_postprocess_start-ts_swap_start, 'Postprocess', time.time()-ts_postprocess_start, 'Total:', time.time()-ts_frame_start)

        cv2.imshow("out", frame)
        cv2.waitKey(1)



if __name__ == '__main__':
    video_test()