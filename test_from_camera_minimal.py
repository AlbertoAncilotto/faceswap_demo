import argparse
import cv2
from camera import Camera
import numpy as np
import onnxruntime
from utils.align_face import dealign, align_img
from utils.prepare_data import LandmarkModel



def video_test():
    landmarkModel = LandmarkModel(name='landmarks')
    landmarkModel.prepare(ctx_id= 0, det_thresh=0.1, det_size=(320,320))

    ort_session = onnxruntime.InferenceSession("demo_images/nc.jpg.onnx", providers=['CPUExecutionProvider'])

    cap = cv2.VideoCapture(0)  # Use camera index 0 for the default webcam
    cap = Camera()
    while True:
        frame = cap.get_frame()
        landmark = landmarkModel.get(frame)
        if landmark is not None:
            att_img, back_matrix = align_img(frame, landmark)
            att_img = np.transpose(cv2.cvtColor(att_img, cv2.COLOR_BGR2RGB), (2, 0, 1)).astype(np.float32)/ 255.0
            ort_inputs = {ort_session.get_inputs()[0].name: att_img[None,...]}
            [res, mask] = ort_session.run(None, ort_inputs)
            res = cv2.cvtColor(np.transpose(res[0]*255, (1, 2, 0)), cv2.COLOR_RGB2BGR)
            mask = np.transpose(mask[0], (1, 2, 0))
            res = dealign(res, frame, back_matrix, mask)
            frame = res
        else:
            print('**** No Face Detect Error ****')
        cv2.imshow("Webcam", frame)
        cv2.waitKey(1)



if __name__ == '__main__':
    video_test()