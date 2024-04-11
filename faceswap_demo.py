import os
import cv2
import numpy as np
import onnxruntime
import time
from utils.align_face import dealign, align_img
from utils.prepare_data import LandmarkModel
from camera import Camera

base_dir = 'demo_images'
MODEL_TIME = 5

cv2.namedWindow('out', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('out',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

def video_test():
    landmarkModel = LandmarkModel(name='landmarks')
    landmarkModel.prepare(ctx_id= 0, det_thresh=0.1, det_size=(320,320))

    inf_sessions = []

    for model in os.listdir(base_dir):
        if not model.endswith('onnx'):
            continue

        ort_session = onnxruntime.InferenceSession(os.path.join(base_dir, model), providers=['CPUExecutionProvider'])
        ref_img = cv2.imread(os.path.join(base_dir, model.replace('.onnx','')))
        inf_sessions.append({'session':ort_session, 'img':ref_img})
        print('loaded', model)

    # cap = cv2.VideoCapture(0)  # Use camera index 0 for the default webcam
    cap = Camera(width=480, height=320, unzoom=2)
    start = time.time()
    curr_model = 0
    ort_session = inf_sessions[curr_model]['session']
    while True:
        # ret, frame = cap.read()
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

        # overlay reference image
        resized_img = cv2.resize(inf_sessions[curr_model]['img'], (int(frame.shape[1] / 4), int(frame.shape[1] / 4)))
        h, w = resized_img.shape[:2]
        frame[:h, -w:, :] = resized_img

        cv2.imshow("out", frame)
        cv2.waitKey(1)
        if time.time() - start > MODEL_TIME:
            curr_model+=1
            curr_model%=len(inf_sessions)
            ort_session = inf_sessions[curr_model]['session']
            start = time.time()



if __name__ == '__main__':
    video_test()