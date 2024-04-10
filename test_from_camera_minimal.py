import argparse
import cv2
import numpy as np
import onnxruntime
from utils.align_face import dealign, align_img
from utils.prepare_data import LandmarkModel



def video_test(args):
    landmarkModel = LandmarkModel(name='landmarks')
    landmarkModel.prepare(ctx_id= 0, det_thresh=0.1, det_size=(320,320))

    ort_session = onnxruntime.InferenceSession("model_static_sim.onnx", providers=['CPUExecutionProvider'])

    cap = cv2.VideoCapture(0)  # Use camera index 0 for the default webcam
    while True:
        ret, frame = cap.read()
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

    parser = argparse.ArgumentParser(description="MobileFaceSwap Test")

    parser = argparse.ArgumentParser(description="MobileFaceSwap Test")
    parser.add_argument('--source_img_path', default='nicholas-cage.jpg', type=str, help='path to the source image')
    parser.add_argument('--image_size', type=int, default=256,help='size of the test images (224 SimSwap | 256 FaceShifter)')
    parser.add_argument('--merge_result', type=bool, default=True, help='output with whole image')
    parser.add_argument('--use_gpu', type=bool, default=False)

    args = parser.parse_args()
    video_test(args)