import argparse
from YOLO.utils import *
from cv2 import cuda
from pipeline import *
import imutils

if cuda.getCudaEnabledDeviceCount() > 0:
    cuda.setDevice(0)

cfgfile = "YOLO/yolov4-tiny.cfg"
weightfile = "YOLO/yolov4-tiny.weights"
namesfile = "YOLO/coco.names"
class_names = load_class_names(namesfile)


def predict_video(video, expand=0.05, backend="cuda", k=5, conf=0.7, nms=0.01):

    if video == "webcam":
        video = 0

    expanding_factor = 0

    cv2.namedWindow("window")
    cap = cv2.VideoCapture(video)

    while cap.isOpened():

        ret, frame = cap.read()
        start_time = time.time()
        gpu_frame = cv2.cuda_GpuMat(frame)
        stream = cv2.cuda_Stream()
        gpu_frame.upload(frame, stream)
        frame = gpu_frame.download(stream)

        if ret:

            frame = imutils.resize(frame, width=1000, height=480)
            width = frame.shape[1]
            height = frame.shape[0]

            with open('YOLO/coco.names', 'r') as f:
                classes = [line.strip() for line in f.readlines()]

            colors = np.random.uniform(0, 255, size=(len(classes), 3))

            net = cv2.dnn.readNet(weightfile, cfgfile)

            if backend == "cuda":
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            else:
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(get_output_layers(net))

            class_ids = []
            confidences = []
            boxes = []
            conf_threshold = conf
            nms_threshold = nms

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = center_x - w / 2
                        y = center_y - h / 2
                        class_ids.append(class_id)
                        confidences.append(float(confidence))
                        boxes.append([x, y, w, h])

            indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

            indices = indices[0] if len(indices) > 0 else []

            for i in indices:
                box = boxes[i]
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]

                try:
                    sentence, word_list, label = draw_prediction(frame, class_ids[i],
                                                                 int(x - expanding_factor * x),
                                                                 int(y - expanding_factor * y),
                                                                 int(x + w + expanding_factor * (x + w)),
                                                                 int(y + h + expanding_factor * (y + h)),
                                                                 colors, classes, k)
                except TypeError:
                    continue

                if not any(word == label for word in word_list):
                    # only expand if withing the frame boundaries
                    if (x - expanding_factor * x) > 0 and (y - expanding_factor * y) > 0 and \
                            (x + w + expanding_factor * (x + w)) < width and \
                            (y + h + expanding_factor * (y + h)) < height:
                        expanding_factor += expand

                end_time = time.time()
                inference_time = end_time - start_time

                FPS = 1.0 / inference_time

                cv2.putText(frame, "FPS: {:.2f}".format(FPS), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("window", frame)

            key = cv2.waitKey(20)

            if key > 0:  # exit by pressing any key
                cv2.destroyAllWindows()

                for i in range(1, 5):
                    cv2.waitKey(1)
                return
        else:
            continue

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()


if __name__ == '__main__':

    print("Inference on: ", cuda.getCudaEnabledDeviceCount(), "GPU(s)")
    print("Device: ", device)

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="videos/cow.mp4", help="path to video file")
    parser.add_argument("--expand", type=float, default=0.05, help="pyramid image expanding factor")
    parser.add_argument("--backend", type=str, default="cuda", help="backend to use (cuda or cpu)")
    parser.add_argument("--k", type=int, default=5, help="number of hypotheses held in the beam")
    parser.add_argument("--conf", type=float, default=0.7, help="confidence threshold")
    parser.add_argument("--nms", type=float, default=0.01, help="nms threshold")

    args = parser.parse_args()

    predict_video(args.video, args.expand, args.backend, args.k, args.conf, args.nms)
