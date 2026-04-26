from inference_engine import predict_from_frame
import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    pred, conf = predict_from_frame(frame)

    print("PRED:", pred, "CONF:", conf)

    cv2.imshow("test", frame)

    if cv2.waitKey(1) == 27:
        break