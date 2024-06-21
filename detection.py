import os
import cv2 as cv
import numpy as np
from datetime import datetime
from easyocr import Reader
# from db.postgresql import find_vehicle_id_by_number_plate
# from db.redisdb import save_anpr_reading

from config import (
    model_configuration,
    model_weights,
    classes_file,
    result_path
)
from config import (
    conf_threshold,
    nms_threshold,
    inp_width,
    inp_height
)

classes = None
with open(classes_file, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

reader = Reader(['en'])

net = cv.dnn.readNetFromDarknet(model_configuration, model_weights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

def get_output_names(net):
    layersNames = net.getLayerNames()
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]

def draw_prediction(classId, conf, left, top, right, bottom, frame, capture_counter):
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 5)# Label with confidence
    label = '%.2f' % conf
    if classes:
        assert(classId < len(classes))
        label = f"{classes[classId]}:{label}"

    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(
        frame,
        (left, top - 10 - round(1.5*labelSize[1])),
        (left + round(1.5*labelSize[0]), top - 10 + baseLine),
        (0, 70, 255),
        cv.FILLED
    )
    cv.putText(
        frame,
        label,
        (left,
        top - 10),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255,255,255),
        2
    )

    plate_region = frame[top:bottom, left:right]
    ocr_result = reader.readtext(plate_region, width_ths=0.7, link_threshold=0.4, decoder='beamsearch')
    
    detected_text = None
    detected_conf = 0
    for (_, text, prob) in ocr_result:
        text = text.upper()
        if prob > conf_threshold:
            detected_text = text
            detected_conf = prob
            print(f"OCR Detected text: {text} with confidence: {prob}")
            text_size = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            text_x = left + (right - left) // 2 - text_size[0] // 2
            text_y = top - 10
            cv.rectangle(
                frame,
                (text_x, text_y - text_size[1]),
                (text_x + text_size[0], text_y + baseLine),
                (0, 70, 255), cv.FILLED
            )
            cv.putText(
                frame,
                text,
                (text_x, text_y),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255,255,255),
                2
            )
            # vid = find_vehicle_id_by_number_plate(text)
            # if vid:
            #     save_anpr_reading(vid, text)

    if detected_conf > conf_threshold:
        save_image(frame)
        save_ocr_result(f"detected_text: {detected_text}, detected_conf: {detected_conf}")

    return detected_text, detected_conf


def process_frame(frame, capture_counter):
    blob = cv.dnn.blobFromImage(frame, 1/255, (inp_width, inp_height), [0,0,0], 1, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_names(net))

    frameHeight, frameWidth = frame.shape[:2]
    classIds, confidences, boxes = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > conf_threshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indices = cv.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    for i in indices:
        box = boxes[i]
        left, top, width, height = box[0], box[1], box[2], box[3]
        try:
            draw_prediction(classIds[i], confidences[i], left, top, left + width, top + height, frame, capture_counter)
        except Exception as e:
            print(e)

def save_image(frame):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(result_path, f"r_img-{timestamp}.jpg")
    cv.imwrite(filename, frame)
    print(f"Capture saved as {filename}")

def save_ocr_result(ocr_result):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(result_path, f"r_ocr-{timestamp}.txt")
    f = open(filename, "w")
    f.write(str(ocr_result))
    f.close()
    print(f"OCR result saved as {filename}")