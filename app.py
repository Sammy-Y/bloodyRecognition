from flask import Flask, jsonify, request
from flask_cors import CORS
from OpenSSL import SSL
import base64
import cv2 
import numpy as np
import easyocr
from PIL import Image
from ultralytics import YOLO
import uvicorn

# context = SSL.Context(SSL.SSLv23_METHOD)
model = YOLO('model/v2.pt')
number_model = YOLO('model/numberRecg/20240213v2.pt')
app = Flask(__name__)
CORS(app)

# NMS 非极大值抑制
def non_max_suppression(bounding_boxes, confidence_scores, threshold):
    if len(bounding_boxes) == 0:
        return [], []

    # Convert bounding boxes and confidence scores to numpy arrays
    bounding_boxes = np.array(bounding_boxes)
    confidence_scores = np.array(confidence_scores)

    # Sort bounding boxes by their left x-coordinate
    sorted_indices = np.argsort(bounding_boxes[:, 0])

    # Reorder bounding boxes and confidence scores according to the sorted indices
    bounding_boxes = bounding_boxes[sorted_indices]
    confidence_scores = confidence_scores[sorted_indices]

    # Get coordinates of bounding boxes
    start_x = bounding_boxes[:, 0]
    start_y = bounding_boxes[:, 1]
    end_x = bounding_boxes[:, 2]
    end_y = bounding_boxes[:, 3]

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score in descending order
    order = np.argsort(confidence_scores)[::-1]

    picked_boxes = []
    picked_scores = []

    while len(order) > 0:
        # Pick the bounding box with the highest confidence score
        index = order[0]
        picked_boxes.append(bounding_boxes[index])
        picked_scores.append(confidence_scores[index])

        # Compute coordinates of intersection-over-union (IOU)
        x1 = np.maximum(start_x[index], start_x[order[1:]])
        x2 = np.minimum(end_x[index], end_x[order[1:]])
        y1 = np.maximum(start_y[index], start_y[order[1:]])
        y2 = np.minimum(end_y[index], end_y[order[1:]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[1:]] - intersection)

        # Remove overlapping bounding boxes
        remaining_indices = np.where(ratio <= threshold)[0]
        order = order[remaining_indices + 1]

    return picked_boxes, picked_scores

@app.route('/api/testAPI', methods=['GET'])
def testAPI():
    return jsonify({
        'message' : 'this is testAPI'
        })

@app.route('/api/postPic', methods=['POST'])
def postPic():
    data = request.json
    # 獲取base64資料
    image_data = data.get('params').get('img_base64')
    image_data = image_data[image_data.index(',')+1:]
    # Decode base64 image data
    image_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(image_bytes, np.uint8)
    # 轉為 img
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    result_number = {}
    # 進行 yolo 辨識
    result = model(image)
    for r in result:
        border_boxes = r.boxes
        # 先處理 sys dia pul的辨識
        for box in border_boxes:
            b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
            c = box.cls

            #add
            object_name = model.names[int(c)]
            x, y, w, h = map(int, b.tolist())  # 去除最後一個元素並轉換為整數
            # print(x, y, w, h)
            tempImg = image[y:h, x:w]
            # tempImg = cv2.cvtColor(tempImg, cv2.COLOR_RGB2GRAY)
            # cv2.imshow('YOLO V8 Detection', tempImg)
            #  將圖片轉換為連續的數據
            tempImg = np.ascontiguousarray(tempImg)
            # 辨識出sys, dia, pul的數字
            regNumImgResult = number_model(tempImg)
            for n in regNumImgResult:
                boxes = n.boxes
                # 从当前对象中提取xyxy和confidence数据
                bounding_boxes = boxes.xyxy
                confidence_scores = boxes.conf
                # nms 處理 使用非最大值抑制处理边界框
                picked_boxes, picked_score = non_max_suppression(bounding_boxes, confidence_scores, 0.5)

                picked_boxes = np.array(picked_boxes)
                picked_score = np.array(picked_score)

                # 对边界框按照左上角 x 坐标进行排序，確定有值才處理
                if picked_boxes.size > 0:
                    sorted_indices = np.argsort(picked_boxes[:, 0])
                    picked_boxes = picked_boxes[sorted_indices]
                    picked_score = picked_score[sorted_indices]

                # 遍历NMS结果
                number = ''
                for i, (picked_box, picked_confidence) in enumerate(zip(picked_boxes, picked_score)):
                # 在原始对象中找到与NMS结果匹配的边界框
                    for j, box in enumerate(boxes):
                        # 将PyTorch张量转换为NumPy数组
                        box_xyxy_np = box.xyxy.cpu().numpy()
                        # 检查边界框是否与NMS结果相似（这里可以根据需要自定义相似性条件）
                        if (abs(picked_box - box_xyxy_np) < 1.0).all() and abs(picked_confidence - box.conf.item()) < 0.01:
                            # 打印匹配对象的类别信息
                            number += number_model.names[int(box.cls.item())]
                            print(f"Object {i+1} class:", box.cls.item())
                print('object_name',object_name)
                print('number', number)
                if(number != ''):
                    result_number[object_name] = number
            print('result_number', result_number)   

    return jsonify({
        'success': True,
        'data': result_number
        })

@app.route('/api/upload', methods=['POST'])
def upload_image():
    data = request.json
    image_data = data.get('image')

    # Decode base64 image data
    image_bytes = base64.b64decode(image_data)
    # nparr = np.frombuffer(image_bytes, np.uint8)
    # image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Perform image recognition (replace this with your actual image processing code)
    # result = recognize_image(image)

    return jsonify(result=result)

def recognize_image(image):
    # Your image recognition logic goes here
    # Replace this placeholder with your actual implementation
    return {'recognized': 'Object'}

if __name__ == '__main__': #http
    # app.run(debug=True, host='192.168.1.105')
    app.run(debug=True, host='192.168.1.127')
# if __name__ == '__main__': #https
#     app.run(debug=True, ssl_context='adhoc')