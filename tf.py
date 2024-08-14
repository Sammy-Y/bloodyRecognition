import cv2 
import easyocr
import base64
import numpy as np
import torch
from torchvision.ops import nms
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated
from PIL import Image
from io import BytesIO
import pytesseract


model = YOLO('model/20240211v1.pt')
number_model = YOLO('model/numberRecg/20240601v1.pt')
# img_path = 'image.png'
img_path = 'differ_img.jpeg'

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
        print("iou", ratio)
        # Remove overlapping bounding boxes
        remaining_indices = np.where(ratio <= threshold)[0]
        order = order[remaining_indices + 1]

    return picked_boxes, picked_scores

def non_ms(bounding_boxes, confidence_score, threshold):
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []
    # Bounding boxes
    boxes = np.array(bounding_boxes)
    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]
    # Confidence scores of bounding boxes
    score = np.array(confidence_score)
    # Picked bounding boxes
    picked_boxes = []
    picked_score = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return picked_boxes, picked_score

data = open('base64.txt','r').read()
data = data[data.index(',')+1:] # 只取後面base64的值
image_bytes = base64.b64decode(data)
img = Image.open(BytesIO(image_bytes))
img = img.convert('RGB')
img.save('image.png')
# 轉為 img   

result_number = {}
img_path = 'image.png'
img = cv2.imread(img_path) # read image


# Perform inference 
result = model(img) # 先辨識出sys, dia, pul 的位置
for r in result:
    annotator = Annotator(img)
    border_boxes = r.boxes
    # 先處理 sys dia pul的辨識
    for box in border_boxes:
        b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
        c = box.cls

        #add
        object_name = model.names[int(c)]
        x, y, w, h = map(int, b.tolist())  # 去除最後一個元素並轉換為整數
        # print(x, y, w, h)
        tempImg = img[y:h, x:w]
        # tempImg = cv2.cvtColor(tempImg, cv2.COLOR_RGB2GRAY)
        # cv2.imshow('YOLO V8 Detection', tempImg)
        #  將圖片轉換為連續的數據
        tempImg = np.ascontiguousarray(tempImg)
        # 辨識出sys, dia, pul的數字
        regNumImgResult = number_model(tempImg)
        for n in regNumImgResult:
            anta = Annotator(tempImg)
            boxes = n.boxes
            # 从当前对象中提取xyxy和confidence数据
            bounding_boxes = boxes.xyxy
            confidence_scores = boxes.conf
            # nms 處理 使用非最大值抑制处理边界框
            picked_boxes, picked_score = non_max_suppression(bounding_boxes, confidence_scores, 0.5)

            picked_boxes = np.array(picked_boxes)
            picked_score = np.array(picked_score)

            # 对边界框按照左上角 x 坐标进行排序
            # sorted_indices = np.argsort(picked_boxes[:, 0])
            # picked_boxes = picked_boxes[sorted_indices]
            # picked_score = picked_score[sorted_indices]

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
            result_number[object_name] = number
        print('result_number', result_number)
        #add

        # object_name = model.names[int(c)]
        # conf = box.conf.item() # 獲取信任值
        
        # # 換數字辨識
        # x, y, w, h = map(int, b.tolist())  # 去除最後一個元素並轉換為整數
        # tempImg = img[y:h, x:w]
        # #  將圖片轉換為連續的數據
        # tempImg = np.ascontiguousarray(tempImg)
        # # 辨識出sys, dia, pul的數字
        # regNumImgResult = number_model(tempImg)
        # for n in regNumImgResult:
        #     anta = Annotator(tempImg)
        #     number_boxes = n.boxes
        #     number_conf = n.boxes.conf
        #     # 按照辨识框的左上角 x 坐标进行排序 （由左至右數字排序）
        #     sorted_indices = torch.argsort(torch.tensor([box.xyxy[0][0] for box in number_boxes]))
        #     # 根据排序后的索引对检测框进行重排
        #     sorted_boxes = [number_boxes[i] for i in sorted_indices]

        #     # 从当前对象中提取xyxy和confidence数据
        #     bounding_boxes = boxes.xyxy
        #     confidence_scores = boxes.conf
        #     # nms 處理 使用非最大值抑制处理边界框
        #     picked_boxes, picked_score = non_max_suppression(bounding_boxes, confidence_scores, 0.5)
        #     picked_boxes = np.array(picked_boxes)
        #     picked_score = np.array(picked_score)

        #     # 对边界框按照左上角 x 坐标进行排序
        #     sorted_indices = np.argsort(picked_boxes[:, 0])
        #     picked_boxes = picked_boxes[sorted_indices]
        #     picked_score = picked_score[sorted_indices]
        #     # 辨識數字
        #     number = ''
        #     for i, (picked_box, picked_confidence) in enumerate(zip(picked_boxes, picked_score)):
        #         for j, box in enumerate(boxes):
        #             # 将PyTorch张量转换为NumPy数组
        #             box_xyxy_np = box.xyxy.cpu().numpy()
        #             # 检查边界框是否与NMS结果相似（这里可以根据需要自定义相似性条件）
        #             if (abs(picked_box - box_xyxy_np) < 1.0).all() and abs(picked_confidence - box.conf.item()) < 0.01:
        #                 # 打印匹配对象的类别信息
        #                 number += number_model.names[int(box.cls.item())]
        #                 print(f"Object {i+1} class:", box.cls.item())
        #     result_number[object_name] = number
        #     print('result_number', result_number)
            
            # 辨識出數字
            # for box in sorted_boxes:
            #     b = box.xyxy[0]
            #     number_c = box.cls
            #     conf = box.conf.item() # 獲取信任值
            #     # 標記出數字
            #     anta.box_label(b, f"{number_model.names[int(number_c)]} {conf:.2f}")
            #     number += number_model.names[int(number_c)]
            # result_number[object_name] = number
        # annotator.box_label(b, f"{model.names[int(c)]} {conf:.2f}")
    # print('result', result_number)

    # print(border_boxes)
    b = border_boxes[1].xyxy[0]
    c = border_boxes[1].cls
    object_name = model.names[int(c)]
    x, y, w, h = map(int, b.tolist())  # 去除最後一個元素並轉換為整數
    tempImg = img[y:h, x:w]
    #  將圖片轉換為連續的數據
    tempImg = np.ascontiguousarray(tempImg)
    # 辨識出sys, dia, pul的數字
    regNumImgResult = number_model(tempImg)
    for n in regNumImgResult:
        anta = Annotator(tempImg)
        boxes = n.boxes
        # 从当前对象中提取xyxy和confidence数据
        bounding_boxes = boxes.xyxy
        confidence_scores = boxes.conf
        # nms 處理 使用非最大值抑制处理边界框
        picked_boxes, picked_score = non_max_suppression(bounding_boxes, confidence_scores, 0.5)

        picked_boxes = np.array(picked_boxes)
        picked_score = np.array(picked_score)

        # 对边界框按照左上角 x 坐标进行排序
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
        result_number[object_name] = number
        # print(result_number)
        # 保留NMS后的边界框和置信度
        # nms_boxes = [boxes[i] for i in picked_boxes]
        # print(nms_boxes)
        # keep_indices = non_max_suppression(boxes.xyxy, boxes.cls, 0.5)
        # keep_indices = [[idx.numpy().tolist() for idx in indices] for indices in keep_indices]
        # print(keep_indices)
        # for i in keep_indices[0]:
        #     print('i ',boxes.xyxy[i])
        # nms_boxes = [boxes[i] for i in keep_indices[0][0]]
        # print(nms_boxes)
      
        # 按照辨识框的左上角 x 坐标进行排序 （由左至右數字排序）
        sorted_indices = torch.argsort(torch.tensor([box.xyxy[0][0] for box in boxes]))
        # 根据排序后的索引对检测框进行重排
        sorted_boxes = [boxes[i] for i in sorted_indices]

        ######## 20240805 MODI
        # for box in sorted_boxes:
        #     b = box.xyxy[0]
        #     number_c = box.cls
        #     conf = box.conf.item() # 獲取信任值
        #     # 標記出數字
        #     anta.box_label(b, f"{number_model.names[int(number_c)]} {conf:.2f}")
        #     number += number_model.names[int(number_c)]

        # 這邊以下是處理NMS結果的顯示圖像
        nms_boxes = []
        for picked_box, picked_confidence in zip(picked_boxes, picked_score):
            for j, box in enumerate(boxes):
                box_xyxy_np = box.xyxy.cpu().numpy()
                if (abs(picked_box - box_xyxy_np) < 1.0).all() and abs(picked_confidence - box.conf.item()) < 0.01:
                    nms_boxes.append(box)

        sorted_nms_boxes = sorted(nms_boxes, key=lambda x: x.xyxy[0][0])

        for box in sorted_nms_boxes:
            b = box.xyxy[0]
            number_c = box.cls
            conf = box.conf.item()
            anta.box_label(b, f"{number_model.names[int(number_c)]} {conf:.2f}")
            number += number_model.names[int(number_c)]

        ########
        img = anta.result()
        cv2.imshow('YOLO V8 Detection', img)  
        pil_img = Image.fromarray(img)
        pil_img = pil_img.convert('RGB')
        pil_img.save('t.png')
    # img_gray2 = cv2.medianBlur(tempImg, 5)   # 模糊化 
    # output2 = cv2.adaptiveThreshold(img_gray2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2) # 二值化
    cv2.waitKey(0)
    cv2.destroyAllWindows()