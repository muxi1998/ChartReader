from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
from PIL import Image
from setuptools import setup
from io import open
import re
from paddleocr import PaddleOCR, draw_ocr

from mmdet.apis import init_detector, inference_detector
import mmcv
import os

import torch
import json

chart_dete_config_file = './ChartDete/work_dirs/cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss/cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss.py'
chart_dete_checkpoint_file = './ChartDete/work_dirs/cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss/checkpoint.pth'

CLASSES = ['x_title', 'y_title', 'plot_area', 'other', 'xlabel', 'ylabel', 'chart_title', 'x_tick', 'y_tick', 'legend_patch', 'legend_label', 'legend_title', 'legend_area', 'mark_label', 'value_label', 'y_axis_area', 'x_axis_area', 'tick_grouping']

# build the model from a config file and a checkpoint file
chart_dete_model = init_detector(chart_dete_config_file, chart_dete_checkpoint_file, device='cpu')

# Example result frome OCR
# [[[156.0, 15.0], [485.0, 15.0], [485.0, 31.0], [156.0, 31.0]], ('Control Chart with Stability Analysis', 0.9464997053146362)]
# [[[33.0, 45.0], [65.0, 45.0], [65.0, 58.0], [33.0, 58.0]], ('10.6', 0.9948348999023438)]
# [[[39.0, 76.0], [63.0, 76.0], [63.0, 92.0], [39.0, 92.0]], ('9.6', 0.999164879322052)]
# [[[91.0, 84.0], [115.0, 84.0], [115.0, 96.0], [91.0, 96.0]], ('UCI', 0.9108545184135437)]

# Example of chart element lable and location
# chart_title 0, conf<0.7239867448806763>: tensor([154.9298,  13.5539, 484.4514,  34.9978])
# mark_label 0, conf<0.7774506211280823>: tensor([512.3945, 155.1012, 632.9523, 169.9546])
# mark_label 1, conf<0.6870139241218567>: tensor([441.2546,  76.7057, 565.6805,  91.9275])
# mark_label 2, conf<0.6422735452651978>: tensor([ 93.2445, 320.4388, 113.9430, 331.5377])
# mark_label 3, conf<0.6197637915611267>: tensor([ 92.3864,  83.3271, 117.7661,  94.7353])
# mark_label 4, conf<0.5134041905403137>: tensor([ 92.8448, 201.4870, 108.1105, 212.4897])
# y_axis_area 0, conf<0.9712439775466919>: tensor([ 15.6435,  46.1464,  79.0129, 379.8453])
# x_axis_area 0, conf<0.9736053347587585>: tensor([ 69.9538, 371.6832, 636.4146, 451.0058])

# Match OCR result with chart element by computing the IOU of bounding boxes
# If the IOU is greater than a threshold, we consider the OCR result is matched with the chart element
# give me the algorithm
# The algorithm is as follows:
# 1. For each OCR result, we calculate the IOU with each chart element
# 2. If the IOU is greater than a threshold, we consider the OCR result is matched with the chart element
# 3. We store the matched OCR result and chart element in a dictionary
# 4. We return the dictionary as the final result

# The function to calculate the IOU of two bounding boxes
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    iou = intersection / union
    return iou

# the function is use the bbox instance from chart dete to find the most matched bbox from ocr result
def find_matched_bbox(ocr_result, chart_element_result):
    matched_result = {}
    for chart_element, chart_element_bboxes in chart_element_result.items():
        matched_result[chart_element] = []
        for chart_element_bbox in chart_element_bboxes:
            max_iou = 0
            matched_ocr_result = None
            for ocr_bbox in ocr_result:
                iou = calculate_iou(ocr_bbox[0], chart_element_bbox)
                if iou > max_iou:
                    max_iou = iou
                    matched_ocr_result = ocr_bbox
            if max_iou > 0.5:
                matched_result[chart_element].append(matched_ocr_result)
    return matched_result


# appoarch 2: iterate the ocr result and find the most matched bbox from chart element result to assign the elemebt label
def find_matched_bbox_ocr2chartdete(ocr_result, chart_element_result):
    matched_result = {}
    for ocr_bbox in ocr_result:
        max_iou = 0
        matched_chart_element = None
        for chart_element, chart_element_bboxes in chart_element_result.items():
            for chart_element_bbox in chart_element_bboxes:
                iou = calculate_iou(ocr_bbox[0], chart_element_bbox)
                if iou > max_iou:
                    max_iou = iou
                    matched_chart_element = chart_element
        if max_iou > 0.5:
            if matched_chart_element not in matched_result:
                matched_result[matched_chart_element] = []
            matched_result[matched_chart_element].append(ocr_bbox)
    return matched_result


def process_ocr(img_path):
    # Initialize OCR model
    ocr_model = PaddleOCR(use_angle_cls=True, lang='en')

    # OCR model processes the image
    ocr_output = ocr_model.ocr(img_path, cls=True)
    for idx in range(len(ocr_output)):
        res = ocr_output[idx]
        for line in res:
            print(line)
    return ocr_output


def process_chart_dete(img_path):
    img = mmcv.imread(img_path)
    bbox_result_dict = {}

    chart_element_result = inference_detector(chart_dete_model, img_path)
    chart_dete_model.show_result(img_path, chart_element_result, out_file='./test_chart.jpg')
    
    for i, cls in enumerate(zip(CLASSES, chart_element_result)):
        class_name, class_result = cls
        # breakpoint()
        bbox_result_dict[class_name] = []
        if len(class_result) == 0:
            continue

        for j, bbox in enumerate(class_result):
            conf = bbox[-1]
            if conf < 0.5: continue
            bbox = bbox[:-1].tolist()
            bbox_result_dict[class_name].append(bbox)
            # bbox = bbox_cxcywh_to_xyxy(bbox)
            # bbox_result_dict[class_name][j] = bbox.numpy().tolist()
            # there may be multiple instances of the same class
            print(f'{class_name} {j}, conf<{conf}>: {bbox}')
            # save the bounding box coordinates and the confidence to a file
            # get the image patch of the bounding box and save it as a jpg file in a folder
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # read image 
            patch = img[y1:y2, x1:x2]
            if os.path.exists(f'./test_chart_patch/{class_name}'):
                os.makedirs(f'./test_chart_patch/{class_name}', exist_ok=True)
            mmcv.imwrite(patch, f'./test_chart_patch/{class_name}/{j}.jpg')
    return bbox_result_dict


# result from ocr [[156.0, 15.0], [485.0, 15.0], [485.0, 31.0], [156.0, 31.0]]
# result from chart dete [154.9298,  13.5539, 484.4514,  34.9978]
# these two result are matched, help me comvert the result of ocr to the same format as the result of chart dete
def convert_ocr_result(ocr_result):
    return [[covert_bbox_format(ele[0]), ele[1][0]] for ele in ocr_result[0]]

def covert_bbox_format(ocr_bbox):
    x1, y1 = ocr_bbox[0][0], ocr_bbox[0][1]
    x3, y3 = ocr_bbox[2][0], ocr_bbox[2][1]
    return [x1, y1, x3, y3]


def generate_auxiliary_prompt(chart_info):
    prompt_lines = [
        "[Auxiliary Prompt for Vision Language Model]:\n",
        "The following information pertains to a control chart with stability analysis. Please use this information to better understand the chart and answer any related questions:\n"
    ]
    
    # Standard keys to be included in specific order
    standard_keys = ['chart_title', 'x_title', 'y_title', 'xlabel', 'ylabel', 'mark_label']
    labels_mapping = {
        'chart_title': '[Chart Title]:',
        'x_title': '[X-Axis Title]:',
        'y_title': '[Y-Axis Title]:',
        'xlabel': '[X-Axis Label (Values)]:',
        'ylabel': '[Y-Axis Label (Values)]:',
        'mark_label': '[Mark Labels]:'
    }

    for key in standard_keys:
        if key in chart_info:
            if key in ['ylabel', 'xlabel', 'mark_label']:
                # Join list items into a single string for these keys
                value = ', '.join(map(str, chart_info[key]))
            else:
                value = chart_info[key] if not isinstance(chart_info[key], list) else chart_info[key][0]
            prompt_lines.append(f"{labels_mapping[key]}\n- {value}\n")
    
    # Handle any additional keys not included in the standard keys
    additional_keys = set(chart_info.keys()) - set(standard_keys)
    for key in additional_keys:
        value = chart_info[key]
        if isinstance(value, list):
            value = ', '.join(map(str, value))
        prompt_lines.append(f"[{key.replace('_', ' ').title()}]:\n- {value}\n")

    # Add concluding questions
    prompt_lines.extend([
        "\nPlease analyze the provided chart information and answer the following questions:",
        "1. What trends or patterns can be observed over the specified time period?",
        "2. Are there any significant deviations or unstable points indicated on the chart?",
        "3. How do the values compare to the control limits (UCI, CL, LCL)?",
        "4. What conclusions can be drawn about the stability of the process?"
    ])
    
    return '\n'.join(prompt_lines).strip()


def main(img_path):
    # Run OCR on the image
    ocr_results = process_ocr(img_path)
    ocr_results = convert_ocr_result(ocr_results)

    # Run Chart Detection on the image
    bbox_result_dict = process_chart_dete(img_path)

    matched_results = find_matched_bbox_ocr2chartdete(ocr_results, bbox_result_dict)

    # create a dictionary only store the key and value list
    matched_results = {k: [v[1] for v in vs] for k, vs in matched_results.items()}
    # pretty print the matched results
    print(json.dumps(matched_results, indent=4))

    prompt = generate_auxiliary_prompt(matched_results)
    print(prompt)

    # save the key-value pairs to a json file
    with open('matched_results.json', 'w') as f:
        json.dump(matched_results, f)

if __name__ == '__main__':
    img_path = 'normal_chart.png'
    main(img_path)
