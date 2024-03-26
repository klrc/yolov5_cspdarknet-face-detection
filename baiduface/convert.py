import os
import json
import numpy as np
import cv2
from tqdm import tqdm


def rotate(p, origin=(0, 0), degrees=0):
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T - o.T) + o.T).T)


def clamp(x, min_value=0, max_value=1):
    return min(max(x, min_value), max_value)


def get_image_size(fp):
    im = cv2.imread(fp)
    return im.shape


def parse_label(lines):
    results = []
    for line in lines:
        data = [float(x) for x in line.strip().split(" ")]
        c, _, _, w, h = data
        if w == 0 or h == 0 or c != 0:
            print("[WARNING] Corrupted data:", data)
            continue
        results.append(data)
    return results


def location_to_xywh(location, image_w, image_h):
    x1 = location.get("left")
    y1 = location.get("top")
    w = location.get("width")
    h = location.get("height")
    degrees = location.get("rotation")
    x2 = x1 + w
    y2 = y1 + h

    points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    origin = [(x1, y1)]
    new_points = rotate(points, origin, degrees)

    _x1 = np.min(new_points[:, 0])
    _y1 = np.min(new_points[:, 1])
    _x2 = np.max(new_points[:, 0])
    _y2 = np.max(new_points[:, 1])

    cx = (_x1 + _x2) / 2 / image_w
    cy = (_y1 + _y2) / 2 / image_h
    w = (_x2 - _x1) / image_w
    h = (_y2 - _y1) / image_h
    return [cx, cy, w, h]


def gender_to_value(gender):
    gender_type = gender["type"]
    # gender_prob = gender["probability"]
    if gender_type == "male":
        return 1
        # return gender_prob
    elif gender_type == "female":
        return 0
        # return 1 - gender_prob
    else:
        raise Exception(f"Invalid gender type: {gender_type}")


def glasses_to_value(glasses):
    glasses_type = glasses["type"]
    # glasses_prob = glasses["probability"]
    if glasses_type == "none":
        return 0
        # return 1 - glasses_prob
    else:
        return 1
        # return glasses_prob


def landmark72_to_landmark5(landmark72, image_w, image_h):
    # https://ai.bdstatic.com/file/52BC00FFD4754A6298D977EDAD033DA0
    left_eye = landmark72[21]
    right_eye = landmark72[38]
    nose_tip = landmark72[57]
    mouth_left = landmark72[58]
    mouth_right = landmark72[62]
    outputs = []
    for point in (left_eye, right_eye, nose_tip, mouth_left, mouth_right):
        x = point.get("x") / image_w
        y = point.get("y") / image_h
        outputs.append(clamp(x))
        outputs.append(clamp(y))
    return outputs


def quality_to_list(quality):
    occlusion = quality.get("occlusion")
    blur = quality.get("blur")
    illumination = quality.get("illumination") / 255.0
    completeness = quality.get("completeness")
    return [x for x in occlusion.values()] + [blur, illumination, completeness]


def eye_status_to_list(eye_status):
    left_eye = eye_status.get("left_eye")
    right_eye = eye_status.get("right_eye")
    return [left_eye, right_eye]


def mask_to_value(mask):
    mask_type = mask.get("type")
    # mask_prob = mask.get("probability")
    if mask_type == 0:
        return 0
        # return 1 - mask_prob
    else:
        return 1
        # return mask_prob


def age_to_value(age):
    # 0~100 -> 0~1 -> 0.25~0.75
    return clamp(age / 100 * 0.5 + 0.25)


def parse_json(data, image_w, image_h):
    ret = []
    json_result = data.get("result")
    if json_result is not None:
        json_face_list = json_result.get("face_list")
        for json_face in json_face_list:
            face = {}
            face["location"] = location_to_xywh(json_face["location"], image_w, image_h)
            face["landmarks"] = landmark72_to_landmark5(json_face["landmark72"], image_w, image_h)
            face["eye_status"] = eye_status_to_list(json_face["eye_status"])
            face["quality"] = quality_to_list(json_face["quality"])
            face["age"] = [age_to_value(json_face["age"])]
            face["gender"] = [gender_to_value(json_face["gender"])]
            face["glasses"] = [glasses_to_value(json_face["glasses"])]
            face["mask"] = [mask_to_value(json_face["mask"])]
            ret.append(face)
    return ret


def convert_label(image_path, json_path, save_path):
    # Get image size
    image_h, image_w, _ = get_image_size(image_path)

    # Get json data
    with open(json_path, "r") as f:
        jsons = json.load(f)
    json_data = parse_json(jsons, image_w, image_h)

    # Append json data as txt label
    labels = []
    for json_item in json_data:
        label = [0]
        label.extend(json_item["location"])
        label.extend(json_item["landmarks"])
        label.extend(json_item["eye_status"])
        label.extend(json_item["quality"])
        label.extend(json_item["age"])
        label.extend(json_item["gender"])
        label.extend(json_item["glasses"])
        label.extend(json_item["mask"])
        label = [clamp(x) for x in label]
        labels.append(label)

    # Write file
    with open(save_path, "w") as f:
        for label in labels:
            line = " ".join([str(x) for x in label])
            f.write(line + "\n")


if __name__ == "__main__":
    image_root = "/Users/sh/SourceCode/relative_human_mixed_0124/images"
    json_root = "/Users/sh/SourceCode/relative_human_mixed_0124/baidu_face_v3"
    save_root = "/Users/sh/SourceCode/relative_human_mixed_0124/labels"

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    for fname in tqdm(os.listdir(json_root)):
        if not fname.startswith("._"):
            fid = fname.split(".")[0]
            image_path = f"{image_root}/{fid}.jpg"
            json_path = f"{json_root}/{fid}.json"
            save_path = f"{save_root}/{fid}.txt"

            paths = [image_path, json_path]
            if all([os.path.exists(x) for x in paths]):
                convert_label(image_path, json_path, save_path)
