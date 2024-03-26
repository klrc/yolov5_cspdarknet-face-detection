import torch
import numpy as np
import cv2
import os
import imageio
from multiprocessing import Process, Queue
import time
from typing import Iterable
import mss
import pyautogui as pag
import math
import argparse
from pynput import keyboard
from skimage import transform as T

from dataset import RelativeHumanDataset
from canvas import Canvas, im2tensor, letterbox_padding, tensor2im
from general import Timer
from model import yolov8n_facecap_v2


# demo options
RENDER_OPTIONS = [
    "landmarks",
    "landmarks_half",
    "age",
    "eye_status",
    "gender",
    "occlusion_eyes",
    "occlusion_nose",
    "occlusion_mouth",
    "occlusion_cheek",
    "occlusion_chin_contour",
    "quality",
    "glasses",
    "mask",
    "snapshot",
    "snapshot_with_affine",
]
MAX_SNAPSHOTS = 10
SNAPSHOT_IDX = 0
STANDARD_LANDMARKS = np.array(
    [
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041],
    ],
    dtype=np.float32,
)  # standard for 96 * 112


def pad(image: np.ndarray, pt, pl, pb, pr, padding_value):
    image = np.pad(image, ((pt, pb), (pl, pr), (0, 0)), "constant", constant_values=padding_value)
    return image


def scale(image: np.ndarray, ratio):
    height, width, _ = image.shape
    image = cv2.resize(image, (int(ratio * width), int(ratio * height)))
    return image


def scale_and_pad(image: np.ndarray, dsize, padding_value=114):
    w, h = dsize
    height, width, _ = image.shape
    ratio = min(w / width, h / height)
    image = scale(image, ratio)
    height, width, _ = image.shape
    pt, pl = (h - height) // 2, (w - width) // 2
    pb, pr = h - height - pt, w - width - pl
    image = pad(image, pt, pl, pb, pr, padding_value)
    return image


def camera_stream(size=None, exit_key="q", horizontal_flip=True):
    if isinstance(size, int):
        size = (size, size)

    cap = cv2.VideoCapture(0)
    assert cap.isOpened()

    # Warm-up camera
    for _ in range(3):
        _ = cap.read()

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord(exit_key):
            cap.release()
            break

        success, frame = cap.read()
        if not success:
            cap.release()
            raise Exception("camera connection lost")

        if size:
            frame = scale_and_pad(frame, size)
        if horizontal_flip:
            frame = cv2.flip(frame, 1)
        yield frame


def gif_stream(file_path, size=None, exit_key="q", color_format=cv2.COLOR_RGB2BGR):
    if isinstance(size, int):
        size = (size, size)

    cap = imageio.mimread(file_path)
    for frame in cap:
        key = cv2.waitKey(1) & 0xFF
        if key == ord(exit_key):
            break

        if size:
            frame = scale_and_pad(frame, size)
        if color_format:
            frame = cv2.cvtColor(frame, color_format)
        yield frame


def h264_stream(file_path, size=None, exit_key="q", color_format=None, show_cap_fps=False):
    if isinstance(size, int):
        size = (size, size)

    cap = cv2.VideoCapture(file_path)
    if show_cap_fps:
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        print("video fps:", fps)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord(exit_key):
            cap.release()
            break

        ret, frame = cap.read()
        if not ret:
            break
        if size:
            frame = scale_and_pad(frame, size)
        if color_format:
            frame = cv2.cvtColor(frame, color_format)
        yield frame


def realtime_sampled_work_process(skipped: Queue, stop_flag: Queue, fps: int):
    """
    The function that runs as a separate process and adds frames to the buffer.

    Args:
    - skipped (Queue): skipped frames.
    - fps (int): The frames per second.
    """
    while stop_flag.empty():
        if skipped.full():
            raise NotImplementedError("too many skipped frames")
        time.sleep(1 / fps)  # simulated fps
        skipped.put(1)


def realtime_sampled(frames: Iterable, exit_key="q", fps=24):
    """
    A function to create a realtime video stream from an iterable of frames.

    Args:
    - frames (Iterable): The iterable of frames to stream.
    - fps (int): The frames per second (default=24).
    - buffer_size (int): The maximum size of the buffer (default=2).

    Returns:
    - A generator that yields frames in real-time.
    """
    skipped = Queue(maxsize=60)
    stop_flag = Queue(maxsize=1)

    # Create a virtual video stream
    proc = Process(target=realtime_sampled_work_process, args=(skipped, stop_flag, fps))
    proc.daemon = True
    proc.start()

    for frame in frames:
        key = cv2.waitKey(1) & 0xFF
        if key == ord(exit_key):
            break

        _ = skipped.get()
        if skipped.empty():
            yield frame

    stop_flag.put(1)


def screenshot_stream(size=416, exit_key="q", color_format=cv2.COLOR_RGBA2RGB):
    if isinstance(size, int):
        size = (size, size)
    w, h = size

    capture_range = {"top": 0, "left": 0, "width": w, "height": h}
    cap = mss.mss()

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord(exit_key):
            break
        x, y = pag.position()  # 返回鼠标的坐标
        capture_range["top"] = y - capture_range["height"] // 2
        capture_range["left"] = x - capture_range["width"] // 2
        frame = cap.grab(capture_range)
        frame = np.array(frame)
        frame = cv2.resize(frame, (w, h))
        if color_format:
            frame = cv2.cvtColor(frame, color_format)
        yield frame


def yuv420_stream(file_path, yuv_size=(1920, 1080), size=None, exit_key="q", color_format=cv2.COLOR_YUV2BGR_I420):
    yuv_w, yuv_h = yuv_size
    file_size = os.path.getsize(file_path)
    max_frame = file_size // (yuv_w * yuv_h * 3 // 2) - 1

    cur_frame = 0
    with open(file_path, "rb") as probe:
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord(exit_key):
                break
            cur_frame += 1
            if cur_frame > max_frame:
                break
            yuv = np.frombuffer(probe.read(yuv_w * yuv_h * 3 // 2), dtype=np.uint8).reshape((yuv_h * 3 // 2, yuv_w))
            if color_format:
                yuv = cv2.cvtColor(yuv, color_format)
            if size:
                yuv = scale_and_pad(yuv, size)
            yield yuv


def get_score(conf, blur, illumination, completeness):
    return conf * 0.1 - math.tanh(blur * 5) + illumination * 0.1 + completeness * 0.8


def get_snapshot(frame, det, conf, pt1, pt2):
    blur, illumination, completeness = [float(x) for x in det[23:26]]
    score = get_score(conf, blur, illumination, completeness)
    if score > 0:
        face = frame[:, :, int(pt1[1]) : int(pt2[1]), int(pt1[0]) : int(pt2[0])]
        face = tensor2im(face)
        if face.shape[1] > 0:
            return face, score
    return None, score


def display_snapshot(snapshot, affine=False, landmarks=None):
    global SNAPSHOT_IDX
    if snapshot is not None:
        if affine and landmarks is not None:
            tform = T.SimilarityTransform()
            tform.estimate(landmarks, STANDARD_LANDMARKS)
            M = tform.params[0:2, :]
            snapshot = cv2.warpAffine(snapshot, M, (96, 112), borderValue=0.0)
        else:
            fh, fw, _ = snapshot.shape
            ratio = 96 / fw
            snapshot = cv2.resize(snapshot, (int(ratio * fw), int(ratio * fh)))
        SNAPSHOT_IDX = SNAPSHOT_IDX % MAX_SNAPSHOTS
        cv2.imshow(str(SNAPSHOT_IDX), snapshot)
        cv2.moveWindow(str(SNAPSHOT_IDX), SNAPSHOT_IDX * 96, 0)
        SNAPSHOT_IDX += 1


def switch_option():
    global RENDER_OPTIONS
    RENDER_OPTIONS.append(RENDER_OPTIONS.pop(0))
    print("[INFO] mode switch to", RENDER_OPTIONS[0])


def switch_detach():
    global IS_DETACHED
    IS_DETACHED = not IS_DETACHED
    if IS_DETACHED:
        print("[INFO] video stream detached. ---<  <---")
    else:
        print("[INFO] video stream attached. ----<<----")


def on_press(key):
    if key == keyboard.Key.tab:
        switch_option()
    elif key == keyboard.Key.esc:
        switch_detach()


def run_demo(device, weight, conf_thr, iou_thr, size, camera, screenshot, h264, save_path):
    class_names = RelativeHumanDataset.class_names
    model = yolov8n_facecap_v2()

    # load pretrained weights
    state = torch.load(weight, map_location=device)
    model.load_state_dict(state)
    model.to(device)

    # set nms args
    model.head.nms.conf_threshold = conf_thr
    model.head.nms.iou_threshold = iou_thr

    # you can use any loader from dataloader
    if camera:
        test_data = camera_stream(size)
    elif screenshot:
        test_data = screenshot_stream(size)
    elif h264:
        test_data = h264_stream(h264, size)
    else:
        raise Exception("At least one of the stream mode must be specified, see -help")

    canvas = Canvas()
    time_rec = Timer(10)
    video_writer = None  # use video_writer=VideoSaver(save_path) to save

    # set eval mode
    model.eval()

    # add hot key
    global IS_DETACHED
    IS_DETACHED = False
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # read test data
    try:
        for frame in test_data:
            # preprocess
            if not IS_DETACHED:
                frame = im2tensor(frame)
                frame = letterbox_padding(frame).unsqueeze(0)

            # inference
            with time_rec.record("core"):
                with torch.no_grad():
                    out = model(frame.to(device))

            # Load raw image & parse outputs
            canvas.load(frame)
            for out_batch_i in out:
                if len(out_batch_i) > 0:
                    for det in out_batch_i:
                        # list of detections, on (n,6) tensor per image [xyxy, conf, cls]
                        pt1, pt2, conf, cls = det[:2], det[2:4], det[4], int(det[5])
                        title = f"{str(cls) if not class_names else class_names[cls]}: {conf:.2f}"
                        landmarks = [float(x) for x in det[6:16]]

                        eye_status = [f"{float(x):.1f}" for x in det[16:18]]
                        occlusions = [f"{float(x):.1f}" for x in det[18:25]]
                        qualities = [f"{float(x):.2f}" for x in det[25:28]]

                        age = int(float(det[28]) * 200 - 50)
                        gender = f"{float(det[29]):.2f}"
                        wear_glasses = f"{float(det[30]):.2f}"
                        wear_mask = f"{float(det[31]):.2f}"

                        if RENDER_OPTIONS[0] == "landmarks":  # draw landmarks
                            for li in range(5):
                                x = pt1[0] + landmarks[li * 2] * (pt2[0] - pt1[0])
                                y = pt1[1] + landmarks[li * 2 + 1] * (pt2[1] - pt1[1])
                                canvas.draw_point((x, y))

                        if RENDER_OPTIONS[0] == "landmarks_half":  # draw landmarks
                            for li in [0, 2, 3]:
                                x = pt1[0] + landmarks[li * 2] * (pt2[0] - pt1[0])
                                y = pt1[1] + landmarks[li * 2 + 1] * (pt2[1] - pt1[1])
                                canvas.draw_point((x, y))

                        if RENDER_OPTIONS[0] == "quality":  # draw qualities
                            title = f"blur={qualities[0]}\n"
                            title += f"illumination={qualities[1]}\n"
                            title += f"completeness={qualities[2]}\n"

                        if RENDER_OPTIONS[0] == "age":  # draw age
                            title = f"age={age}\n"

                        if RENDER_OPTIONS[0] == "eye_status":  # draw eye_status
                            title = f"eye_status={eye_status}\n"

                        if RENDER_OPTIONS[0] == "gender":  # draw gender
                            title = f"gender={gender}\n"

                        if RENDER_OPTIONS[0] == "glasses":  # draw glasses
                            title = f"glasses={wear_glasses}\n"

                        if RENDER_OPTIONS[0] == "mask":  # draw mask
                            title = f"mask={wear_mask}\n"

                        if RENDER_OPTIONS[0] == "occlusion_eyes":  # draw occlusions
                            title = f"occlusion_left_eye={occlusions[0]}\n"
                            title += f"occlusion_right_eye={occlusions[1]}\n"

                        if RENDER_OPTIONS[0] == "occlusion_nose":  # draw occlusions
                            title = f"occlusion_nose={occlusions[2]}\n"

                        if RENDER_OPTIONS[0] == "occlusion_mouth":  # draw occlusions
                            title = f"occlusion_mouth={occlusions[3]}\n"

                        if RENDER_OPTIONS[0] == "occlusion_cheek":  # draw occlusions
                            title = f"occlusion_left_cheek={occlusions[4]}\n"
                            title += f"occlusion_right_cheek={occlusions[5]}\n"

                        if RENDER_OPTIONS[0] == "occlusion_chin_contour":  # draw occlusions
                            title = f"occlusion_chin_contour={occlusions[6]}\n"

                        if RENDER_OPTIONS[0] == "snapshot":
                            snapshot, score = get_snapshot(frame, det, conf, pt1, pt2)
                            display_snapshot(snapshot)
                            title = f"{score:.2f}"

                        if RENDER_OPTIONS[0] == "snapshot_with_affine":
                            snapshot, score = get_snapshot(frame, det, conf, pt1, pt2)
                            snapshot_landmarks = np.array(landmarks).reshape(-1, 2) - np.array(pt1.to("cpu"))
                            display_snapshot(snapshot, affine=True, landmarks=snapshot_landmarks)
                            title = f"{score:.2f}"

                        color = canvas.color(cls)
                        canvas.draw_box(pt1, pt2, alpha=0.4, thickness=-1, color=color)
                        canvas.draw_box(pt1, pt2, color=color, title=title)

            # Use external time info
            for i, (title, value) in enumerate(time_rec.step().items()):
                canvas.draw_text(f"{title}={value}", (0, 12 * i), color=canvas.color(title, light_theme=False))
            canvas.show("demo", wait_key=1)

            # Save outputs
            if save_path:
                if video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"XVID")
                    h, w, _ = canvas.image().shape
                    video_writer = cv2.VideoWriter(save_path, fourcc, 24, (w, h))
                video_writer.write(canvas.image())
    except Exception as e:
        raise e
    finally:
        video_writer.release()
        listener.stop()


if __name__ == "__main__":
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="YOLOv8 Face Detection Demo Configuration")

    # 添加命令行参数
    parser.add_argument("--device", type=str, default="cpu", help="指定设备: cpu/mps/cuda")
    parser.add_argument("--weight", type=str, help="模型权重文件的路径")
    parser.add_argument("--conf_thr", type=float, default=0.25, help="置信度阈值，默认为 0.25")
    parser.add_argument("--iou_thr", type=float, default=0.25, help="IoU 阈值，默认为 0.45")
    parser.add_argument("--size", type=int, default=640, help="图像尺寸，默认为 640")
    parser.add_argument("--camera", action="store_true", help="摄像头采集模式")
    parser.add_argument("--screenshot", action="store_true", help="截屏采集模式")
    parser.add_argument("--h264", type=str, default=None, help="h264视频模式")
    parser.add_argument("--save_path", type=str, default=None, help="保存路径, 默认为 None")

    # 解析命令行参数
    args = parser.parse_args()
    run_demo(**vars(args))
