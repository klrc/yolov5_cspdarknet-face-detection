import sys
import cv2
import random

sys.path.append(".")

from dataset import RelativeHumanDataset  # noqa:E402
from canvas import Canvas  # noqa:E402


def visualize(dataset: RelativeHumanDataset):
    canvas = Canvas()
    while True:
        i = random.randint(0, len(dataset) - 1)
        image, labels = dataset.__getitem__(i)
        _, H, W = image.shape
        canvas.load(image)
        labels = labels[:, 1:].numpy()
        # Render labels for object detection datasets
        for j, lb in enumerate(labels):
            _, cx, cy, w, h = int(lb[0]), lb[1], lb[2], lb[3], lb[4]  # cls, cx, cy, w, h (dataset format)
            x1 = int((cx - 0.5 * w) * W)
            y1 = int((cy - 0.5 * h) * H)
            x2 = int((cx + 0.5 * w) * W)
            y2 = int((cy + 0.5 * h) * H)
            print(f"{lb} ({i}-{j})")
            canvas.draw_box((x1, y1), (x2, y2), title=f"{i}-{j}")

            # draw landmarks
            landmarks = lb[5:15]
            for li in range(5):
                x, y = x1 + landmarks[li * 2] * (x2 - x1), y1 + landmarks[li * 2 + 1] * (y2 - y1)
                canvas.draw_point((x, y), thickness=1)

            # draw attributes
            extra_attributes = [f"{float(x):.2f}" for x in lb[15:]]
            info = f"eye_status = {extra_attributes[:2]}\n"
            info += f"occlusions = {extra_attributes[2:9]}\n"
            info += f"quality = {extra_attributes[9:12]}\n"
            info += f"age = {extra_attributes[12]}\n"
            info += f"gender = {extra_attributes[13]}\n"
            info += f"glasses = {extra_attributes[14]}\n"
            info += f"mask = {extra_attributes[15]}\n"
            print(info)

        canvas.show(wait_key=1)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break


if __name__ == "__main__":
    dataset = RelativeHumanDataset(
        training=True,
        dataset_path="/Volumes/ASM236X/datasets/relative_human_mixed_0124/images/train",
        image_size=640,
        batch_size=3,
        stride=32,
    )
    visualize(dataset)
