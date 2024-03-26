import os
import random
from tqdm import tqdm

if __name__ == "__main__":
    images_root = "/Users/sh/SourceCode/relative_human_mixed_old_0124/images"
    labels_root = "/Users/sh/SourceCode/relative_human_mixed_old_0124/labels"

    data = []

    for image_base in tqdm(os.listdir(images_root)):
        imid = image_base.split(".")[0]
        label_base = f"{imid}.txt"
        image_path = os.path.join(images_root, image_base)
        label_path = os.path.join(labels_root, label_base)
        data.append([image_path, label_path])

    random.shuffle(data)

    train_data = data[int(len(data) * 0.05) :]
    val_data = data[: int(len(data) * 0.05)]

    for save_image_root, save_data in [
        ["/Users/sh/SourceCode/relative_human_mixed_0124/images/train", train_data],
        ["/Users/sh/SourceCode/relative_human_mixed_0124/images/val", val_data],
    ]:
        save_label_root = save_image_root.replace("/images/", "/labels/")

        if not os.path.exists(save_image_root):
            os.makedirs(save_image_root)
        if not os.path.exists(save_label_root):
            os.makedirs(save_label_root)

        for image_path, label_path in tqdm(save_data):
            image_base = os.path.basename(image_path)
            label_base = os.path.basename(label_path)
            save_image_path = f"{save_image_root}/{image_base}"
            save_label_path = f"{save_label_root}/{label_base}"
            os.system(f"mv {image_path} {save_image_path}")
            os.system(f"mv {label_path} {save_label_path}")
