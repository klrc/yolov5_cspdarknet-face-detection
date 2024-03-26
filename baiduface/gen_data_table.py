if __name__ == "__main__":
    label_arrangement = [
        ["class_id", 1],
        ["location", 4],
        ["landmarks", 10],
        ["eye_status", 2],
        ["occlusions", 7],
        ["quality_blur", 1],
        ["quality_illumination", 1],
        ["quality_completeness", 1],
        ["age", 1],
        ["gender", 1],
        ["glasses", 1],
        ["mask", 1],
    ]
    label_address = {}
    start_address = 0
    for k, v in label_arrangement:
        label_address[k] = start_address
        start_address += v

    print(label_address)
    with open("../data_table.py", "w") as f:
        f.write(f"DATASET_NUM_DIMS = {start_address}\n")
        f.write("\n")
        f.write(f"IDX_CLASS_ID = {label_address['class_id']}\n")
        f.write(f"IDX_LOCATION = {label_address['location']}\n")
        f.write(f"IDX_LANDMARKS = {label_address['landmarks']}\n")
        f.write(f"IDX_EYE_STATUS = {label_address['eye_status']}\n")
        f.write(f"IDX_OCCLUSIONS = {label_address['occlusions']}\n")
        f.write(f"IDX_BLUR = {label_address['quality_blur']}\n")
        f.write(f"IDX_ILLUMINATION = {label_address['quality_illumination']}\n")
        f.write(f"IDX_COMPLETENESS = {label_address['quality_completeness']}\n")
        f.write(f"IDX_AGE = {label_address['age']}\n")
        f.write(f"IDX_GENDER = {label_address['gender']}\n")
        f.write(f"IDX_GLASSES = {label_address['glasses']}\n")
        f.write(f"IDX_MASK = {label_address['mask']}\n")
