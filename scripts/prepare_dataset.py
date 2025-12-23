import cv2
from pathlib import Path


def crop_yolo_dataset(
    base_dir: Path,
    split: str,
    output_base: Path,
    classes: list[str],
) -> None:
    """
    Convert YOLO formatted dataset into cropped classification dataset.
    """
    img_dir = base_dir / split / "images"
    label_dir = base_dir / split / "labels"
    output_dir = output_base / split

    if not img_dir.exists():
        print(f"Hata: {img_dir} bulunamadı!")
        return

    print(f"İşleniyor: {split} seti...")

    for img_path in img_dir.glob("*.jpg"):
        label_path = label_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w, _ = img.shape

        with label_path.open() as f:
            for i, line in enumerate(f):
                parts = line.split()
                if not parts:
                    continue

                cls_id = int(parts[0])
                x_center, y_center, bw, bh = map(float, parts[1:5])

                x1 = int((x_center - bw / 2) * w)
                y1 = int((y_center - bh / 2) * h)
                x2 = int((x_center + bw / 2) * w)
                y2 = int((y_center + bh / 2) * h)

                crop = img[
                    max(0, y1):min(h, y2),
                    max(0, x1):min(w, x2),
                ]

                if crop.size == 0:
                    continue

                class_name = classes[cls_id]
                save_dir = output_dir / class_name
                save_dir.mkdir(parents=True, exist_ok=True)

                file_name = f"{img_path.stem}_crop_{i}.jpg"
                cv2.imwrite(str(save_dir / file_name), crop)


def main() -> None:
    classes = ["adult", "child"]

    raw_data_path = Path(
        r"C:\Users\zeyne\Desktop\pool_project\pool_guard_data\raw\Child-Adult Classification.v6i.yolov8"
    )
    processed_path = Path(
        r"C:\Users\zeyne\Desktop\pool_project\pool_guard_data\classifier"
    )

    for split in ["train", "valid", "test"]:
        crop_yolo_dataset(raw_data_path, split, processed_path, classes)

    print(f"\nİşlem tamam! Kırpılmış resimler burada: {processed_path}")


if __name__ == "__main__":
    main()
