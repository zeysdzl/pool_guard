from pathlib import Path

def remap_labels(base_dir: Path):
    for split in ["train", "valid", "test"]:
        label_dir = base_dir / split / "labels"
        for label_file in label_dir.glob("*.txt"):
            new_lines = []
            for line in label_file.read_text().splitlines():
                parts = line.split()
                if not parts:
                    continue
                # class id -> 0 (person)
                new_line = "0 " + " ".join(parts[1:])
                new_lines.append(new_line)
            label_file.write_text("\n".join(new_lines))

if __name__ == "__main__":
    base = Path(
        r"C:\Users\zeyne\Desktop\pool_project\pool_guard_data\raw\Child-Adult-Classification.v6i.yolov8-v2"
    )
    remap_labels(base)
