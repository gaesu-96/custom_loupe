import os
from datasets import Dataset, DatasetDict
from datasets import Features, Image, Value
from typing import List, Optional


def load_images_from_dir(directory: str) -> List[str]:
    """특정 폴더 내의 이미지 파일 경로를 전부 불러옵니다."""
    return [
        os.path.join(directory, fname)
        for fname in os.listdir(directory)
        if fname.lower().endswith(("jpg", "jpeg", "png", "tif"))
    ]


def create_split(root_dir: str, split: str) -> Optional[Dataset]:
    """train / valid / test 각각의 split에 대해 Dataset 객체를 만듭니다."""
    fake_dir = os.path.join(root_dir, split, "fake")
    real_dir = os.path.join(root_dir, split, "real")

    # 폴더가 존재하지 않으면 None 리턴
    if all(not os.path.isdir(p) for p in [fake_dir, real_dir]):
        return None

    print(f"Split: {split},", end=" ")

    fake_images, real_images = [], []
    if os.path.isdir(fake_dir):
        fake_images = load_images_from_dir(fake_dir)
        print(f"Fake images: {len(fake_images)}", end="")
    if os.path.isdir(real_dir):
        real_images = load_images_from_dir(real_dir)
        print(f", Real images: {len(real_images)}", end="")
    print()

    # Dataset 객체 생성
    return Dataset.from_dict(
        {
            "path": fake_images + real_images,
            "image": fake_images + real_images,
        },
        features=Features(
            {"path": Value(dtype="string"), "image": Image()}
        ),
    )


def create_dataset(root_dir: str) -> DatasetDict:
    """DatasetDict 형태로 train/valid/test 묶음 생성"""
    return DatasetDict(
        {
            split: d
            for split in ["train", "valid", "test"]
            if (d := create_split(root_dir, split)) is not None
        }
    )


if __name__ == "__main__":
    # 데이터셋 경로와 저장 경로 설정
    root_dir = "../drive/Othercomputers/Mac/gdrive/dataset"
    save_dir = "./data_output"
    os.makedirs(save_dir, exist_ok=True)

    # parquet 변환 실행
    dataset = create_dataset(root_dir)
    for split in dataset:
        parquet_path = os.path.join(save_dir, f"{split}.parquet")
        dataset[split].to_parquet(parquet_path)
        print(f"✅ Saved {split} split to {parquet_path}")
