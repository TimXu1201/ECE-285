import random
import shutil
from pathlib import Path
from tqdm import tqdm


def build_datasets():
    # Paths
    real_normal_dir = Path("./chest_xray/train/NORMAL")
    real_pneumonia_dir = Path("./chest_xray/train/PNEUMONIA")

    wgan_fake_dir = Path("./output_wgan/synthetic_pneumonia_fakes_best")
    ddpm_fake_dir = Path("./output_ddpm/ddpm_pneumonia_fakes_best")

    exp_root = Path("./Experiment_Datasets")

    # Load file lists
    random.seed(42)

    real_normals = list(real_normal_dir.glob("*.*"))
    real_pneumonias = list(real_pneumonia_dir.glob("*.*"))
    wgan_fakes = list(wgan_fake_dir.glob("*.*"))
    ddpm_fakes = list(ddpm_fake_dir.glob("*.*"))

    random.shuffle(real_normals)
    random.shuffle(real_pneumonias)
    random.shuffle(wgan_fakes)
    random.shuffle(ddpm_fakes)

    # Fixed split
    num_normal = len(real_normals)
    num_half = num_normal // 2
    num_rest = num_normal - num_half

    print(f"Total normal images: {num_normal}")
    print(f"Target pneumonia size: {num_normal}")
    print(f"Shared real pneumonia images: {num_half}")

    shared_real_pneumonia = real_pneumonias[:num_half]
    extra_real_pneumonia = real_pneumonias[num_half:num_normal]

    # Copy helper
    def copy_files(file_list, target_dir, desc="Copying"):
        target_dir.mkdir(parents=True, exist_ok=True)
        for f in tqdm(file_list, desc=desc, leave=False):
            shutil.copy(f, target_dir / f.name)

    # Experiment A: real-only baseline
    print("\nBuilding Experiment A (Baseline)...")
    exp_a_norm = exp_root / "Exp_A_Baseline" / "NORMAL"
    exp_a_pneu = exp_root / "Exp_A_Baseline" / "PNEUMONIA"

    copy_files(real_normals, exp_a_norm, "A - Normal")
    copy_files(shared_real_pneumonia, exp_a_pneu, "A - Shared Real")
    copy_files(extra_real_pneumonia, exp_a_pneu, "A - Extra Real")

    # Experiment B: WGAN
    print("\nBuilding Experiment B (WGAN)...")
    exp_b_norm = exp_root / "Exp_B_WGAN" / "NORMAL"
    exp_b_pneu = exp_root / "Exp_B_WGAN" / "PNEUMONIA"

    copy_files(real_normals, exp_b_norm, "B - Normal")
    copy_files(shared_real_pneumonia, exp_b_pneu, "B - Shared Real")
    copy_files(wgan_fakes[:num_rest], exp_b_pneu, "B - WGAN Fake")

    # xperiment C: DDPM
    print("\nBuilding Experiment C (DDPM)...")
    exp_c_norm = exp_root / "Exp_C_DDPM" / "NORMAL"
    exp_c_pneu = exp_root / "Exp_C_DDPM" / "PNEUMONIA"

    copy_files(real_normals, exp_c_norm, "C - Normal")
    copy_files(shared_real_pneumonia, exp_c_pneu, "C - Shared Real")
    copy_files(ddpm_fakes[:num_rest], exp_c_pneu, "C - DDPM Fake")

    print("\nDone. All experiment datasets are saved in ./Experiment_Datasets")
    print("You can load them with torchvision.datasets.ImageFolder.")


if __name__ == "__main__":
    build_datasets()