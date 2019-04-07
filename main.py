from train import Trainer_DPE
from parameter import *
from utils import make_folder
from torchvision.transforms import transforms
import torch.utils.data as data
from data import Load_Data
import os
from tqdm import tqdm
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = '4,5'

def main(config):
    # Data_Loader
    warnings.filterwarnings("ignore")
    print("Loading data")
    dataset = Load_Data(config.img_root1, config.img_root2, config.train_file1, config.train_file2, suffix=['.dng', '.tif'])

    data_loader = data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True
    )

    print("Loading completed")

    # Create directories if not exist
    make_folder(config.model_save_path, config.version)
    make_folder(config.sample_path, config.version)
    make_folder(config.log_path, config.version)
    print("Directories created")

    if config.train:
        if config.model == 'WGAN-v24-cycleganD2':
            trainer = Trainer_DPE(data_loader, config)

        trainer.train()

if __name__ == '__main__':
    config = get_parameters()
    main(config)
