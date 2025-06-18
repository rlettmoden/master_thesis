import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tools.cfg import py2cfg
import os
import torch
from torch import nn
import cv2
import numpy as np
import argparse
from pathlib import Path
from tools.metric import Evaluator
from pytorch_lightning.loggers import CSVLogger
import random
from train_supervision import Supervision_Train
import re
# os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def remove_date(input_string):
    # Regex pattern to match the date and time portion
    pattern = r'_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$'
    
    # Use re.sub to replace the matched pattern with an empty string
    result = re.sub(pattern, '', input_string)
    
    return result

def adapt_state_dict(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("net.", "")
        if new_key.startswith("s1_spatial_decoder.shared_neck"):
            new_key = new_key.replace("s1_spatial_decoder", "modules_by_modality.s1.spatial_decoder")
            new_state_dict[new_key] = value
        elif new_key.startswith("s2_spatial_decoder.shared_neck"):
            new_key = new_key.replace("s2_spatial_decoder", "modules_by_modality.s2.spatial_decoder")
            new_state_dict[new_key] = value
        elif new_key.startswith("planet_spatial_decoder.shared_neck"):
            new_key = new_key.replace("planet_spatial_decoder", "modules_by_modality.planet.spatial_decoder")
            new_state_dict[new_key] = value
        else:
            new_state_dict[new_key] = value
    return new_state_dict


def test(config, args):
    seed_everything(42)

    checkpoint_callback = ModelCheckpoint(save_top_k=config.save_top_k, monitor=config.monitor,
                                          save_last=config.save_last, mode=config.monitor_mode,
                                          dirpath=os.path.join(args.out_dir, config.weights_path),
                                          filename=config.weights_name)
    logger = CSVLogger('lightning_logs', name=os.path.join(args.out_dir, config.log_name))

    model = Supervision_Train(config, args.out_dir)
    if config.pretrained_ckpt_path:
        model = Supervision_Train.load_from_checkpoint(config.pretrained_ckpt_path, config=config)

    trainer = pl.Trainer(devices=1, max_epochs=config.max_epoch, accelerator='auto',
                         check_val_every_n_epoch=config.check_val_every_n_epoch,
                         callbacks=[checkpoint_callback], strategy='ddp_find_unused_parameters_true',
                         logger=logger)
    # trainer.fit(model=model, ckpt_path=config.resume_ckpt_path)
    # model.after_init()
    # trainer.fit(model=model)


    # weights name & weights path
    # trainer = Supervision_Train.load_from_checkpoint(config.pretrained_ckpt_path, config=config)
    name = os.path.basename(remove_date(args.out_dir))
    checkpoint_path = os.path.join(args.out_dir, "model_weights", "testing", name, f"{name}.ckpt")
    weights = torch.load(checkpoint_path)
    adapted_weights = adapt_state_dict(weights["state_dict"])
    model = Supervision_Train.load_from_checkpoint(checkpoint_path, config=config, out_dir=args.out_dir, strict=False)
    model.net.load_state_dict(adapted_weights, strict=False)
    model.eval()
    trainer.test(model=model, dataloaders=config.test_loader)

def main():
    parser = argparse.ArgumentParser(description="Test and visualize model predictions")
    parser.add_argument("out_dir", type=str, help="Path to theoutput directory")
    args = parser.parse_args()

    root_config_dir = "/beegfs/work/y0092788/thesis/GeoSeg-main/config"
    name = os.path.basename(remove_date(args.out_dir))
    config_path = os.path.join(root_config_dir, f"{name}.py")

    config = py2cfg(config_path)
    test(config, args)

if __name__ == "__main__":
    main()
