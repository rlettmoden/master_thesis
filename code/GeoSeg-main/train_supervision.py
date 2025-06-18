import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tools.cfg import py2cfg
import os
import torch
from torch import nn
import cv2
import numpy as np
import argparse
import yaml
from pathlib import Path
import re
from tools.metric import Evaluator
from pytorch_lightning.loggers import CSVLogger
import random
from utils.sample_visualization import visualize_sample_with_predictions, visualize_predictions_and_auxiliary, denormalize
from geoseg.datasets.metrics_visualizer import plot_miou_from_csv, plot_loss_from_csv, plot_aux_loss_from_csv

os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
# os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    arg("-o", "--out_dir", type=Path, help="Path to the out_dir.", required=True)
    arg("-t", "--test_path", type=Path, help="Path to the test_idxs.", required=False, default=None)
    return parser.parse_args()

def test_and_visualize(config, checkpoint_path, num_samples=5):
    # Load the model from checkpoint
    model = Supervision_Train.load_from_checkpoint(checkpoint_path, config=config)
    model.eval()

    # # Create a trainer (no need for training, just for prediction)
    # trainer = Trainer(gpus=1)  # Assuming you want to use GPU

    # Get the test dataloader
    test_loader = config.test_loader

    # Get all test samples
    all_samples = []
    for batch in test_loader:
        all_samples.extend([(i, sample) for i, sample in enumerate(batch)])

    # Randomly select 5 samples
    selected_samples = random.sample(all_samples, min(num_samples, len(all_samples)))

    # Visualize the selected samples
    for idx, sample in selected_samples:
        # Move sample to the same device as the model
        sample = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}

        # Get prediction
        with torch.no_grad():
            prediction = model(sample)
            logits = prediction["logits"]
            pre_mask = torch.nn.functional.softmax(logits, dim=1).argmax(dim=1)

        # Denormalize the input images if necessary
        dataset = config.test_dataset
        for modality in config.active_modalities:
            if dataset.apply_transforms:
                mean, std = dataset.modalities[modality]["mean"], dataset.modalities[modality]["std"]
                sample[modality] = denormalize(sample[modality].cpu(), mean=mean, std=std)

        # Visualize the sample
        visualize_sample_with_predictions(
            sample, 
            idx, 
            dataset, 
            pre_mask.cpu(), 
            epoch=0,  # We're not in training, so epoch doesn't matter
            out_dir=config.out_dir, 
            mode="test_visualization"
        )

    print(f"Visualized {num_samples} random samples from the test set.")


def prettify_confusion_matrix(confusion_matrix, class_names):
    """
    Creates a prettified string representation of a confusion matrix.

    Args:
    confusion_matrix (list of lists): The confusion matrix as a 2D list.
    class_names (list): List of class names.

    Returns:
    str: A formatted string representation of the confusion matrix.
    """
    # Calculate the maximum width needed for each column
    max_width = max(len(str(max(max(row) for row in confusion_matrix))), 
                    max(len(name) for name in class_names))

    # Create the header
    header = "Pred \\ True ".ljust(max_width + 2)
    header += " | ".join(name.center(max_width) for name in class_names)
    
    # Create the separator line
    separator = "-" * len(header)
    
    # Create the matrix rows
    rows = []
    for i, row in enumerate(confusion_matrix):
        row_str = class_names[i].ljust(max_width + 2)
        row_str += " | ".join(str(val).center(max_width) for val in row)
        rows.append(row_str)
    
    # Combine all parts
    pretty_matrix = "\n".join([header, separator] + rows)
    
    return pretty_matrix


class Supervision_Train(pl.LightningModule):
    def __init__(self, config, out_dir):
        super().__init__()
        self.config = config
        self.net = config.net
        self.out_dir = out_dir

        self.loss = config.loss

        self.metrics_train = Evaluator(num_class=config.num_classes, ignore_index=0)
        self.metrics_val = Evaluator(num_class=config.num_classes, ignore_index=0)
        self.metrics_test = Evaluator(num_class=config.num_classes, ignore_index=0)

        self.aux_metrics = {}
        for modality in config.active_modalities:
            self.aux_metrics[modality] = Evaluator(num_class=config.num_classes, ignore_index=0)

        self.samples_data = {}

    def visualize_prediction(self, idx, mode, sample, dataset):
        # TODO generalize
        non_empty_modalities = np.array(sample["non_empty_modalities"]).flatten()
            
        with torch.no_grad():
            prediction = self.net(sample)
            
            prediction["logits"] = prediction["logits"].cpu()


            pre_mask = nn.Softmax(dim=1)(prediction["logits"])
            pre_mask = pre_mask.argmax(dim=1)


            empty_modalities = np.setdiff1d(dataset.active_modalities, non_empty_modalities)
            for modality in empty_modalities:
                for path in sample[f"{modality}_invalid_paths"]:
                    sample[modality].append(dataset.load_invalid_path(path, modality))
                sample[modality] = torch.stack(sample[modality])
                sample[modality] = sample[modality].unsqueeze(0)
            visualize_sample_with_predictions(sample, idx, dataset, pre_mask, self.current_epoch, self.out_dir, mode)
            if self.config.use_aux_head:
                aux_preds = {
                    "s1":None,
                    "s2":None,
                    "planet":None,
                }

                for modality in non_empty_modalities:
                    sample[modality] = sample[modality].cpu()
                    aux_mask = nn.Softmax(dim=1)(prediction[f"{modality}_aux"])
                    aux_mask = aux_mask.argmax(dim=1)
                    aux_preds[modality] = aux_mask
                visualize_predictions_and_auxiliary(sample, idx, dataset, pre_mask, aux_preds, self.current_epoch, self.out_dir, mode)


    def forward(self, x):
        # only net is used in the prediction/inference
        
        seg_pre = self.net(x)
        # TODO FIXME
        # return seg_pre["logits"]
        return seg_pre

    def training_step(self, batch, batch_idx):
        input_dict = batch
            # for modality in input_dict["non_empty_modalities"][0]:
            #     input_dict[modality] = input_dict[modality].unsqueeze(0)
        prediction = self.net(input_dict)
        loss = self.loss(prediction, input_dict["labels"])
        if self.config.single_temporal:
            prediction["logits"] = torch.max(prediction["logits"], dim=0)[0].unsqueeze(0)
            input_dict["labels"] = input_dict["labels"][:1]

        pre_mask = nn.Softmax(dim=1)(prediction["logits"])
        pre_mask = pre_mask.argmax(dim=1)
        for i in range(pre_mask.shape[0]):
            self.metrics_train.add_batch(input_dict["labels"][i].cpu().numpy(), pre_mask[i].cpu().numpy())
        self.metrics_train.add_batch(input_dict["labels"].cpu().numpy(), pre_mask.cpu().numpy())

        for key, value in loss.items():
            if not self.config.use_aux_loss:
                if key == "main":
                    continue
            self.log(f"train_loss_{key}", value, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # # aux_backbone_loss = 0
        # if hasattr(self.config, "backbone_aux") and self.config.backbone_aux:
        #     for modality in self.config.active_modalities:
        #         aux_backbone_loss += prediction[f"{modality}_backbone_aux"]
        # self.log(f"train_loss_backbone_{key}", value, on_step=False, on_epoch=True, prog_bar=True, logger=False, sync_dist=True)
        
        return {"loss": loss["all"]}

    def on_train_epoch_end(self):
        self._epoch_logger(self.metrics_train, "train")

    def validation_step(self, batch, batch_idx):
        input_dict = batch
        prediction = self.forward(input_dict)
        loss_val = self.loss(prediction, input_dict["labels"])
        if self.config.single_temporal:
            prediction["logits"] = torch.max(prediction["logits"], dim=0)[0].unsqueeze(0)
            input_dict["labels"] = input_dict["labels"][:1]

        pre_mask = nn.Softmax(dim=1)(prediction["logits"])
        pre_mask = pre_mask.argmax(dim=1)

        n_samples = 10
        # if (self.current_epoch > 0 & batch_idx < n_samples) and(self.current_epoch % 5 == 0 or self.current_epoch == self.config.max_epoch - 1):
        dataset = self.config.train_dataset
        if (batch_idx < n_samples) and(self.current_epoch % 25 == 0 or self.current_epoch == self.config.max_epoch - 1):
            for modality in self.config.active_modalities:
                if dataset.apply_transforms:
                    mean, std = dataset.modalities[modality]["mean"], dataset.modalities[modality]["std"]
                    input_dict[modality] = denormalize(input_dict[modality].detach().cpu(), mean=mean, std=std)
            visualize_sample_with_predictions(input_dict, batch_idx, dataset, pre_mask.detach().cpu(), self.current_epoch, self.out_dir, "validation")
            if self.config.use_aux_head:
                aux_preds = {
                    "s1":None,
                    "s2":None,
                    "planet":None,
                }

                for modality in input_dict["non_empty_modalities"]:
                    aux_mask = nn.Softmax(dim=1)(prediction[f"{modality[0]}_aux"])
                    aux_mask = aux_mask.argmax(dim=1)
                    aux_preds[modality[0]] = aux_mask
                visualize_predictions_and_auxiliary(input_dict, batch_idx, self.config.test_dataset, pre_mask.detach().cpu(), aux_preds, self.current_epoch, self.out_dir, "validation")
        # for i in range(mask.shape[0]):
        # self.metrics_val.add_batch(input_dict["labels"].cpu().numpy(), pre_mask.cpu().numpy())
        for i in range(pre_mask.shape[0]):
            self.metrics_val.add_batch(input_dict["labels"][i].cpu().numpy(), pre_mask[i].cpu().numpy())

        for key, value in loss_val.items():
            if not self.config.use_aux_loss:
                if key == "main":
                    continue
            self.log(f"val_loss_{key}", value, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)


        # # aux_backbone_loss = 0
        # if hasattr(self.config, "backbone_aux") and self.config.backbone_aux:
        #     for modality in self.config.active_modalities:
        #         aux_backbone_loss += prediction[f"{modality}_backbone_aux"]
        # self.log(f"val_loss_backbone_{key}", value, on_step=False, on_epoch=True, prog_bar=True, logger=False, sync_dist=True)

        return {"loss_val": loss_val["all"]}

    def save_precision_recall(self):
        with open(os.path.join(self.out_dir, "precision_recall_per_sample.yaml"), "w") as f:
            yaml.dump(self.samples_data, f, default_flow_style=False)

    def test_step(self, batch, batch_idx):
        input_dict = batch
        # input_dict["planet"] = input_dict["planet"][0][0].unsqueeze(0).unsqueeze(0)
        prediction = self.forward(input_dict)
        loss_test = self.loss(prediction, input_dict["labels"])
        if self.config.single_temporal:
            prediction["logits"] = torch.max(prediction["logits"], dim=0)[0].unsqueeze(0)
            input_dict["labels"] = input_dict["labels"][:1]

        pre_mask = nn.Softmax(dim=1)(prediction["logits"])
        pre_mask = pre_mask.argmax(dim=1)

        if self.config.use_aux_head:
            for modality in self.config.active_modalities:
                aux_mask = nn.Softmax(dim=1)(prediction[f"{modality}_aux"])
                aux_mask = aux_mask.argmax(dim=1)
                for i in range(aux_mask.shape[0]):
                    self.aux_metrics[modality].add_batch(input_dict["labels"][i].cpu().numpy(), aux_mask[i].cpu().numpy())

        # Calculate per-class precision and recall for this sample
        evaluater = Evaluator(self.config.num_classes, ignore_index=0)
        for i in range(pre_mask.shape[0]):
            evaluater.add_batch(input_dict["labels"][i].cpu().numpy(), pre_mask[i].cpu().numpy())
        precision = evaluater.Precision()
        recall = evaluater.Recall()
        f1 = evaluater.F1()

        # # Ensure the tensor is on CPU for counting
        # labelmap = labelmap.cpu()
        
        # # Get unique classes and their counts
        # unique_classes, pixel_counts = torch.unique(input_dict["labels"][0].cpu().numpy(), return_counts=True)
        
        # # Create a list of counts for all classes (including zeros)
        # all_class_counts = [0] * (self.config.num_classes)
        
        # # Fill in the counts for classes that are present
        # for class_id, count in zip(unique_classes.tolist(), pixel_counts.tolist()):
        #     # Ensure we don't access indices beyond max_class_index
        #     if class_id < self.config.num_classes:
        #         all_class_counts[class_id] = count

        sample_id = input_dict["idx"][0].item()
        sample_data = {
            "precision": {f"{self.config.test_dataset.class_names[j]}": float(precision[j-1]) for j in range(1, self.config.num_classes)},
            "recall": {f"{self.config.test_dataset.class_names[j]}": float(recall[j-1]) for j in range(1, self.config.num_classes)},
            "f1": {f"{self.config.test_dataset.class_names[j]}": float(f1[j-1]) for j in range(1, self.config.num_classes)}
        }
        self.samples_data[sample_id] = sample_data

        n_samples = 5
        # if (self.current_epoch > 0 & batch_idx < n_samples) and(self.current_epoch % 5 == 0 or self.current_epoch == self.config.max_epoch - 1):
        dataset = self.config.test_dataset
        if batch_idx < n_samples:
            for modality in self.config.active_modalities:
                if dataset.apply_transforms:
                    mean, std = dataset.modalities[modality]["mean"], dataset.modalities[modality]["std"]
                    input_dict[modality] = denormalize(input_dict[modality].detach().cpu(), mean=mean, std=std)
            visualize_sample_with_predictions(input_dict, batch_idx, self.config.test_dataset, pre_mask.detach().cpu(), self.current_epoch, self.out_dir, "test")
            
        # for i in range(mask.shape[0]):
        # self.metrics_val.add_batch(input_dict["labels"].cpu().numpy(), pre_mask.cpu().numpy())
        for i in range(pre_mask.shape[0]):
            self.metrics_test.add_batch(input_dict["labels"][i].cpu().numpy(), pre_mask[i].cpu().numpy())

        for key, value in loss_test.items():
            if not self.config.use_aux_loss:
                if key == "main":
                    continue
            self.log(f"test_loss_{key}", value, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # # aux_backbone_loss = 0
        # if hasattr(self.config, "backbone_aux") and self.config.backbone_aux:
        #     for modality in self.config.active_modalities:
        #         aux_backbone_loss += prediction[f"{modality}_backbone_aux"]
        # self.log(f"test_loss_backbone_{key}", value, on_step=False, on_epoch=True, prog_bar=True, logger=False, sync_dist=True)

        return {"loss_test": loss_test["all"]}

    def _epoch_logger(self, metrics, mode):
        # updated ignore indices!
        mIoU = np.nanmean(metrics.Intersection_over_Union())
        F1 = np.nanmean(metrics.F1())

        OA = np.nanmean(metrics.OA())
        iou_per_class = metrics.Intersection_over_Union()

        eval_value = {'mIoU': mIoU,
                      'F1': F1,
                      'OA': OA}
        # print('val:', eval_value)
        iou_value = {}
        # regard ignore index
        for class_name, iou in zip(self.config.train_dataset.class_names[1:], iou_per_class):
            if class_name == "tree":
                continue
            iou_value[f"IoU_{mode}_{class_name}"] = iou
        self.log_dict(iou_value, prog_bar=False, sync_dist=True)


        conf_str = prettify_confusion_matrix(metrics.confusion_matrix, self.config.train_dataset.class_names)
        print(conf_str)
        # self.log("confusion matrix", conf_str, logger=False, prog_bar=False, sync_dist=True)
        metrics.reset()
        log_dict = {f'{mode}_mIoU': mIoU, f'{mode}_F1': F1, f'{mode}_OA': OA}
        self.log_dict(log_dict, prog_bar=True, sync_dist=True)
    
    def on_test_epoch_end(self):
        self.save_precision_recall()
        confusion_matrix = self.metrics_test.confusion_matrix
        out_file = os.path.join(self.out_dir, "test_confusion.npz")
        np.savez(out_file, confusion_matrix=confusion_matrix)
        self._epoch_logger(self.metrics_test, "test")

        if self.config.use_aux_head:
            for modality in self.config.active_modalities:
                metrics = self.aux_metrics[modality]
                mIoU = np.nanmean(metrics.Intersection_over_Union())
                F1 = np.nanmean(metrics.F1())

                OA = np.nanmean(metrics.OA())
                log_dict = {f'test_{modality}_mIoU': mIoU, f'test_{modality}_F1': F1, f'test_{modality}_OA': OA}
                self.log_dict(log_dict, prog_bar=False, sync_dist=True)
                iou_per_class = metrics.Intersection_over_Union()
                iou_value = {}
                # regard ignore index
                for class_name, iou in zip(self.config.train_dataset.class_names[1:], iou_per_class):
                    if class_name == "tree":
                        continue
                    iou_value[f"IoU_{modality}_test_{class_name}"] = iou
                self.log_dict(iou_value, prog_bar=False, sync_dist=True)

    def on_validation_epoch_end(self):
        self._epoch_logger(self.metrics_val, "val")

    def configure_optimizers(self):
        optimizer = self.config.optimizer
        lr_scheduler = self.config.lr_scheduler

        return [optimizer], [lr_scheduler]

    def train_dataloader(self):

        return self.config.train_loader

    def val_dataloader(self):

        return self.config.val_loader


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

# training
def main():
    args = get_args()
    config = py2cfg(args.config_path)

    seed = config.seed if hasattr(config, 'seed') else 42
    seed_everything(seed)

    print(str(args.out_dir))
    checkpoint_callback = ModelCheckpoint(save_top_k=config.save_top_k, monitor=config.monitor,
                                          save_last=config.save_last, mode=config.monitor_mode,
                                          dirpath=os.path.join(args.out_dir, config.weights_path),
                                          filename=config.weights_name)
    logger = CSVLogger('lightning_logs', name=os.path.join(args.out_dir, config.log_name))

    model = Supervision_Train(config, args.out_dir)
    if config.pretrained_ckpt_path:
        model = Supervision_Train.load_from_checkpoint(config.pretrained_ckpt_path, config=config)

    trainer = pl.Trainer(devices=config.gpus, max_epochs=config.max_epoch, accelerator='auto',
                         check_val_every_n_epoch=config.check_val_every_n_epoch,
                         callbacks=[checkpoint_callback], strategy='ddp_find_unused_parameters_true',
                         logger=logger)
    # trainer.fit(model=model, ckpt_path=config.resume_ckpt_path)
    # model.after_init()
    trainer.fit(model=model)

    out_dir = str(args.out_dir)
    plot_miou_from_csv(f"{logger.log_dir}/metrics.csv", f"{out_dir}/mIoU.png")
    plot_loss_from_csv(f"{logger.log_dir}/metrics.csv", f"{out_dir}/loss.png")

    if config.use_aux_head:
        plot_aux_loss_from_csv(f"{logger.log_dir}/metrics.csv", f"{out_dir}/aux_loss.png", config.train_dataset.active_modalities)


    # TEST
    model = None
    trainer = pl.Trainer(devices=1, max_epochs=config.max_epoch, accelerator='auto',
                         check_val_every_n_epoch=config.check_val_every_n_epoch,
                         callbacks=[checkpoint_callback], strategy='ddp_find_unused_parameters_true',
                         logger=logger)
    # trainer.fit(model=model, ckpt_path=config.resume_ckpt_path)
    # model.after_init()
    # trainer.fit(model=model)


    # weights name & weights path
    # trainer = Supervision_Train.load_from_checkpoint(config.pretrained_ckpt_path, config=config)
    name = os.path.basename(remove_date(out_dir))
    checkpoint_path = os.path.join(out_dir, "model_weights", "testing", name, f"{name}.ckpt")
    weights = torch.load(checkpoint_path)
    adapted_weights = adapt_state_dict(weights["state_dict"])
    model = Supervision_Train.load_from_checkpoint(checkpoint_path, config=config, out_dir=args.out_dir, strict=False)
    model.net.load_state_dict(adapted_weights, strict=False)
    model.eval()
    trainer.test(model=model, dataloaders=config.test_loader)



if __name__ == "__main__":
   main()
