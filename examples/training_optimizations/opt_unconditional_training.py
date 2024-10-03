import argparse
import inspect
import logging
import math
import os
from datetime import timedelta
from pathlib import Path
import yaml

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler, DDIMPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import is_tensorboard_available, is_wandb_available, check_min_version
from packaging import version


# Ensure the correct version of diffusers is installed
check_min_version("0.31.0.dev0")

# Logger setup
logger = get_logger(__name__, log_level="INFO")

class Trainer:
    def __init__(self, args):
        self.args = args
        self.accelerator = self._initialize_accelerator(args)
        self.model = self._initialize_model(args)
        self.ema_model = self._initialize_ema_model(args)
        self.noise_scheduler = self._initialize_scheduler(args)
        self.optimizer = self._initialize_optimizer(args)
        self.train_dataloader = self._initialize_dataloader(args)
        self.lr_scheduler = self._initialize_lr_scheduler(args)
        self.use_SNR_weights = args.use_SNR_weights


    def _initialize_accelerator(self, args):
        logging_dir = os.path.join(args.output_dir, args.logging_dir)
        accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

        kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))  # High timeout for big datasets
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with=args.logger,
            project_config=accelerator_project_config,
            kwargs_handlers=[kwargs],
        )

        return accelerator

    def _initialize_model(self, args):
        if args.model_config_name_or_path is None:
            model = UNet2DModel(
                sample_size=args.resolution,
                in_channels=3,
                out_channels=3,
                layers_per_block=2,
                block_out_channels=(128, 128, 256, 256, 512, 512),
                down_block_types=(
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "AttnDownBlock2D",
                    "DownBlock2D",
                ),
                up_block_types=(
                    "UpBlock2D",
                    "AttnUpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                ),
            )
        else:
            config = UNet2DModel.load_config(args.model_config_name_or_path)
            model = UNet2DModel.from_config(config)

        return model

    def _initialize_ema_model(self, args):
        if args.use_ema:
            ema_model = EMAModel(
                self.model.parameters(),
                decay=args.ema_max_decay,
                use_ema_warmup=True,
                inv_gamma=args.ema_inv_gamma,
                power=args.ema_power,
                model_cls=UNet2DModel,
                model_config=self.model.config,
            )
            return ema_model
        return None

    def _initialize_scheduler(self, args):
        accepts_prediction_type = "prediction_type" in set(inspect.signature(DDPMScheduler.__init__).parameters.keys())
        if accepts_prediction_type:
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=args.ddpm_num_steps,
                beta_schedule=args.ddpm_beta_schedule,
                prediction_type=args.prediction_type,
            )
        else:
            noise_scheduler = DDPMScheduler(num_train_timesteps=args.ddpm_num_steps, beta_schedule=args.ddpm_beta_schedule)
        
        return noise_scheduler

    def _initialize_optimizer(self, args):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
        return optimizer
    
    def _initialize_lr_scheduler(self,args):
        lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=self.optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=(len(self.train_dataloader) * args.num_epochs),
        )

        return lr_scheduler
        
    def _initialize_dataloader(self, args):
        dataset = self._load_dataset(args)

        augmentations = transforms.Compose([
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        def transform_images(examples):
            images = [augmentations(image.convert("RGB")) for image in examples["image"]]
            return {"input": images}

        dataset.set_transform(transform_images)
        train_dataloader = DataLoader(
            dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
        )

        return train_dataloader

    def _load_dataset(self, args):
        if args.dataset_name:
            dataset = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                cache_dir=args.cache_dir,
                split="train",
            )
        else:
            dataset = load_dataset("imagefolder", data_dir=args.train_data_dir, cache_dir=args.cache_dir, split="train")

        return dataset

    def evaluate_model_for_visual_inspection(self, args, epoch, global_step):
        if self.accelerator.is_main_process and (epoch % args.save_images_epochs == 0 or epoch == args.num_epochs - 1):
            unet = self.accelerator.unwrap_model(self.model)

            if args.use_ema and self.ema_model is not None:
                self.ema_model.store(unet.parameters())
                self.ema_model.copy_to(unet.parameters())

            ### We use DDIM for inmferencing here, with recommendations for the bytedance paper
            inf_noise_scheduler = DDIMScheduler.from_config(self.noise_scheduler.config,
                                                             rescale_betas_zero_snr=True, timestep_spacing="trailing")

            pipeline = DDIMPipeline(unet=unet, scheduler=inf_noise_scheduler)
            generator = torch.Generator(device=pipeline.device).manual_seed(0)
            
            images = pipeline(
                generator=generator,
                batch_size=args.eval_batch_size,
                num_inference_steps=args.ddpm_num_inference_steps,
                output_type="np",
            ).images

            if args.use_ema and self.ema_model is not None:
                self.ema_model.restore(unet.parameters())

            images_processed = (images * 255).round().astype("uint8")

            if args.logger == "tensorboard":
                tracker = self.accelerator.get_tracker("tensorboard", unwrap=True)
                tracker.add_images("test_samples", images_processed.transpose(0, 3, 1, 2), epoch)
            elif args.logger == "wandb":
                self.accelerator.get_tracker("wandb").log(
                    {"test_samples": [wandb.Image(img) for img in images_processed], "epoch": epoch},
                    step=global_step,
                )

    def train(self):
        if self.accelerator:
            self.model, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
                self.model, self.optimizer, self.train_dataloader, self.lr_scheduler
            )
        if self.args.use_ema:
            self.ema_model.to(self.accelerator.device)

        total_batch_size = self.args.train_batch_size * self.accelerator.num_processes * self.args.gradient_accumulation_steps
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.args.gradient_accumulation_steps)
        max_train_steps = self.args.num_epochs * num_update_steps_per_epoch

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataloader.dataset)}")
        logger.info(f"  Num Epochs = {self.args.num_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.args.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_train_steps}")

        global_step = 0
        first_epoch = 0

        # Training loop here...


def parse_yaml_config(yaml_file):
    """Parse the YAML configuration file."""
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def merge_args_with_config(args, config):
    """Merge command-line arguments with YAML configuration, giving priority to command-line args."""
    for key, value in config.items():
        if isinstance(value, dict):
            # For nested dictionaries, we need to handle merging recursively
            for sub_key, sub_value in value.items():
                if hasattr(args, sub_key):
                    setattr(args, sub_key, sub_value)
        else:
            if hasattr(args, key):
                setattr(args, key, value)
    return args

def parse_args():
    parser = argparse.ArgumentParser(description="DDPM Training Configuration")

    # Add YAML config argument
    parser.add_argument('--config_file', type=str, default=None, help="Path to the YAML configuration file")

    # Training config
    parser.add_argument('--output_dir', type=str, default="ddpm-model-64", help="Directory to output model and images")
    parser.add_argument('--overwrite_output_dir', action='store_true', help="Overwrite output directory if it exists")
    parser.add_argument('--cache_dir', type=str, default=None, help="Cache directory")
    parser.add_argument('--num_epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--save_images_epochs', type=int, default=10, help="Save generated images every N epochs")
    parser.add_argument('--save_model_epochs', type=int, default=10, help="Save model checkpoints every N epochs")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument('--train_batch_size', type=int, default=16, help="Training batch size")
    parser.add_argument('--eval_batch_size', type=int, default=16, help="Evaluation batch size")
    parser.add_argument('--dataloader_num_workers', type=int, default=0, help="Number of dataloader workers")
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument('--checkpointing_steps', type=int, default=500, help="Number of steps between checkpoints")
    parser.add_argument('--checkpoints_total_limit', type=int, default=None, help="Max number of checkpoints to keep")
    parser.add_argument('--local_rank', type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument('--logger', type=str, default="tensorboard", help="Logger type (e.g., tensorboard)")
    parser.add_argument('--logging_dir', type=str, default="logs", help="Directory for logging")
    parser.add_argument('--mixed_precision', type=str, choices=['no', 'fp16', 'bf16'], default="no", help="Mixed precision training")
    parser.add_argument('--push_to_hub', action='store_true', help="Push model to Hugging Face hub")
    parser.add_argument('--hub_model_id', type=str, default=None, help="Model ID for Hugging Face hub")
    parser.add_argument('--hub_private_repo', action='store_true', help="Create private repo on Hugging Face hub")
    parser.add_argument('--hub_token', type=str, default=None, help="Hugging Face token")
    parser.add_argument('--enable_xformers_memory_efficient_attention', action='store_true', help="Use memory efficient attention (xformers)")
    parser.add_argument('--use_ema', action='store_true', help="Use exponential moving average (EMA)")

    # Image augmentation config
    parser.add_argument('--resolution', type=int, default=64, help="Image resolution")
    parser.add_argument('--center_crop', action='store_true', help="Center crop the images")
    parser.add_argument('--random_flip', action='store_true', help="Randomly flip images during training")

    # Learning rate and scheduler config
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument('--lr_scheduler', type=str, choices=['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup'], default='cosine', help="Learning rate scheduler type")
    parser.add_argument('--lr_warmup_steps', type=int, default=500, help="Number of warmup steps")
    parser.add_argument('--adam_beta1', type=float, default=0.95, help="Beta1 for Adam optimizer")
    parser.add_argument('--adam_beta2', type=float, default=0.999, help="Beta2 for Adam optimizer")
    parser.add_argument('--adam_weight_decay', type=float, default=1e-6, help="Weight decay for Adam optimizer")
    parser.add_argument('--adam_epsilon', type=float, default=1e-08, help="Epsilon for Adam optimizer")

    # DDPM noise schedule config
    parser.add_argument('--ddpm_num_steps', type=int, default=1000, help="Number of DDPM steps")
    parser.add_argument('--ddpm_num_inference_steps', type=int, default=1000, help="Number of inference steps in DDPM")
    parser.add_argument('--ddpm_beta_schedule', type=str, choices=['linear', 'cosine'], default='linear', help="Beta schedule for DDPM")
    parser.add_argument('--prediction_type', type=str, choices=['epsilon', 'sample', 'v_prediction'], default='epsilon', help="Prediction type for DDPM")

    # EMA config
    parser.add_argument('--ema_inv_gamma', type=float, default=1.0, help="EMA inverse gamma")
    parser.add_argument('--ema_power', type=float, default=0.75, help="EMA power")
    parser.add_argument('--ema_max_decay', type=float, default=0.9999, help="EMA maximum decay")

    # Dataset config
    parser.add_argument('--dataset_name', type=str, default=None, help="Dataset name")
    parser.add_argument('--dataset_config_name', type=str, default=None, help="Dataset configuration name")
    parser.add_argument('--train_data_dir', type=str, default=None, help="Directory of training data")

    args = parser.parse_args()

    # Load and merge YAML config if specified
    if args.config_file:
        yaml_config = parse_yaml_config(args.config_file)
        args = merge_args_with_config(args, yaml_config)

    return args

# def parse_args():
#     parser = argparse.ArgumentParser(description="Simple example of a training script.")
#     parser.add_argument(
#         "--dataset_name",
#         type=str,
#         default=None,
#         help=(
#             "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
#             " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
#             " or to a folder containing files that HF Datasets can understand."
#         ),
#     )
#     parser.add_argument(
#         "--dataset_config_name",
#         type=str,
#         default=None,
#         help="The config of the Dataset, leave as None if there's only one config.",
#     )
#     parser.add_argument(
#         "--model_config_name_or_path",
#         type=str,
#         default=None,
#         help="The config of the UNet model to train, leave as None to use standard DDPM configuration.",
#     )
#     parser.add_argument(
#         "--train_data_dir",
#         type=str,
#         default=None,
#         help=(
#             "A folder containing the training data. Folder contents must follow the structure described in"
#             " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
#             " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
#         ),
#     )
#     parser.add_argument(
#         "--output_dir",
#         type=str,
#         default="ddpm-model-64",
#         help="The output directory where the model predictions and checkpoints will be written.",
#     )
#     parser.add_argument("--overwrite_output_dir", action="store_true")
#     parser.add_argument(
#         "--cache_dir",
#         type=str,
#         default=None,
#         help="The directory where the downloaded models and datasets will be stored.",
#     )
#     parser.add_argument(
#         "--resolution",
#         type=int,
#         default=64,
#         help=(
#             "The resolution for input images, all the images in the train/validation dataset will be resized to this"
#             " resolution"
#         ),
#     )
#     parser.add_argument(
#         "--center_crop",
#         default=False,
#         action="store_true",
#         help=(
#             "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
#             " cropped. The images will be resized to the resolution first before cropping."
#         ),
#     )
#     parser.add_argument(
#         "--random_flip",
#         default=False,
#         action="store_true",
#         help="whether to randomly flip images horizontally",
#     )
#     parser.add_argument(
#         "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
#     )
#     parser.add_argument(
#         "--eval_batch_size", type=int, default=16, help="The number of images to generate for evaluation."
#     )
#     parser.add_argument(
#         "--dataloader_num_workers",
#         type=int,
#         default=0,
#         help=(
#             "The number of subprocesses to use for data loading. 0 means that the data will be loaded in the main"
#             " process."
#         ),
#     )
#     parser.add_argument("--num_epochs", type=int, default=100)
#     parser.add_argument("--save_images_epochs", type=int, default=10, help="How often to save images during training.")
#     parser.add_argument(
#         "--save_model_epochs", type=int, default=10, help="How often to save the model during training."
#     )
#     parser.add_argument(
#         "--gradient_accumulation_steps",
#         type=int,
#         default=1,
#         help="Number of updates steps to accumulate before performing a backward/update pass.",
#     )
#     parser.add_argument(
#         "--learning_rate",
#         type=float,
#         default=1e-4,
#         help="Initial learning rate (after the potential warmup period) to use.",
#     )
#     parser.add_argument(
#         "--lr_scheduler",
#         type=str,
#         default="cosine",
#         help=(
#             'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
#             ' "constant", "constant_with_warmup"]'
#         ),
#     )
#     parser.add_argument(
#         "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
#     )
#     parser.add_argument("--adam_beta1", type=float, default=0.95, help="The beta1 parameter for the Adam optimizer.")
#     parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
#     parser.add_argument(
#         "--adam_weight_decay", type=float, default=1e-6, help="Weight decay magnitude for the Adam optimizer."
#     )
#     parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer.")
#     parser.add_argument(
#         "--use_ema",
#         action="store_true",
#         help="Whether to use Exponential Moving Average for the final model weights.",
#     )
#     parser.add_argument("--ema_inv_gamma", type=float, default=1.0, help="The inverse gamma value for the EMA decay.")
#     parser.add_argument("--ema_power", type=float, default=3 / 4, help="The power value for the EMA decay.")
#     parser.add_argument("--ema_max_decay", type=float, default=0.9999, help="The maximum decay magnitude for EMA.")
#     parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
#     parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
#     parser.add_argument(
#         "--hub_model_id",
#         type=str,
#         default=None,
#         help="The name of the repository to keep in sync with the local `output_dir`.",
#     )
#     parser.add_argument(
#         "--hub_private_repo", action="store_true", help="Whether or not to create a private repository."
#     )
#     parser.add_argument(
#         "--logger",
#         type=str,
#         default="tensorboard",
#         choices=["tensorboard", "wandb"],
#         help=(
#             "Whether to use [tensorboard](https://www.tensorflow.org/tensorboard) or [wandb](https://www.wandb.ai)"
#             " for experiment tracking and logging of model metrics and model checkpoints"
#         ),
#     )
#     parser.add_argument(
#         "--logging_dir",
#         type=str,
#         default="logs",
#         help=(
#             "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
#             " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
#         ),
#     )
#     parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
#     parser.add_argument(
#         "--mixed_precision",
#         type=str,
#         default="no",
#         choices=["no", "fp16", "bf16"],
#         help=(
#             "Whether to use mixed precision. Choose"
#             "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
#             "and an Nvidia Ampere GPU."
#         ),
#     )
#     parser.add_argument(
#         "--prediction_type",
#         type=str,
#         default="epsilon",
#         choices=["epsilon", "sample", "v_prediction"],
#         help="Whether the model should predict the 'epsilon'/noise error or directly the reconstructed image 'x0', OR use velocity prediction.",
#     )
#     parser.add_argument("--ddpm_num_steps", type=int, default=1000)
#     parser.add_argument("--ddpm_num_inference_steps", type=int, default=1000)
#     parser.add_argument("--ddpm_beta_schedule", type=str, default="linear")
#     parser.add_argument(
#         "--checkpointing_steps",
#         type=int,
#         default=500,
#         help=(
#             "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
#             " training using `--resume_from_checkpoint`."
#         ),
#     )
#     parser.add_argument(
#         "--checkpoints_total_limit",
#         type=int,
#         default=None,
#         help=("Max number of checkpoints to store."),
#     )
#     parser.add_argument(
#         "--resume_from_checkpoint",
#         type=str,
#         default=None,
#         help=(
#             "Whether training should be resumed from a previous checkpoint. Use a path saved by"
#             ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
#         ),
#     )
#     parser.add_argument(
#         "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
#     )

#     args = parser.parse_args()
#     env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
#     if env_local_rank != -1 and env_local_rank != args.local_rank:
#         args.local_rank = env_local_rank

#     if args.dataset_name is None and args.train_data_dir is None:
#         raise ValueError("You must specify either a dataset name from the hub or a train data directory.")

#     return args

if __name__ == "__main__":
    args = parse_args()
    if args.logger == "tensorboard":
        if not is_tensorboard_available():
            raise ImportError("Make sure to install tensorboard if you want to use it for logging during training.")

    elif args.logger == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb
    trainer = Trainer(args)
    trainer.train()








class CustomDiffusionTrainer:
    def __init__(
        self,
        model: UNet2DModel,
        noise_scheduler: DDPMScheduler,
        train_dataset,
        train_batch_size: int = 16,
        gradient_accumulation_steps: int = 1,
        num_train_epochs: int = 100,
        max_train_steps: int = None,
        learning_rate: float = 1e-4,
        lr_scheduler_type: str = "cosine",
        num_warmup_steps: int = 500,
        adam_beta1: float = 0.95,
        adam_beta2: float = 0.999,
        adam_weight_decay: float = 1e-6,
        adam_epsilon: float = 1e-08,
        use_ema: bool = True,
        ema_inv_gamma: float = 1.0,
        ema_power: float = 3/4,
        ema_max_decay: float = 0.9999,
    ):
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.train_dataset = train_dataset
        self.train_batch_size = train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_train_epochs = num_train_epochs
        self.max_train_steps = max_train_steps
        self.learning_rate = learning_rate
        self.lr_scheduler_type = lr_scheduler_type
        self.num_warmup_steps = num_warmup_steps
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_weight_decay = adam_weight_decay
        self.adam_epsilon = adam_epsilon
        self.use_ema = use_ema
        self.ema_inv_gamma = ema_inv_gamma
        self.ema_power = ema_power
        self.ema_max_decay = ema_max_decay

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(self.adam_beta1, self.adam_beta2),
            weight_decay=self.adam_weight_decay,
            eps=self.adam_epsilon,
        )

        self.lr_scheduler = get_scheduler(
            self.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.max_train_steps,
        )

        if self.use_ema:
            self.ema_model = EMAModel(
                self.model,
                inv_gamma=self.ema_inv_gamma,
                power=self.ema_power,
                max_value=self.ema_max_decay,
            )

    def train(self):
        progress_bar = tqdm(range(self.max_train_steps), desc="Training")
        global_step = 0

        for epoch in range(self.num_train_epochs):
            self.model.train()
            for step, batch in enumerate(self.train_dataset):
                clean_images = batch["pixel_values"].to(self.device)
                
                # Sample noise to add to the images
                noise = torch.randn(clean_images.shape).to(clean_images.device)
                bs = clean_images.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device).long()

                # Add noise to the clean images according to the noise magnitude at each timestep
                noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)

                # Predict the noise residual
                noise_pred = self.model(noisy_images, timesteps, return_dict=False)[0]

                loss = torch.nn.functional.mse_loss(noise_pred, noise)
                loss.backward()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    if self.use_ema:
                        self.ema_model.step(self.model)

                    progress_bar.update(1)
                    global_step += 1

                if global_step >= self.max_train_steps:
                    break

            if global_step >= self.max_train_steps:
                break

        # Save the model after training
        self.model.save_pretrained("path_to_save_model")
        if self.use_ema:
            self.ema_model.save_pretrained("path_to_save_ema_model")

def parse_args():
    parser = argparse.ArgumentParser(description="Train a diffusion model")
    parser.add_argument("--model_path", type=str, default=None, help="Path to pretrained model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the training dataset")
    parser.add_argument("--output_dir", type=str, default="./output", help="Where to save the trained model")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--num_train_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Max number of training steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", help="The scheduler type to use")
    parser.add_argument("--num_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler")
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model")
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0, help="The inverse gamma value for the EMA decay")
    parser.add_argument("--ema_power", type=float, default=3/4, help="The power value for the EMA decay")
    parser.add_argument("--ema_max_decay", type=float, default=0.9999, help="The maximum decay magnitude for EMA")
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # Load model
    if args.model_path:
        model = UNet2DModel.from_pretrained(args.model_path)
    else:
        model = UNet2DModel()
    
    # Initialize noise scheduler
    noise_scheduler = DDPMScheduler()
    
    # Load dataset (you'll need to implement this part based on your dataset)
    train_dataset = load_dataset(args.dataset_path)
    
    # Initialize trainer
    trainer = CustomDiffusionTrainer(
        model=model,
        noise_scheduler=noise_scheduler,
        train_dataset=train_dataset,
        train_batch_size=args.train_batch_size,
        num_train_epochs=args.num_train_epochs,
        max_train_steps=args.max_train_steps,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr_scheduler_type=args.lr_scheduler,
        num_warmup_steps=args.num_warmup_steps,
        use_ema=args.use_ema,
        ema_inv_gamma=args.ema_inv_gamma,
        ema_power=args.ema_power,
        ema_max_decay=args.ema_max_decay,
    )
    
    # Start training
    trainer.train()
    
    # Save the final model
    trainer.model.save_pretrained(args.output_dir)
    if args.use_ema:
        trainer.ema_model.save_pretrained(f"{args.output_dir}_ema")

if __name__ == "__main__":
    main()