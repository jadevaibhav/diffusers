import torch
import argparse
from diffusers import UNet2DModel, DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import logging
from tqdm.auto import tqdm

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