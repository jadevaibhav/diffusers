### I am creating simple training pipeline for diffusion training, using proposed changes from multiple papers:
1. https://arxiv.org/pdf/2202.00512 (v-prediction and SNR weighting)
2. https://arxiv.org/pdf/2305.08891 (improving on common flaws in noise schedules and sampling)

I am starting with unconditional training class, and perhaps extend this to conditional generation and several other training methods. 
As the focus is on DDPM/DDIM, I am using DDPM for training (there's no difference in noise schedule for training).