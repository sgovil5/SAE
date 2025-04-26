import torch as t
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from typing import Optional

from ..dictionary import Dictionary
from ..trainers.trainer import (
    SAETrainer,
    get_lr_schedule,
    set_decoder_norm_to_unit_norm,
    remove_gradient_parallel_to_decoder_directions,
)


class WTASAE(Dictionary, nn.Module):
    """
    Winner Takes All Sparse Autoencoder.

    For each dictionary atom (i.e. each column of the encoder's output), this model
    only retains the top k activations across the batch, where
    k_per_feature = max(1, int(batch_size * sparsity_rate)).
    """
    def __init__(self, activation_dim: int, dict_size: int, sparsity_rate: float):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        assert 0 < sparsity_rate <= 1.0, f"sparsity_rate must be in (0, 1], got {sparsity_rate}"
        self.sparsity_rate = sparsity_rate

        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        self.decoder.weight.data = set_decoder_norm_to_unit_norm(
            self.decoder.weight, activation_dim, dict_size
        )

        self.encoder = nn.Linear(activation_dim, dict_size)
        self.encoder.weight.data = self.decoder.weight.t().clone()
        self.encoder.bias.data.zero_()

        self.b_dec = nn.Parameter(t.zeros(activation_dim))

        # Add threshold parameter
        self.register_buffer("threshold", t.tensor(-1.0, dtype=t.float32))

    def encode(self, x: t.Tensor, return_active: bool = False, use_threshold: bool = True, **kwargs):
        """
        Computes encoder activations followed by a winner-takes-all (WTA) selection per dictionary atom.
        """
        pre_relu_acts = F.relu(self.encoder(x - self.b_dec))  # (B, dict_size)
        
        if use_threshold and self.threshold > 0:
            # Apply threshold before WTA
            pre_relu_acts = pre_relu_acts * (pre_relu_acts > self.threshold)
        
        batch_size = x.size(0)
        k_per_feature = max(1, int(batch_size * self.sparsity_rate))

        # Transpose so each row corresponds to one dictionary element
        acts_t = pre_relu_acts.t()  # (dict_size, B)
        topk_values, _ = acts_t.topk(k_per_feature, dim=1)
        thresholds = topk_values[:, -1].unsqueeze(1)  # (dict_size, 1)
        mask = (acts_t >= thresholds).float()

        wta_acts = (acts_t * mask).t()  # (B, dict_size)

        if return_active:
            active_indices = (wta_acts.sum(dim=0) > 0)
            return wta_acts, active_indices, pre_relu_acts
        else:
            return wta_acts

    def decode(self, encoded: t.Tensor) -> t.Tensor:
        return self.decoder(encoded) + self.b_dec

    def forward(self, x: t.Tensor, output_features: bool = False):
        encoded = self.encode(x, return_active=False)
        x_hat = self.decode(encoded)
        if not output_features:
            return x_hat
        else:
            return x_hat, encoded

    def scale_biases(self, scale: float):
        self.b_dec.data *= scale
        if self.threshold >= 0:
            self.threshold *= scale

    @classmethod
    def from_pretrained(cls, path, sparsity_rate=None, device=None, **kwargs) -> "WTASAE":
        state_dict = t.load(path)
        dict_size, activation_dim = state_dict["encoder.weight"].shape
        if sparsity_rate is None:
            sparsity_rate = 0.1  # default value if not supplied
        autoencoder = cls(activation_dim, dict_size, sparsity_rate)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder


class WTATrainer(SAETrainer):
    """
    Trainer for the Winner-Takes-All Sparse Autoencoder.
    
    Follows a similar training framework as BatchTopKTrainer, but applies the WTA mechanism in the autoencoder.
    """
    def __init__(
        self,
        steps: int,  # total training steps
        activation_dim: int,
        dict_size: int,
        sparsity_rate: float,
        layer: int,
        lm_name: str,
        dict_class: type = WTASAE,
        lr: Optional[float] = None,
        auxk_alpha: float = 1 / 32,
        warmup_steps: int = 1000,  # Added parameter
        decay_start: Optional[int] = None,  # Added parameter
        seed: Optional[int] = None,
        device: Optional[str] = None,
        wandb_name: str = "WTASAE",
        submodule_name: Optional[str] = None,
        threshold_beta: float = 0.999,
        threshold_start_step: int = 1000,
    ):
        super().__init__(seed)
        self.seed = seed  # store for logging
        assert layer is not None and lm_name is not None
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        self.wandb_name = wandb_name
        self.steps = steps

        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)

        self.ae = dict_class(activation_dim, dict_size, sparsity_rate)

        if device is None:
            self.device = "cuda" if t.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.ae.to(self.device)

        if lr is not None:
            self.lr = lr
        else:
            # LR using 1/sqrt(d) scaling law (see paper Figure 3)
            scale = dict_size / (2**14)
            self.lr = 2e-4 / scale**0.5

        self.auxk_alpha = auxk_alpha
        self.dead_feature_threshold = 10_000_000
        self.top_k_aux = activation_dim // 2  # heuristic from the paper
        self.num_tokens_since_fired = t.zeros(dict_size, dtype=t.long, device=self.device)
        self.logging_parameters = ["effective_l0", "dead_features", "pre_norm_auxk_loss"]
        self.effective_l0 = -1
        self.dead_features = -1
        self.pre_norm_auxk_loss = -1

        self.optimizer = t.optim.Adam(self.ae.parameters(), lr=self.lr, betas=(0.9, 0.999))
        lr_fn = get_lr_schedule(steps, warmup_steps, decay_start=decay_start)
        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)

        self.warmup_steps = warmup_steps
        self.decay_start = decay_start

        self.threshold_beta = threshold_beta
        self.threshold_start_step = threshold_start_step

    def get_auxiliary_loss(self, residual_BD: t.Tensor, post_relu_acts_BF: t.Tensor):
        dead_features = self.num_tokens_since_fired >= self.dead_feature_threshold
        self.dead_features = int(dead_features.sum())

        if dead_features.sum() > 0:
            k_aux = min(self.top_k_aux, int(dead_features.sum()))
            auxk_latents = t.where(dead_features[None], post_relu_acts_BF, -t.inf)
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, dim=-1, sorted=False)
            auxk_buffer_BF = t.zeros_like(post_relu_acts_BF)
            auxk_acts_BF = auxk_buffer_BF.scatter_(-1, auxk_indices, auxk_acts)
            # Use decoder without adding b_dec separately
            x_reconstruct_aux = self.ae.decoder(auxk_acts_BF)
            l2_loss_aux = (residual_BD.float() - x_reconstruct_aux.float()).pow(2).sum(dim=-1).mean()
            self.pre_norm_auxk_loss = l2_loss_aux

            residual_mu = residual_BD.mean(dim=0, keepdim=True)
            loss_denom = (residual_BD.float() - residual_mu.float()).pow(2).sum(dim=-1).mean()
            normalized_auxk_loss = l2_loss_aux / loss_denom
            return normalized_auxk_loss.nan_to_num(0.0)
        else:
            self.pre_norm_auxk_loss = -1
            return t.tensor(0, dtype=residual_BD.dtype, device=residual_BD.device)

    def loss(self, x, step=None, logging=False):
        use_threshold = step > self.threshold_start_step if step is not None else False
        f, active_indices_F, post_relu_acts_BF = self.ae.encode(
            x, return_active=True, use_threshold=use_threshold
        )
        
        if step is not None and step > self.threshold_start_step:
            self.update_threshold(f)
        
        self.effective_l0 = (f != 0).float().sum(dim=-1).mean().item()

        x_hat = self.ae.decode(f)
        e = x - x_hat

        num_tokens_in_step = x.size(0)
        did_fire = t.zeros_like(self.num_tokens_since_fired, dtype=t.bool)
        did_fire[active_indices_F] = True
        self.num_tokens_since_fired += num_tokens_in_step
        self.num_tokens_since_fired[did_fire] = 0

        l2_loss = e.pow(2).sum(dim=-1).mean()
        loss = l2_loss
        # auxk_loss = self.get_auxiliary_loss(e.detach(), post_relu_acts_BF)
        # loss = l2_loss + self.auxk_alpha * auxk_loss

        if not logging:
            return loss
        else:
            return namedtuple("LossLog", ["x", "x_hat", "f", "losses"])(
                x,
                x_hat,
                f,
                {"l2_loss": l2_loss.item(), "loss": loss.item()},
            )

    def update(self, step, x):
        if step == 0:
            median = self.geometric_median(x)
            median = median.to(self.ae.b_dec.dtype)
            self.ae.b_dec.data = median

        x = x.to(self.device)
        loss = self.loss(x, step=step)
        loss.backward()

        self.ae.decoder.weight.grad = remove_gradient_parallel_to_decoder_directions(
            self.ae.decoder.weight,
            self.ae.decoder.weight.grad,
            self.ae.activation_dim,
            self.ae.dict_size,
        )
        t.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)

        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()

        self.ae.decoder.weight.data = set_decoder_norm_to_unit_norm(
            self.ae.decoder.weight, self.ae.activation_dim, self.ae.dict_size
        )

        return loss.item()

    @property
    def config(self):
        return {
            "trainer_class": "WTATrainer",
            "dict_class": "WTASAE",
            "lr": self.lr,
            "steps": self.steps,
            "auxk_alpha": self.auxk_alpha,
            "seed": self.seed,
            "activation_dim": self.ae.activation_dim,
            "dict_size": self.ae.dict_size,
            "sparsity_rate": self.ae.sparsity_rate,
            "device": self.device,
            "layer": self.layer,
            "lm_name": self.lm_name,
            "wandb_name": self.wandb_name,
            "submodule_name": self.submodule_name,
            "warmup_steps": self.warmup_steps,
            "decay_start": self.decay_start,
            "threshold_beta": self.threshold_beta,
            "threshold_start_step": self.threshold_start_step,
        }

    @staticmethod
    def geometric_median(points: t.Tensor, max_iter: int = 100, tol: float = 1e-5):
        guess = points.mean(dim=0)
        prev = t.zeros_like(guess)
        weights = t.ones(len(points), device=points.device)
        for _ in range(max_iter):
            prev = guess
            weights = 1 / t.norm(points - guess, dim=1)
            weights /= weights.sum()
            guess = (weights.unsqueeze(1) * points).sum(dim=0)
            if t.norm(guess - prev) < tol:
                break
        return guess

    def update_threshold(self, f: t.Tensor):
        device_type = "cuda" if f.is_cuda else "cpu"
        with t.autocast(device_type=device_type, enabled=False), t.no_grad():
            active = f[f > 0]
            
            if active.size(0) == 0:
                min_activation = 0.0
            else:
                min_activation = active.min().detach().to(dtype=t.float32)
                
            if self.ae.threshold < 0:
                self.ae.threshold = min_activation
            else:
                self.ae.threshold = (self.threshold_beta * self.ae.threshold) + (
                    (1 - self.threshold_beta) * min_activation
                )
