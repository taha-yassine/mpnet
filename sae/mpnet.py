import json
from fnmatch import fnmatch
from pathlib import Path
from typing import NamedTuple

import einops
import torch
from huggingface_hub import snapshot_download
from natsort import natsorted
from safetensors.torch import load_model, save_model
from torch import Tensor, nn

from .config import MpnetConfig
from .utils import decoder_impl


class EncoderOutput(NamedTuple):
    top_acts: Tensor
    """Activations of the top-k latents."""

    top_indices: Tensor
    """Indices of the top-k features."""


class ForwardOutput(NamedTuple):
    sae_out: Tensor

    latent_acts: Tensor
    """Activations of the top-k latents."""

    latent_indices: Tensor
    """Indices of the top-k features."""

    fvu: Tensor
    """Fraction of variance unexplained."""


class MPNet(nn.Module):
    def __init__(
        self,
        d_in: int,
        cfg: MpnetConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.d_in = d_in
        self.num_latents = cfg.num_latents or d_in * cfg.expansion_factor

        # Encoder and decoder weights
        if cfg.tie_iterations:
            self.encoder = nn.Linear(d_in, self.num_latents, device=device, dtype=dtype)
            self.encoder.bias.data.zero_()
        else:
            # TODO: Implement untied weights across iterations
            raise NotImplementedError("Untying weights across iterations not yet implemented")

        if cfg.tie_encoder_decoder:
            self.W_dec = self.encoder.weight
            self.b_dec = self.encoder.bias
        else:
            self.W_dec = nn.Parameter(self.encoder.weight.data.clone())
            self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))
            if cfg.normalize_decoder:
                self.set_decoder_norm_to_unit_norm()

    @staticmethod
    def load_many(
        name: str,
        local: bool = False,
        layers: list[str] | None = None,
        device: str | torch.device = "cpu",
        *,
        decoder: bool = True,
        pattern: str | None = None,
    ) -> dict[str, "MPNet"]:
        """Load SAEs for multiple hookpoints on a single model and dataset."""
        pattern = pattern + "/*" if pattern is not None else None
        if local:
            repo_path = Path(name)
        else:
            repo_path = Path(snapshot_download(name, allow_patterns=pattern))

        if layers is not None:
            return {
                layer: MPNet.load_from_disk(
                    repo_path / layer, device=device, decoder=decoder
                )
                for layer in natsorted(layers)
            }
        files = [
            f
            for f in repo_path.iterdir()
            if f.is_dir() and (pattern is None or fnmatch(f.name, pattern))
        ]
        return {
            f.name: MPNet.load_from_disk(f, device=device, decoder=decoder)
            for f in natsorted(files, key=lambda f: f.name)
        }

    @staticmethod
    def load_from_hub(
        name: str,
        hookpoint: str | None = None,
        device: str | torch.device = "cpu",
        *,
        decoder: bool = True,
    ) -> "MPNet":
        # Download from the HuggingFace Hub
        repo_path = Path(
            snapshot_download(
                name,
                allow_patterns=f"{hookpoint}/*" if hookpoint is not None else None,
            )
        )
        if hookpoint is not None:
            repo_path = repo_path / hookpoint

        # No layer specified, and there are multiple layers
        elif not repo_path.joinpath("cfg.json").exists():
            raise FileNotFoundError("No config file found; try specifying a layer.")

        return MPNet.load_from_disk(repo_path, device=device, decoder=decoder)

    @staticmethod
    def load_from_disk(
        path: Path | str,
        device: str | torch.device = "cpu",
        *,
        decoder: bool = True,
    ) -> "MPNet":
        path = Path(path)

        with open(path / "cfg.json", "r") as f:
            cfg_dict = json.load(f)
            d_in = cfg_dict.pop("d_in")
            cfg = MpnetConfig.from_dict(cfg_dict, drop_extra_fields=True)

        sae = MPNet(d_in, cfg, device=device, decoder=decoder)
        load_model(
            model=sae,
            filename=str(path / "mpnet.safetensors"),
            device=str(device),
            # TODO: Maybe be more fine-grained about this in the future?
            strict=decoder,
        )
        return sae

    def save_to_disk(self, path: Path | str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        save_model(self, str(path / "mpnet.safetensors"))
        with open(path / "cfg.json", "w") as f:
            json.dump(
                {
                    **self.cfg.to_dict(),
                    "d_in": self.d_in,
                },
                f,
            )

    @property
    def device(self):
        return self.encoder.weight.device

    @property
    def dtype(self):
        return self.encoder.weight.dtype

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode the input by iteratively selecting the top-k latents."""
        residual = (x - self.b_dec).clone()

        accumulated_acts = list()
        accumulated_indices = list()
        
        for i in range(self.cfg.k):
            # Encode
            z = self.encoder(residual)
            top_act, top_indice = z.topk(k=1, sorted=False)
            top_act = nn.functional.relu(top_act)  # Apply post-TopK activation

            accumulated_acts.append(top_act)
            accumulated_indices.append(top_indice)
            
            # Decode
            if i != self.cfg.k - 1: 
                if self.W_dec is not None:
                    decoded = self.decode(top_act, top_indice)
                    residual = residual - decoded

        return EncoderOutput(
            top_acts=torch.stack(accumulated_acts),
            top_indices=torch.stack(accumulated_indices),
        )

    def decode(self, top_acts: Tensor, top_indices: Tensor) -> Tensor:
        assert self.W_dec is not None, "Decoder weight was not initialized."

        y = decoder_impl(top_indices, top_acts.to(self.dtype), self.W_dec.mT)
        return y

    def forward(
        self, x: Tensor
    ) -> ForwardOutput:
        # Encode
        top_acts, top_indices = self.encode(x)

        # Decode
        sae_out = self.decode(top_acts, top_indices) + self.b_dec
        
        # Compute residual error
        e = sae_out - x
        
        # Used as a denominator for putting everything on a reasonable scale
        total_variance = (x - x.mean(0)).pow(2).sum()

        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        return ForwardOutput(
            sae_out,
            top_acts,
            top_indices,
            fvu,
        )

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        assert self.W_dec is not None, "Decoder weight was not initialized."

        eps = torch.finfo(self.W_dec.dtype).eps
        norm = torch.norm(self.W_dec.data, dim=1, keepdim=True)
        self.W_dec.data /= norm + eps

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        assert self.W_dec is not None, "Decoder weight was not initialized."
        assert self.W_dec.grad is not None  # keep pyright happy

        parallel_component = einops.einsum(
            self.W_dec.grad,
            self.W_dec.data,
            "d_sae d_in, d_sae d_in -> d_sae",
        )
        self.W_dec.grad -= einops.einsum(
            parallel_component,
            self.W_dec.data,
            "d_sae, d_sae d_in -> d_sae d_in",
        )
