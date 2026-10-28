import torch
from torch import nn

try:
    from speechbrain.inference.classifiers import EncoderClassifier
except ImportError:
    from speechbrain.pretrained import EncoderClassifier


class SpeakerEmbeddingExtractor(nn.Module):
    """Load a SpeechBrain ECAPA-TDNN model from Hugging Face and encode waveforms."""

    def __init__(self, model_id: str, device: str | torch.device = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.encoder = EncoderClassifier.from_hparams(
            source=model_id,
            run_opts={"device": str(self.device)},
        )
        self.eval()

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        if waveforms.ndim == 1:
            waveforms = waveforms.unsqueeze(0)
        if waveforms.ndim != 2:
            raise ValueError(
                f"waveforms must have shape [batch, samples], got {tuple(waveforms.shape)}"
            )

        embeddings = self.encoder.encode_batch(
            waveforms.to(device=self.device, dtype=torch.float32)
        )
        if embeddings.ndim == 3 and embeddings.shape[1] == 1:
            embeddings = embeddings.squeeze(1)
        if embeddings.ndim != 2:
            raise RuntimeError(
                f"Expected embeddings with shape [batch, dim], got {tuple(embeddings.shape)}"
            )
        return embeddings