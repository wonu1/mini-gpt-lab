# -*- coding: utf-8 -*-
"""GPT 모델 구성 요소 과제 템플릿."""

import torch
import torch.nn as nn

try:
    from .attention import MultiHeadAttention
    from .embeddings import InputEmbedding
except ImportError:
    from attention import MultiHeadAttention
    from embeddings import InputEmbedding


class LayerNorm(nn.Module):
    """마지막 차원 기준 Layer Normalization."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        마지막 차원 기준으로 각 token 벡터를 정규화합니다.

        입력/출력 shape:
            (B, T, C) -> (B, T, C)
        """

        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)

        normalized = (x - mean) / torch.sqrt(variance + self.eps)

        return self.gamma * normalized + self.beta
        raise NotImplementedError("LayerNorm.forward를 구현하세요.")


class GELU(nn.Module):
    """GPT FeedForward에서 사용하는 GELU 활성화 함수."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO: tanh 근사식 또는 torch 연산으로 GELU를 구현합니다."""
        return torch.nn.functional.gelu(x)
        raise NotImplementedError("GELU.forward를 구현하세요.")


class FeedForward(nn.Module):
    """Transformer FFN: Linear -> GELU -> Linear -> Dropout."""

    def __init__(self, d_model: int, dropout: float = 0.1, mult: int = 4):
        super().__init__()
        # TODO: d_model -> mult*d_model -> d_model 구조의 작은 MLP를 정의하세요.
        hidden_dim = mult * d_model

        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            GELU(),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )
        # raise NotImplementedError("FeedForward.__init__을 구현하세요.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO: FeedForward 네트워크를 통과시킵니다."""
        return self.net(x)
        raise NotImplementedError("FeedForward.forward를 구현하세요.")


class TransformerBlock(nn.Module):
    """
    GPT block: LayerNorm -> Causal Self-Attention -> residual,
    LayerNorm -> FeedForward -> residual.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        drop_rate: float = 0.1,
        qkv_bias: bool = False,
    ):
        super().__init__()
        # TODO: attention, ffn, layernorm, dropout을 정의하세요.

        self.ln1 = LayerNorm(d_model)
        self.attention = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            drop_rate=drop_rate,
            qkv_bias=qkv_bias,
        )

        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout=drop_rate)

        self.dropout = nn.Dropout(drop_rate)

        # raise NotImplementedError("TransformerBlock.__init__을 구현하세요.")

    def forward(self, x: torch.Tensor, causal_mask: bool = True) -> torch.Tensor:
        """TODO: attention과 ffn을 residual connection으로 연결합니다."""

        """
        attention과 feed-forward를 residual connection으로 연결합니다.

        입력/출력 shape:
            (B, T, d_model) -> (B, T, d_model)
        """

        attention_out = self.attention(
            self.ln1(x),
            causal_mask=causal_mask,
        )
        x = x + self.dropout(attention_out)

        ffn_out = self.ffn(self.ln2(x))
        x = x + self.dropout(ffn_out)

        return x
    
        raise NotImplementedError("TransformerBlock.forward를 구현하세요.")


class GPTModel(nn.Module):
    """InputEmbedding -> TransformerBlock N개 -> LayerNorm -> LM head."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        # TODO: embedding, blocks, final layernorm, lm_head를 정의하세요.

        self.embedding = InputEmbedding(
            vocab_size=config["vocab_size"],
            emb_dim=config["emb_dim"],
            context_length=config["context_length"],
            drop_rate=config["drop_rate"],
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config["emb_dim"],
                    n_heads=config["n_heads"],
                    drop_rate=config["drop_rate"],
                    qkv_bias=config["qkv_bias"],
                )
                for _ in range(config["n_layers"])
            ]
        )

        self.final_norm = LayerNorm(config["emb_dim"])
        self.lm_head = nn.Linear(config["emb_dim"], config["vocab_size"])

        # raise NotImplementedError("GPTModel.__init__을 구현하세요.")

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: logits를 만들고, targets가 있으면 cross entropy loss도 함께 반환합니다.

        Returns:
            targets가 None이면 logits
            targets가 있으면 (loss, logits)
        """

        x = self.embedding(idx)

        for block in self.blocks:
            x = block(x, causal_mask=True)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits

        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
        )

        return loss, logits
    
        # raise NotImplementedError("GPTModel.forward를 구현하세요.")


def generate_text_simple(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
) -> torch.Tensor:
    """TODO: greedy 방식으로 max_new_tokens만큼 다음 토큰을 이어 붙입니다."""
    raise NotImplementedError("generate_text_simple을 구현하세요.")
