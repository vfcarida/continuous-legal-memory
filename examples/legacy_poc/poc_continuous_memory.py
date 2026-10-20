"""
Proof of Concept: Incremental Legal Normative Adaptation via Continuous Memory.

This self-contained script demonstrates the foundational research concept behind
Continuous Legal Memory: a frozen base encoder paired with an external associative
memory module (`HopeModule`) and dual-timescale neural heads (`fast_net`, `slow_net`).
New legal norms are ingested online without backpropagating into the base transformer.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class FrozenBaseModel(nn.Module):
    """
    Frozen Base Encoder: Extracts semantic representations without fine-tuning.
    Uses standard Hugging Face transformers with all parameters frozen.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.bert.eval()

        # Critical invariant: freeze all base encoder parameters (requires_grad = False)
        for param in self.bert.parameters():
            param.requires_grad = False

    def get_embedding(self, texts: list[str]) -> torch.Tensor:
        """
        Return the semantic embedding of input texts using attention-masked mean pooling.
        """
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )

        with torch.no_grad():
            outputs = self.bert(**inputs)

        attention_mask = inputs["attention_mask"]
        token_embeddings = outputs.last_hidden_state

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return sum_embeddings / sum_mask


class MemoryMLP(nn.Module):
    """
    Active Memory Network (MLP) representing associative memory mapping.
    Maps a high-dimensional key embedding to a low-dimensional decision vector.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ContinuousMemory(nn.Module):
    """
    Continuum Memory System (CMS) implementing fast-slow associative adaptation.
    Combines an episodic key-value buffer with surprise-scaled online parameter adaptation.
    """

    keys: torch.Tensor
    values: torch.Tensor
    rule_importance: torch.Tensor
    surprise_momentum: torch.Tensor

    def __init__(self, embed_dim: int, value_dim: int = 2, hidden_dim: int = 64) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.value_dim = value_dim
        self.hidden_dim = hidden_dim

        self.fast_net = MemoryMLP(embed_dim, hidden_dim, value_dim)
        self.slow_net = MemoryMLP(embed_dim, hidden_dim, value_dim)
        self.slow_net.load_state_dict(self.fast_net.state_dict())

        self.register_buffer("keys", torch.empty(0, embed_dim))
        self.register_buffer("values", torch.empty(0, value_dim))
        self.register_buffer("rule_importance", torch.empty(0))
        self.texts: list[str] = []
        self.register_buffer("surprise_momentum", torch.zeros(1))

    def add_memory(self, key_vector: torch.Tensor, value_vector: torch.Tensor, text: str) -> None:
        """
        Insert a new regulatory directive dynamically using fast-loop adaptation and
        slow-loop Exponential Moving Average (EMA) consolidation.
        """
        if not isinstance(text, str) or len(text.strip()) == 0:
            raise ValueError("Rule text must be a non-empty string.")
        if key_vector.ndim != 2 or key_vector.size(1) != self.embed_dim:
            raise ValueError(f"Key vector must be 2D with last dimension equal to {self.embed_dim}.")
        if value_vector.ndim != 2 or value_vector.size(1) != self.value_dim:
            raise ValueError(f"Value vector must be 2D with last dimension equal to {self.value_dim}.")

        self.fast_net.eval()
        self.slow_net.eval()

        surprise = 0.0
        if self.keys.size(0) > 0:
            with torch.no_grad():
                pred_slow = self.slow_net(key_vector)
                surprise = F.mse_loss(pred_slow, value_vector).item()
                self.surprise_momentum = 0.9 * self.surprise_momentum + 0.1 * surprise

        self.keys = torch.cat([self.keys, key_vector], dim=0)
        self.values = torch.cat([self.values, value_vector], dim=0)
        self.texts.append(text)

        importance_val = 1.0 + 3.0 * surprise
        importance_tensor = torch.tensor([importance_val], device=key_vector.device)
        self.rule_importance = torch.cat([self.rule_importance, importance_tensor], dim=0)

        base_lr = 0.01
        lr = base_lr * (1.0 + min(2.0, surprise * 1.5))
        epochs = int(35 * (1.0 + min(1.0, surprise)))

        optimizer = torch.optim.Adam(self.fast_net.parameters(), lr=lr, weight_decay=1e-4)

        self.fast_net.train()
        num_stored = self.keys.size(0)

        for _ in range(epochs):
            optimizer.zero_grad()

            pred_new = self.fast_net(key_vector)
            loss_new = F.mse_loss(pred_new, value_vector) * self.rule_importance[-1]

            loss_replay = torch.tensor(0.0, device=key_vector.device)
            if num_stored > 1:
                prev_keys = self.keys[:-1]
                prev_values = self.values[:-1]
                pred_prev = self.fast_net(prev_keys)
                raw_errors = (pred_prev - prev_values) ** 2
                weighted_errors = raw_errors * self.rule_importance[:-1].unsqueeze(-1)
                loss_replay = weighted_errors.mean()

            loss_reg = torch.tensor(0.0, device=key_vector.device)
            for p_fast, p_slow in zip(self.fast_net.parameters(), self.slow_net.parameters()):
                loss_reg += F.mse_loss(p_fast, p_slow)

            total_loss = loss_new + 0.7 * loss_replay + 0.15 * loss_reg
            total_loss.backward()
            optimizer.step()

        self.fast_net.eval()

        tau = 0.15
        with torch.no_grad():
            for p_fast, p_slow in zip(self.fast_net.parameters(), self.slow_net.parameters()):
                p_slow.copy_(p_slow + tau * (p_fast - p_slow))


class HopeModule(nn.Module):
    """
    Hierarchical Optimization & Retrieval attention layer (Hope Module).
    Combines direct associative retrieval with non-linear fast-slow prediction routing.
    """

    def __init__(
        self,
        embed_dim: int,
        value_dim: int = 2,
        temperature: float = 0.05,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.memory = ContinuousMemory(embed_dim, value_dim, hidden_dim)
        self.temperature = temperature

    def forward(self, query_embed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        num_keys = self.memory.keys.size(0)
        if num_keys == 0:
            device = query_embed.device
            return torch.zeros(query_embed.size(0), self.memory.value_dim, device=device), None

        self.memory.fast_net.eval()
        self.memory.slow_net.eval()

        with torch.no_grad():
            v_fast = self.memory.fast_net(query_embed)
            v_slow = self.memory.slow_net(query_embed)

        query_norm = F.normalize(query_embed, p=2, dim=-1)
        keys_norm = F.normalize(self.memory.keys, p=2, dim=-1)
        scores = torch.matmul(query_norm, keys_norm.T)

        scaled_scores = scores * self.memory.rule_importance
        attention_weights = F.softmax(scaled_scores / self.temperature, dim=-1)
        v_retrieved = torch.matmul(attention_weights, self.memory.values)

        max_sim = torch.max(scores, dim=-1)[0].unsqueeze(-1)
        gate = torch.clamp((max_sim - 0.2) / 0.6, min=0.0, max=1.0)
        v_net = gate * v_fast + (1.0 - gate) * v_slow

        retrieved_values = 0.4 * v_retrieved + 0.6 * v_net
        return retrieved_values, attention_weights


class LegalIncrementalModel(nn.Module):
    """
    Self-contained PoC orchestrator integrating the Frozen Base Encoder and Continuum Memory System.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", value_dim: int = 2) -> None:
        super().__init__()
        self.encoder = FrozenBaseModel(model_name)
        embed_dim = self.encoder.bert.config.hidden_size
        self.hope_module = HopeModule(embed_dim, value_dim)

    def update_memory(self, rule_text: str, action_vector: list[float]) -> None:
        """
        Update active memory with a new regulatory rule without backpropagating into the base encoder.
        """
        if not isinstance(action_vector, list) or len(action_vector) != self.hope_module.memory.value_dim:
            raise ValueError(
                f"Action vector must be a list of floats of length {self.hope_module.memory.value_dim}."
            )

        try:
            embed = self.encoder.get_embedding([rule_text])
        except Exception as err:
            raise RuntimeError(f"Failed to generate embeddings from Base Model: {err}") from err

        val_tensor = torch.tensor([action_vector], dtype=torch.float32)
        self.hope_module.memory.add_memory(embed, val_tensor, rule_text)

    def predict(self, query_text: str) -> dict[str, Any]:
        """
        Predict decision vectors using combined base encoder representation and active memory networks.
        """
        if not isinstance(query_text, str) or len(query_text.strip()) == 0:
            raise ValueError("Query text must be a non-empty string.")

        try:
            query_embed = self.encoder.get_embedding([query_text])
        except Exception as err:
            raise RuntimeError(f"Failed to generate query embeddings from Base Model: {err}") from err

        predicted_action, attention_weights = self.hope_module(query_embed)

        pred_action_list = predicted_action.squeeze().tolist()
        if isinstance(pred_action_list, float):
            pred_action_list = [pred_action_list]

        result: dict[str, Any] = {
            "query": query_text,
            "predicted_action_vector": pred_action_list,
        }

        if attention_weights is not None:
            weights = attention_weights.squeeze().tolist()
            if isinstance(weights, float):
                weights = [weights]

            top_idx = int(torch.argmax(attention_weights, dim=-1).item())
            result["most_relevant_rule"] = self.hope_module.memory.texts[top_idx]
            result["confidence"] = weights[top_idx]

        return result


def mock_execution() -> None:
    """Execute the standalone proof-of-concept simulation."""
    # Output decision representation: [Delete Probability, Retain Probability]
    action_delete = [1.0, 0.0]
    action_retain = [0.0, 1.0]

    print("=" * 70)
    print(" INITIALIZING CONTINUOUS LEGAL MEMORY PROOF-OF-CONCEPT PIPELINE")
    print("=" * 70)

    model = LegalIncrementalModel(model_name="BAAI/bge-small-en-v1.5")

    base_knowledge = [
        "Article 1: Every client has the right to request complete deletion of personal data from financial institution databases at any time (Right to Erasure).",
        "Article 2: Personal data deletion requests must be completed within 15 business days of submission.",
        "Article 3: Credit risk evaluation relies on historical transaction records; without records, the credit score is reset.",
    ]

    new_normative_rule = (
        "Anti-Fraud Directive: It is strictly prohibited to delete data linked to credit operations "
        "active or settled within the past 5 years, overriding any data subject erasure request. "
        "Such records must be retained in an isolated audit vault for systemic fraud prevention."
    )

    query = (
        "Client John Smith, who paid off a personal loan last month, submitted a formal request "
        "demanding immediate and complete deletion of his personal data and financial history."
    )

    print("\n[ STAGE 1 ] INGESTING BASELINE STATUTES")
    for rule in base_knowledge:
        model.update_memory(rule, action_delete)
        print(f" [+] Memory Added -> Default Action: Delete | Rule: {rule[:65]}...")

    print("\n[ STAGE 2 ] EVALUATING QUERY BEFORE NORMATIVE OVERRIDE")
    res_before = model.predict(query)
    print(f" -> Query: {query}")
    print(
        f" -> Retrieved Rule: '{res_before['most_relevant_rule']}' "
        f"(Confidence: {res_before['confidence']:.2%})"
    )
    print(
        f" -> Predicted Action Vector (Delete, Retain): "
        f"[{res_before['predicted_action_vector'][0]:.4f}, {res_before['predicted_action_vector'][1]:.4f}]"
    )

    print("\n[ STAGE 3 ] CONTINUOUS ADAPTATION: INJECTING ANTI-FRAUD DIRECTIVE")
    model.update_memory(new_normative_rule, action_retain)
    print(f" [+] New Memory Added -> Mapped Action: Retain | Rule: {new_normative_rule[:65]}...")

    print("\n[ STAGE 4 ] EVALUATING QUERY AFTER INCREMENTAL ADAPTATION")
    res_after = model.predict(query)
    print(f" -> Query: {query}")
    print(
        f" -> Newly Retrieved Rule: '{res_after['most_relevant_rule']}' "
        f"(Confidence: {res_after['confidence']:.2%})"
    )
    print(
        f" -> Updated Action Vector (Delete, Retain): "
        f"[{res_after['predicted_action_vector'][0]:.4f}, {res_after['predicted_action_vector'][1]:.4f}]"
    )

    print("\n" + "=" * 70)
    print(" PROOF-OF-CONCEPT VERIFICATION SUMMARY")
    print("=" * 70)
    decision_before = (
        "DELETE"
        if res_before["predicted_action_vector"][0] > res_before["predicted_action_vector"][1]
        else "RETAIN"
    )
    decision_after = (
        "DELETE"
        if res_after["predicted_action_vector"][0] > res_after["predicted_action_vector"][1]
        else "RETAIN"
    )

    print(f"Model Decision BEFORE new directive : {decision_before}")
    print(f"Model Decision AFTER new directive  : {decision_after}")

    if decision_before == "DELETE" and decision_after == "RETAIN":
        print("\n[SUCCESS] The model dynamically updated its legal decision output without base encoder fine-tuning.")
    else:
        print("\n[WARNING] Semantic variation did not invert the decision under the current softmax temperature.")


if __name__ == "__main__":
    mock_execution()
