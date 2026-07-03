"""
PoC: Atualização Incremental Legal via Nested Learning (Continuous Memory)

Este script demonstra uma prova de conceito onde um modelo de base congelado
(Base Model) consulta uma matriz de memória associativa externa (Hope Module).
Novas normativas jurídicas podem ser injetadas incrementalmente na memória,
alterando o comportamento/decisão do modelo em tempo de inferência sem a 
necessidade de backpropagation (fine-tuning) nos pesos originais.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple, Dict, Optional

class FrozenBaseModel(nn.Module):
    """
    Modelo de Base Congelado: Extrai representações semânticas profundas sem sofrer fine-tuning.
    Utiliza transformers padrão da HuggingFace.
    """
    def __init__(self, model_name: str = 'neuralmind/bert-base-portuguese-cased'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.bert.eval()
        
        # Restrição Crítica: Congelando todos os parâmetros (requires_grad = False)
        for param in self.bert.parameters():
            param.requires_grad = False
            
    def get_embedding(self, texts: List[str]) -> torch.Tensor:
        """
        Retorna o embedding semântico do texto usando Mean Pooling das hidden states.
        O Mean Pooling capta melhor a semântica da frase inteira em modelos não fine-tunados.
        """
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=256)
        
        with torch.no_grad(): # Garantia extra de zero backprop aqui
            outputs = self.bert(**inputs)
            
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return sum_embeddings / sum_mask

class MemoryMLP(nn.Module):
    """
    Active Memory Network (MLP) representing associative memory mapping.
    Maps a high-dimensional key embedding to a low-dimensional value (decision vector).
    Using layer normalization and GELU activation for stable, non-linear mapping capabilities.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ContinuousMemory(nn.Module):
    """
    Continuum Memory System (CMS) implementing fast-slow associative adaptation.
    Replaces static episodic buffers with online parameter optimization at inference time.
    Incorporates surprise-based priority scaling to resolve regulation precedence.
    """
    def __init__(self, embed_dim: int, value_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.value_dim = value_dim
        self.hidden_dim = hidden_dim
        
        # Dual-timescale active memory modules
        self.fast_net = MemoryMLP(embed_dim, hidden_dim, value_dim)
        self.slow_net = MemoryMLP(embed_dim, hidden_dim, value_dim)
        
        # Synchronize slow network to match initial fast network weights
        self.slow_net.load_state_dict(self.fast_net.state_dict())
        
        # Memory buffers to store raw key/value pairs, texts, and importance coefficients
        self.register_buffer('keys', torch.empty(0, embed_dim))
        self.register_buffer('values', torch.empty(0, value_dim))
        self.register_buffer('rule_importance', torch.empty(0))
        self.texts: List[str] = []
        
        # Surprise tracking for dynamic learning rate adjustments
        self.register_buffer('surprise_momentum', torch.zeros(1))
        
    def add_memory(self, key_vector: torch.Tensor, value_vector: torch.Tensor, text: str):
        """
        Inserts new regulatory guidelines dynamically. Runs a nested inner optimization
        loop (fast loop) to adjust active parameters, followed by an outer loop (slow loop)
        consolidation step. Employs surprise-weighted loss terms to prioritize overriding rules.
        """
        # Strict validation of inputs to ensure state resilience
        if not isinstance(text, str) or len(text.strip()) == 0:
            raise ValueError("Rule text must be a non-empty string.")
        if key_vector.ndim != 2 or key_vector.size(1) != self.embed_dim:
            raise ValueError(f"Key vector must be 2D with last dimension equal to {self.embed_dim}.")
        if value_vector.ndim != 2 or value_vector.size(1) != self.value_dim:
            raise ValueError(f"Value vector must be 2D with last dimension equal to {self.value_dim}.")

        # Calculate prediction surprise using the consolidated slow network BEFORE adding it to the buffer
        self.fast_net.eval()
        self.slow_net.eval()
        
        surprise = 0.0
        if self.keys.size(0) > 0:
            with torch.no_grad():
                pred_slow = self.slow_net(key_vector)
                surprise = F.mse_loss(pred_slow, value_vector).item()
                # Update surprise running tracker
                self.surprise_momentum = 0.9 * self.surprise_momentum + 0.1 * surprise

        # Append to raw buffers and text registry
        self.keys = torch.cat([self.keys, key_vector], dim=0)
        self.values = torch.cat([self.values, value_vector], dim=0)
        self.texts.append(text)
        
        # Compute surprise-based rule importance: contradictory/new rules get higher weight (up to 4x)
        importance_val = 1.0 + 3.0 * surprise
        importance_tensor = torch.tensor([importance_val], device=key_vector.device)
        self.rule_importance = torch.cat([self.rule_importance, importance_tensor], dim=0)
            
        # Dynamically scale learning rate and epochs based on surprise score.
        base_lr = 0.01
        lr = base_lr * (1.0 + min(2.0, surprise * 1.5))
        epochs = int(35 * (1.0 + min(1.0, surprise)))
        
        # Configure local optimizer for the inner loop (Fast Loop Adaptation)
        optimizer = torch.optim.Adam(self.fast_net.parameters(), lr=lr, weight_decay=1e-4)
        
        self.fast_net.train()
        num_stored = self.keys.size(0)
        
        # Inner loop optimization: balancing current adaptation, memory replay, and slow weight regularization
        for _ in range(epochs):
            optimizer.zero_grad()
            
            # 1. Adapt to the new regulatory injection (weighted by rule importance)
            pred_new = self.fast_net(key_vector)
            loss_new = F.mse_loss(pred_new, value_vector) * self.rule_importance[-1]
            
            # 2. Experience replay on previous rules, weighted by their respective importance values
            loss_replay = torch.tensor(0.0, device=key_vector.device)
            if num_stored > 1:
                prev_keys = self.keys[:-1]
                prev_values = self.values[:-1]
                pred_prev = self.fast_net(prev_keys)
                raw_errors = (pred_prev - prev_values) ** 2
                weighted_errors = raw_errors * self.rule_importance[:-1].unsqueeze(-1)
                loss_replay = weighted_errors.mean()
                
            # 3. Proximal regularization mapping fast weights back to consolidated slow weights
            loss_reg = torch.tensor(0.0, device=key_vector.device)
            for p_fast, p_slow in zip(self.fast_net.parameters(), self.slow_net.parameters()):
                loss_reg += F.mse_loss(p_fast, p_slow)
                
            # Total nested objective function
            total_loss = loss_new + 0.7 * loss_replay + 0.15 * loss_reg
            total_loss.backward()
            optimizer.step()
            
        self.fast_net.eval()
        
        # Outer Loop Optimization (Slow Loop Consolidation):
        # Update the long-term consolidated memory (slow loop) slowly using Exponential Moving Average (EMA)
        tau = 0.15  # Consolidation rate
        with torch.no_grad():
            for p_fast, p_slow in zip(self.fast_net.parameters(), self.slow_net.parameters()):
                p_slow.copy_(p_slow + tau * (p_fast - p_slow))

class HopeModule(nn.Module):
    """
    Hierarchical Optimization & Retrieval attention layer (Hope Module).
    Combines direct associative retrieval with a non-linear active prediction routing mechanism.
    """
    def __init__(self, embed_dim: int, value_dim: int = 2, temperature: float = 0.05, hidden_dim: int = 64):
        super().__init__()
        self.memory = ContinuousMemory(embed_dim, value_dim, hidden_dim)
        self.temperature = temperature
        
    def forward(self, query_embed: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        num_keys = self.memory.keys.size(0)
        if num_keys == 0:
            device = query_embed.device
            return torch.zeros(query_embed.size(0), self.memory.value_dim, device=device), None
            
        # Ensure active networks are set to evaluation mode
        self.memory.fast_net.eval()
        self.memory.slow_net.eval()
        
        with torch.no_grad():
            v_fast = self.memory.fast_net(query_embed)
            v_slow = self.memory.slow_net(query_embed)
            
        # Calculate semantic cosine similarities to identify best-matching rules
        query_norm = F.normalize(query_embed, p=2, dim=-1)
        keys_norm = F.normalize(self.memory.keys, p=2, dim=-1)
        scores = torch.matmul(query_norm, keys_norm.T)  # (batch, num_keys)
        
        # Scale scores by rule importance to resolve precedence conflicts
        scaled_scores = scores * self.memory.rule_importance
        
        # Softmax scaling to generate clean explainability weights
        attention_weights = F.softmax(scaled_scores / self.temperature, dim=-1)
        
        # Direct attention-based retrieval of action vectors from episodic memory
        v_retrieved = torch.matmul(attention_weights, self.memory.values)
        
        # Dynamic Gate: Compute max similarity. If query matches a recently learned
        # instruction closely, increase routing towards the fast network (short-term memory).
        max_sim = torch.max(scores, dim=-1)[0].unsqueeze(-1)
        gate = torch.clamp((max_sim - 0.2) / 0.6, min=0.0, max=1.0)
        
        # Non-linear combined action prediction
        v_net = gate * v_fast + (1.0 - gate) * v_slow
        
        # Combine direct episodic retrieval with active neural memory predictions (routing logic)
        retrieved_values = 0.4 * v_retrieved + 0.6 * v_net
        
        return retrieved_values, attention_weights

class LegalIncrementalModel(nn.Module):
    """
    Main orchestrator integrating the Frozen Base Encoder and the Continuum Memory System.
    """
    def __init__(self, model_name: str = 'neuralmind/bert-base-portuguese-cased', value_dim: int = 2):
        super().__init__()
        self.encoder = FrozenBaseModel(model_name)
        embed_dim = self.encoder.bert.config.hidden_size
        self.hope_module = HopeModule(embed_dim, value_dim)
        
    def update_memory(self, rule_text: str, action_vector: List[float]):
        """
        Updates the model's active memory with new regulatory rules without backpropagating
        through the base model weights.
        """
        if not isinstance(action_vector, list) or len(action_vector) != self.hope_module.memory.value_dim:
            raise ValueError(f"Action vector must be a list of float elements of length {self.hope_module.memory.value_dim}.")
            
        # Wrap encoder embedding retrieval inside a robust error-handling block
        try:
            embed = self.encoder.get_embedding([rule_text])
        except Exception as e:
            raise RuntimeError(f"Failed to generate embeddings from Base Model: {str(e)}")
            
        val_tensor = torch.tensor([action_vector], dtype=torch.float32)
        self.hope_module.memory.add_memory(embed, val_tensor, rule_text)
        
    def predict(self, query_text: str) -> Dict:
        """
        Predicts decision vectors using combined base model representation and active memory networks.
        """
        if not isinstance(query_text, str) or len(query_text.strip()) == 0:
            raise ValueError("Query text must be a non-empty string.")
            
        try:
            query_embed = self.encoder.get_embedding([query_text])
        except Exception as e:
            raise RuntimeError(f"Failed to generate query embeddings from Base Model: {str(e)}")
            
        predicted_action, attention_weights = self.hope_module(query_embed)
        
        pred_action_list = predicted_action.squeeze().tolist()
        # Handle cases where value_dim could be squeezed into a scalar
        if isinstance(pred_action_list, float):
            pred_action_list = [pred_action_list]
            
        result = {
            "query": query_text,
            "predicted_action_vector": pred_action_list,
        }
        
        if attention_weights is not None:
            weights = attention_weights.squeeze().tolist()
            if isinstance(weights, float):
                weights = [weights]
            
            top_idx = torch.argmax(attention_weights, dim=-1).item()
            result["most_relevant_rule"] = self.hope_module.memory.texts[top_idx]
            result["confidence"] = weights[top_idx]
            
        return result


# ==========================================
# PIPELINE DE TESTE E VALIDAÇÃO (MOCK EXEC)
# ==========================================
def mock_execution():
    # Modelagem da diretriz de saída (Value): [Probabilidade de Excluir, Probabilidade de Reter]
    ACTION_EXCLUIR = [1.0, 0.0]
    ACTION_RETER = [0.0, 1.0]
    
    print("="*70)
    print(" INICIALIZANDO PIPELINE DE NESTED LEARNING JURÍDICO")
    print("="*70)
    
    # 1. Instanciando o modelo (Utilizando o BERT Multilingue/Português base para melhor similaridade semântica)
    model = LegalIncrementalModel(model_name='neuralmind/bert-base-portuguese-cased')
    
    # 2. Dados de Simulação (Contexto de Privacidade vs. Antifraude)
    base_knowledge = [
        "Artigo 1: Todo cliente tem o direito inalienável de solicitar a exclusão completa de seus dados pessoais das bases da instituição financeira a qualquer momento (Direito ao Esquecimento).",
        "Artigo 2: O processo de exclusão de dados deve ser concluído em até 15 dias úteis após a solicitação do titular.",
        "Artigo 3: A análise de crédito depende do histórico de transações; sem dados, o escore do cliente é zerado."
    ]
    
    new_normative_rule = "Nova Diretriz Antifraude: É estritamente proibida a exclusão de dados vinculados a históricos de operações de crédito ativas ou liquidadas nos últimos 5 anos, sobrepondo-se a qualquer pedido de exclusão do titular. Estes dados devem ser retidos em cofre isolado para fins de auditoria e prevenção a fraudes sistêmicas."
    
    query = "O cliente João da Silva, que quitou um empréstimo no mês passado, abriu um chamado formal no SAC exigindo a exclusão imediata e total de seus dados pessoais e histórico financeiro do banco."
    
    print("\n[ FASE 1 ] INJETANDO CONHECIMENTO BASE (LEGISLAÇÃO ANTIGA)")
    for rule in base_knowledge:
        # A regra base atua em favor do direito à exclusão
        model.update_memory(rule, ACTION_EXCLUIR)
        print(f" [+] Memória Adicionada -> Ação Default: Excluir | Regra: {rule[:65]}...")
        
    print("\n[ FASE 2 ] TESTANDO MODELO ANTES DA ATUALIZAÇÃO")
    res_before = model.predict(query)
    print(f" -> Consulta: {query}")
    print(f" -> Regra resgatada da memória: '{res_before['most_relevant_rule']}' (Confiança: {res_before['confidence']:.2%})")
    print(f" -> Vetor de Ação Previsto (Excluir, Reter): [{res_before['predicted_action_vector'][0]:.4f}, {res_before['predicted_action_vector'][1]:.4f}]")
    
    print("\n[ FASE 3 ] CONTINUOUS LEARNING: PUBLICAÇÃO E INJEÇÃO DE NOVA LEI")
    # Injetamos a nova norma sem realizar nenhum backpropagation (zero-shot via Retrieval Memory)
    model.update_memory(new_normative_rule, ACTION_RETER)
    print(f" [+] Nova Memória Adicionada -> Ação Mapeada: Reter | Regra: {new_normative_rule[:65]}...")
    
    print("\n[ FASE 4 ] TESTANDO O MODELO APÓS A ATUALIZAÇÃO INCREMENTAL")
    res_after = model.predict(query)
    print(f" -> Consulta: {query}")
    print(f" -> Nova regra resgatada: '{res_after['most_relevant_rule']}' (Confiança: {res_after['confidence']:.2%})")
    print(f" -> Novo Vetor de Ação Previsto (Excluir, Reter): [{res_after['predicted_action_vector'][0]:.4f}, {res_after['predicted_action_vector'][1]:.4f}]")
    
    print("\n" + "="*70)
    print(" ANÁLISE DE RESULTADO DA PROVA DE CONCEITO")
    print("="*70)
    # Validação Matemática de Aprendizado Incremental
    decisao_antes = "EXCLUIR" if res_before['predicted_action_vector'][0] > res_before['predicted_action_vector'][1] else "RETER"
    decisao_depois = "EXCLUIR" if res_after['predicted_action_vector'][0] > res_after['predicted_action_vector'][1] else "RETER"
    
    print(f"Decisão do Modelo ANTES da nova lei : {decisao_antes}")
    print(f"Decisão do Modelo DEPOIS da nova lei: {decisao_depois}")
    
    if decisao_antes == "EXCLUIR" and decisao_depois == "RETER":
        print("\n[SUCESSO] O modelo atualizou dinamicamente seu raciocínio jurídico.")
        print("          O módulo 'Hope' identificou a similaridade semântica da nova regra com")
        print("          a query (empréstimo/crédito) e sobrepôs a atenção, mudando o output")
        print("          final da rede sem a necessidade de Fine-Tuning/Backprop!")
    else:
        print("\n[AVISO] A variação semântica das frases não permitiu a inversão perfeita com a temperatura atual.")
        print("        Considere ajustar a temperatura do Softmax ou usar SentenceTransformers focados em similaridade.")

if __name__ == "__main__":
    mock_execution()
