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

class ContinuousMemory(nn.Module):
    """
    Módulo de Memória Contínua (Associative Memory).
    Mantém uma matriz crescente de Keys (embeddings de regras) e Values (vetores de diretrizes/ações).
    """
    def __init__(self, embed_dim: int, value_dim: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.value_dim = value_dim
        
        # Memória inicialmente vazia. Vai crescendo dinamicamente.
        self.register_buffer('keys', torch.empty(0, embed_dim))
        self.register_buffer('values', torch.empty(0, value_dim))
        self.texts: List[str] = []
        
    def add_memory(self, key_vector: torch.Tensor, value_vector: torch.Tensor, text: str):
        """
        Anexa a representação da nova norma incrementalmente à matriz.
        """
        self.keys = torch.cat([self.keys, key_vector], dim=0)
        self.values = torch.cat([self.values, value_vector], dim=0)
        self.texts.append(text)

class HopeModule(nn.Module):
    """
    Camada Personalizada de Atenção (Nested Learning / Retrieval-Augmented Attention).
    Dada uma consulta, gera a "Query" e atende à matriz de memória.
    """
    def __init__(self, embed_dim: int, value_dim: int = 2, temperature: float = 0.05):
        super().__init__()
        self.memory = ContinuousMemory(embed_dim, value_dim)
        # Temperatura baixa para forçar a atenção (Softmax) a ser mais incisiva na regra mais similar
        self.temperature = temperature
        
    def forward(self, query_embed: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.memory.keys.size(0) == 0:
            # Caso base (sem memória): Vetor zerado (sem diretriz recuperada)
            device = query_embed.device
            return torch.zeros(query_embed.size(0), self.memory.value_dim).to(device), None
            
        # Calcula similaridade de cosseno para estabilidade de busca
        query_norm = F.normalize(query_embed, p=2, dim=-1)
        keys_norm = F.normalize(self.memory.keys, p=2, dim=-1)
        
        # Produto escalar (Atenção) entre Query e Keys
        scores = torch.matmul(query_norm, keys_norm.T) 
        
        # Softmax com temperature scaling para afiar as probabilidades
        attention_weights = F.softmax(scores / self.temperature, dim=-1)
        
        # Combina dinamicamente o conteúdo recuperado (Values)
        retrieved_values = torch.matmul(attention_weights, self.memory.values)
        
        return retrieved_values, attention_weights

class LegalIncrementalModel(nn.Module):
    """
    Arquitetura Principal que orquestra o Base Model e o Continuous Memory Module.
    """
    def __init__(self, model_name: str = 'neuralmind/bert-base-portuguese-cased', value_dim: int = 2):
        super().__init__()
        self.encoder = FrozenBaseModel(model_name)
        embed_dim = self.encoder.bert.config.hidden_size
        self.hope_module = HopeModule(embed_dim, value_dim)
        
    def update_memory(self, rule_text: str, action_vector: List[float]):
        """
        Pipeline de injeção contínua (Update Memory). 
        Calcula o embedding e atualiza a matriz. Sem treinamento.
        """
        embed = self.encoder.get_embedding([rule_text])
        val_tensor = torch.tensor([action_vector], dtype=torch.float32)
        self.hope_module.memory.add_memory(embed, val_tensor, rule_text)
        
    def predict(self, query_text: str) -> Dict:
        """
        Realiza a inferência considerando o conteúdo base + a memória anexada.
        """
        query_embed = self.encoder.get_embedding([query_text])
        predicted_action, attention_weights = self.hope_module(query_embed)
        
        result = {
            "query": query_text,
            "predicted_action_vector": predicted_action.squeeze().tolist(),
        }
        
        # Formata o resultado de atenção para validação analítica
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
