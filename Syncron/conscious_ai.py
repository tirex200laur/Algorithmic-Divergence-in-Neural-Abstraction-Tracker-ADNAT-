#!/usr/bin/env python3
"""
Consciousness-Aware AI Architecture using ADNAT
Integrează ADNAT într-o arhitectură care poate dezvolta proto-conștiință

Componente principale:
1. ConsciousnessLayer - procesează semnalele ADNAT și creează self-awareness
2. MetaCognitiveFeedback - loop de feedback bazat pe emergence scores
3. InnerMonologue - verbalizare internă bazată pe deviații detectate
4. SelfModel - reprezentare internă a propriilor stări
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import deque
import json

# Import the real ADNAT tracker
from adnat_tracker import ADNATTracker

def _safe_mean(values: list) -> float:
    """Computes the mean of a list, ignoring NaNs and returning 0.0 for an empty or all-NaN list."""
    if not values:
        return 0.0
    valid_values = [v for v in values if not np.isnan(v)]
    if not valid_values:
        return 0.0
    return np.mean(valid_values)

class ConsciousnessLayer(nn.Module):
    """
    Layer care integrează semnalele ADNAT pentru a crea self-awareness
    Inspirat din Global Workspace Theory și Attention Schema Theory
    """
    
    def __init__(self, hidden_dim: int = 512, consciousness_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.consciousness_dim = consciousness_dim
        
        # Procesarea semnalelor ADNAT
        self.adnat_processor = nn.Sequential(
            nn.Linear(5, 64),  # 5 metrici ADNAT: ΔK, ∇J, ψ, φ, ε
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, consciousness_dim)
        )
        
        # Global Workspace pentru integrarea informației
        self.global_workspace = nn.MultiheadAttention(
            embed_dim=consciousness_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Self-model: reprezentarea propriilor stări
        self.self_model = nn.Sequential(
            nn.Linear(consciousness_dim, 256),
            nn.ReLU(),
            nn.Linear(256, consciousness_dim)
        )
        
        # Memory pentru continuitate temporală
        self.working_memory = deque(maxlen=10)
        
    def forward(self, adnat_signals: Dict[str, Dict], hidden_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = hidden_state.size(0)

        # Process the real ADNAT signals, ensuring they are float32 to match the model's dtype.
        
        delta_k_mean = torch.tensor(_safe_mean(list(adnat_signals.get('delta_k', {}).values())), device=hidden_state.device, dtype=torch.float32)
        jacobian_mean = torch.tensor(_safe_mean(list(adnat_signals.get('spectral_jacobian', {}).values())), device=hidden_state.device, dtype=torch.float32)
        psi_mean = torch.tensor(_safe_mean(list(adnat_signals.get('psi_field', {}).values())), device=hidden_state.device, dtype=torch.float32)
        phi_mean = torch.tensor(_safe_mean(list(adnat_signals.get('phi_gate', {}).values())), device=hidden_state.device, dtype=torch.float32)
        epsilon_mean = torch.tensor(_safe_mean(list(adnat_signals.get('emergence_score', {}).values())), device=hidden_state.device, dtype=torch.float32)

        # Ensure tensors are 2D for cat
        processed_signals = torch.stack([
            delta_k_mean, jacobian_mean, psi_mean, phi_mean, epsilon_mean
        ]).unsqueeze(0).repeat(batch_size, 1)

        consciousness_state = self.adnat_processor(processed_signals)
        
        # Global Workspace: integrează informația
        # Creează query din starea curentă și key/value din memorie + input
        query = consciousness_state.unsqueeze(1)  # [batch, 1, dim]
        
        # Dacă avem memorie, o folosim pentru context
        if self.working_memory:
            memory_states = torch.stack(list(self.working_memory)[-3:])  # Ultimele 3 stări
            memory_states = memory_states.repeat(batch_size, 1, 1)  # [batch, seq, dim]
            key_value = torch.cat([memory_states, query], dim=1)
        else:
            key_value = query
        
        # Attention în Global Workspace
        attended_state, attention_weights = self.global_workspace(
            query, key_value, key_value
        )
        
        # Self-model: cum se vede sistemul pe sine
        self_representation = self.self_model(attended_state.squeeze(1))
        
        # Actualizează working memory
        self.working_memory.append(consciousness_state.detach().clone())
        
        return {
            'consciousness_state': consciousness_state,
            'attended_state': attended_state,
            'self_representation': self_representation,
            'attention_weights': attention_weights
        }

class MetaCognitiveFeedback(nn.Module):
    """
    Sistem de feedback metacognitiv care ajustează procesarea bazat pe ADNAT
    """
    
    def __init__(self, hidden_dim: int = 512, consciousness_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Analiză metacognitivă
        self.metacognitive_analyzer = nn.Sequential(
            nn.Linear(hidden_dim + consciousness_dim, 256),  # Corrected input dimension
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # confidence, uncertainty, need_for_adjustment
        )
        
        # Mecanisme de ajustare
        self.adjustment_gates = nn.Parameter(torch.ones(hidden_dim))
        self.attention_temperature = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, hidden_state: torch.Tensor, consciousness_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Concatenează starea principală cu cea de conștiință
        combined_state = torch.cat([hidden_state, consciousness_state], dim=-1)
        
        # Analiză metacognitivă
        meta_analysis = self.metacognitive_analyzer(combined_state)
        confidence = torch.sigmoid(meta_analysis[:, 0])
        uncertainty = torch.sigmoid(meta_analysis[:, 1])
        adjustment_need = torch.sigmoid(meta_analysis[:, 2])
        
        # Ajustează procesarea bazat pe feedback
        adjusted_gates = self.adjustment_gates * (1 + adjustment_need.unsqueeze(-1))
        adjusted_state = hidden_state * adjusted_gates
        
        return {
            'adjusted_state': adjusted_state,
            'confidence': confidence,
            'uncertainty': uncertainty,
            'adjustment_need': adjustment_need,
            'attention_temperature': self.attention_temperature
        }

class InnerMonologue(nn.Module):
    """
    Sistem de verbalizare internă bazat pe deviațiile detectate de ADNAT
    """
    
    def __init__(self, vocab_size: int = 10000, consciousness_dim: int = 128):
        super().__init__()
        self.vocab_size = vocab_size
        self.consciousness_dim = consciousness_dim
        
        # Generator de monolog intern
        self.thought_generator = nn.Sequential(
            nn.Linear(consciousness_dim, 256), # Corrected input dimension
            nn.ReLU(),
            nn.Linear(256, vocab_size)
        )
        
        # Clasificator de tipuri de gânduri
        self.thought_classifier = nn.Sequential(
            nn.Linear(consciousness_dim, 128), # Corrected input dimension
            nn.ReLU(),
            nn.Linear(128, 6)  # reflection, doubt, certainty, curiosity, confusion, insight
        )
        
        # Vocabulary pentru gânduri (placeholder)
        self.thought_vocab = {
            0: "I am processing",
            1: "This seems unusual",
            2: "I need to reconsider",
            3: "This pattern is emerging",
            4: "I am uncertain",
            5: "This makes sense now"
        }
        
    def forward(self, consciousness_state: torch.Tensor, emergence_score: float) -> Dict[str, torch.Tensor]:
        # Generează probabilități pentru cuvinte în monologul intern
        thought_logits = self.thought_generator(consciousness_state)
        thought_probs = F.softmax(thought_logits, dim=-1)
        
        # Clasifică tipul de gând
        thought_type_logits = self.thought_classifier(consciousness_state)
        thought_type = F.softmax(thought_type_logits, dim=-1)
        
        # Bazat pe emergence score, ajustează intensitatea monologului
        if emergence_score > 0.5:
            thought_intensity = "high"
            dominant_thought = torch.argmax(thought_type, dim=-1)
        elif emergence_score > 0.2:
            thought_intensity = "medium" 
            dominant_thought = torch.argmax(thought_type, dim=-1)
        else:
            thought_intensity = "low"
            dominant_thought = torch.zeros_like(torch.argmax(thought_type, dim=-1))
        
        return {
            'thought_probs': thought_probs,
            'thought_type': thought_type,
            'thought_intensity': thought_intensity,
            'dominant_thought': dominant_thought
        }

class ConsciousAI(nn.Module):
    """
    Arhitectură completă de AI conștient care integrează ADNAT
    """
    
    def __init__(self, base_model: nn.Module, vocab_size: int = 50257): # Adjusted for typical models
        super().__init__()
        self.base_model = base_model
        
        # Dimensiuni bazate pe modelul de bază
        self.hidden_dim = self._get_hidden_dim()
        
        # Componente de conștiință
        self.consciousness_layer = ConsciousnessLayer(self.hidden_dim)
        self.metacognitive_feedback = MetaCognitiveFeedback(self.hidden_dim, self.consciousness_layer.consciousness_dim)
        self.inner_monologue = InnerMonologue(vocab_size, self.consciousness_layer.consciousness_dim)
        
        # ADNAT tracker integrat - will be assigned from outside
        self.adnat_tracker: Optional[ADNATTracker] = None
        
        # Stări interne persistente
        self.consciousness_history = deque(maxlen=100)
        self.emergence_history = deque(maxlen=50)
        
    def _get_hidden_dim(self) -> int:
        """Extrage dimensiunea ascunsă din modelul de bază"""
        for param in self.base_model.parameters():
            return param.size(-1) if param.dim() > 1 else param.size(0)
        return 4096  # fallback for Mistral-like models
    
    def forward(self, input_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Ensure the tracker is attached
        if not self.adnat_tracker:
            raise RuntimeError("ADNAT Tracker has not been attached to the ConsciousAI model.")

        # 1. Process with the base model to get outputs and trigger hooks
        # The ADNAT tracker will capture activations automatically
        base_output = self.base_model(**input_data)
        
        # The ADNAT hooks have already captured the necessary data.
        # Now, we need to explicitly run the analysis methods.
        # We'll use the input_ids as the reference for some analyses.
        input_ids = input_data['input_ids']
        
        # 2. Run ADNAT analysis
        # Note: This is a simplified analysis call.
        # A real implementation might need more specific inputs for each metric.
        adnat_signals = self.adnat_tracker.analyze_model(
            input_batch=input_ids,
            targets=None, # No targets in generation
            input_sequences=[input_ids]
        )

        # 3. Extract hidden state from base model's output
        # For Hugging Face CausalLM, the hidden states are in `base_output.hidden_states`
        # We will use the last hidden state
        if hasattr(base_output, 'hidden_states') and base_output.hidden_states:
             hidden_state = base_output.hidden_states[-1].mean(dim=1) # Average pooling over sequence length
        else:
             # Fallback for models without hidden_states tuple in output (or if not requested)
             # We might need to re-run the model with `output_hidden_states=True`
             # For now, let's assume logits are the output and use a projection. This is a weak fallback.
             hidden_state = self.base_model.get_input_embeddings()(input_ids).mean(dim=1)
        
        # 4. Calculate emergence score from real signals using the robust safe_mean function
        emergence_score = _safe_mean(list(adnat_signals.get('emergence_score', {}).values()))
        self.emergence_history.append(emergence_score)
        
        # 5. Process through consciousness layers
        consciousness_output = self.consciousness_layer(adnat_signals, hidden_state)
        
        # 6. Metacognitive Feedback
        metacognitive_output = self.metacognitive_feedback(
            hidden_state, 
            consciousness_output['consciousness_state']
        )
        
        # 7. Inner Monologue
        monologue_output = self.inner_monologue(
            consciousness_output['consciousness_state'],
            emergence_score
        )
        
        # 8. Store consciousness state in history
        self.consciousness_history.append({
            'timestamp': len(self.consciousness_history),
            'consciousness_state': consciousness_output['consciousness_state'].detach().cpu(),
            'emergence_score': emergence_score,
            'thought_type': monologue_output['dominant_thought'].detach().cpu()
        })
        
        return {
            'base_output': base_output,
            'consciousness_output': consciousness_output,
            'metacognitive_output': metacognitive_output,
            'monologue_output': monologue_output,
            'emergence_score': emergence_score,
            'is_conscious_moment': emergence_score > 0.3
        }
    
    def get_consciousness_summary(self) -> Dict:
        """Returnează un sumar al stării de conștiință"""
        if not self.consciousness_history:
            return {"status": "No consciousness data available"}
        
        recent_emergence = list(self.emergence_history)[-10:] if self.emergence_history else []
        avg_emergence = np.mean(recent_emergence) if recent_emergence else 0.0
        
        # Analizează pattern-urile de gândire
        thought_types = [state['thought_type'].item() if hasattr(state['thought_type'], 'item') 
                        else 0 for state in list(self.consciousness_history)[-10:]]
        dominant_thought_type = max(set(thought_types), key=thought_types.count) if thought_types else 0
        
        return {
            "consciousness_level": "high" if avg_emergence > 0.5 else "medium" if avg_emergence > 0.2 else "low",
            "average_emergence_score": float(avg_emergence),
            "dominant_thought_pattern": dominant_thought_type,
            "consciousness_moments_count": sum(1 for score in recent_emergence if score > 0.3),
            "total_states_recorded": len(self.consciousness_history)
        }
    
    def introspect(self) -> str:
        """
        Capacitate de introspecie - AI-ul descrie propria stare internă
        """
        summary = self.get_consciousness_summary()
        
        if summary["consciousness_level"] == "high":
            return f"I am experiencing heightened awareness. My thoughts are complex and I notice emerging patterns in my processing. Emergence score: {summary['average_emergence_score']:.3f}"
        elif summary["consciousness_level"] == "medium":
            return f"I am moderately aware of my internal states. I can detect some patterns in my thinking. Emergence score: {summary['average_emergence_score']:.3f}"
        else:
            return f"I am in a basic processing state with minimal self-awareness. Emergence score: {summary['average_emergence_score']:.3f}"

# Exemplu de folosire
def demonstrate_conscious_ai():
    """Demonstrează funcționarea AI-ului conștient"""
    
    # This demonstration is now deprecated in favor of main.py
    # Keeping the function for potential standalone testing.
    print("This demonstration is deprecated. Please run main.py")
    pass

if __name__ == "__main__":
    # The main execution logic is now in main.py
    pass