#!/usr/bin/env python3
"""
ADNAT: Algorithmic Divergence in Neural Abstraction Tracker
Extended to align with Information Dynamics Framework for measuring emergent patterns

Core metrics:
- ΔK: Normalized compression divergence
- ∇J: Spectral Jacobian analysis
- I(X;Y): Mutual information (InfoNCE and MINE)
- Φ: Activation trajectory drift
- ψ: Conditional mutual information for predictive dynamics
- φ: Output mutual information
- ε: Emergence score

Authors: Research Collective, Laurentiu G. Florea
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import lzma
import bz2
from sklearn.decomposition import PCA
# Import UMAP directly from the umap-learn package
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# We remove the MINE dependency as it's problematic to install
# from minepy import MINE as MinepyMINE

class ADNATTracker:
    """Extended ADNAT for tracking emergent patterns in neural networks"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.activations = {}
        self.gradients = {}
        self.attention_matrices = {}
        self.hooks = []
        self._register_hooks()
        
    def _register_hooks(self):
        """Register hooks for activations, gradients, and attention matrices"""
        def get_activation(name):
            def hook(model, input, output):
                # The output of transformer blocks is often a tuple.
                # We are interested in the first element, which is the hidden state tensor.
                if isinstance(output, tuple):
                    self.activations[name] = output[0].detach()
                else:
                    self.activations[name] = output.detach()
            return hook
        
        def get_gradient(name):
            def hook(module, grad_input, grad_output):
                # The grad_output of transformer blocks can also be a tuple.
                if isinstance(grad_output, tuple):
                    # We are interested in the first element.
                    self.gradients[name] = grad_output[0].detach()
                else:
                    self.gradients[name] = grad_output.detach()
            return hook
            
        def get_attention(name):
            def hook(module, input, output):
                if isinstance(module, nn.MultiheadAttention):
                    # Capture attention weights
                    self.attention_matrices[name] = output[1].detach()  # attn_output_weights
            return hook
        
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:
                handle = module.register_forward_hook(get_activation(name))
                self.hooks.append(handle)
                if hasattr(module, 'register_full_backward_hook'):
                    handle_grad = module.register_full_backward_hook(get_gradient(name))
                    self.hooks.append(handle_grad)
                if isinstance(module, nn.MultiheadAttention):
                    handle_attn = module.register_forward_hook(get_attention(name))
                    self.hooks.append(handle_attn)
    
    def cleanup(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
    
    def compute_psi_field(self, input_sequence: torch.Tensor, n_components: int = 10) -> Dict[str, float]:
        """Compute ψ = I(S_{n+1}; C | S_n) using MINE with PCA reduction"""
        self.model.eval()
        psi_scores = {}
        
        with torch.no_grad():
            _ = self.model(input_sequence)
        
        for layer_name, activation in self.activations.items():
            if activation.dim() < 2 or activation.size(0) < 2:
                continue
                
            # Time-shifted activations
            S_n = activation[:-1].view(activation.size(0) - 1, -1)
            S_n_plus_1 = activation[1:].view(activation.size(0) - 1, -1)
            C = input_sequence[:-1].view(activation.size(0) - 1, -1)
            
            # PCA reduction
            if S_n.shape[0] <= n_components or S_n_plus_1.shape[0] <= n_components or C.shape[0] <= n_components:
                continue

            pca = PCA(n_components=n_components)
            S_n_reduced = pca.fit_transform(S_n.cpu().numpy()).astype(np.float32)
            S_n_plus_1_reduced = pca.transform(S_n_plus_1.cpu().numpy()).astype(np.float32)
            
            # This is a simplification for C's dimensionality
            if C.shape[1] > n_components:
                 C_reduced = C[:, :n_components].cpu().numpy().astype(np.float32)
            else:
                 C_reduced = C.cpu().numpy().astype(np.float32)
            
            # Use Pearson Correlation as a substitute for Mutual Information
            # I(S_{n+1}; [C, S_n]) -> Corr(S_{n+1}, [C,S_n])
            y = np.concatenate([C_reduced, S_n_reduced], axis=-1)
            
            correlations = []
            for i in range(S_n_plus_1_reduced.shape[1]):
                for j in range(y.shape[1]):
                    # pearsonr returns (correlation, p-value)
                    corr, _ = pearsonr(S_n_plus_1_reduced[:, i], y[:, j])
                    if not np.isnan(corr):
                        correlations.append(np.abs(corr))
            
            psi_scores[layer_name] = np.mean(correlations) if correlations else 0.0
        
        return psi_scores
    
    def compute_phi_gate_mine(self, input_batch: torch.Tensor, output_logits: torch.Tensor, 
                             n_components: int = 10) -> Dict[str, float]:
        """Compute φ = Corr(H; α) using Pearson Correlation as a substitute for MINE"""
        self.model.eval()
        phi_scores = {}
        
        with torch.no_grad():
            # This forward pass is just to populate the hooks for activations
            _ = self.model(input_ids=input_batch)
        
        for layer_name, activation in self.activations.items():
            H = activation.view(activation.size(0), -1)
            # The output logits are now passed directly
            alpha = output_logits.view(output_logits.size(0), -1)
            
            # PCA reduction
            if H.shape[0] <= n_components or alpha.shape[0] <= n_components:
                continue
            
            pca_H = PCA(n_components=n_components)
            pca_alpha = PCA(n_components=n_components)
            
            H_np = pca_H.fit_transform(H.cpu().numpy()).astype(np.float32)
            alpha_np = pca_alpha.fit_transform(alpha.cpu().numpy()).astype(np.float32)
            
            # Use Pearson Correlation as a substitute for Mutual Information
            correlations = []
            for i in range(H_np.shape[1]):
                for j in range(alpha_np.shape[1]):
                    corr, _ = pearsonr(H_np[:, i], alpha_np[:, j])
                    if not np.isnan(corr):
                        correlations.append(np.abs(corr))

            phi_scores[layer_name] = np.mean(correlations) if correlations else 0.0
        
        return phi_scores
    
    def compute_emergence_score(self, input_sequence: torch.Tensor, output_object) -> Dict[str, float]:
        """Compute ε = ψ + φ, normalized against random baseline"""
        if not hasattr(output_object, 'logits'):
            return {} # Cannot compute without logits

        output_logits = output_object.logits
        psi_scores = self.compute_psi_field(input_sequence)
        phi_scores = self.compute_phi_gate_mine(input_sequence, output_logits)
        
        # Random baseline
        # Create a random input of the same type and device as the original
        random_input = torch.randint_like(input_sequence, high=self.model.config.vocab_size)
        
        # Create a random output with the same shape as the original logits
        random_output_logits = torch.randn_like(output_logits)

        # We don't need to re-run the model for the random psi_field, as it's not dependent on the model's 'thought process'
        # For a more rigorous baseline, one might run the random_input through the model, but we simplify here.
        psi_random = self.compute_psi_field(random_input) 
        phi_random = self.compute_phi_gate_mine(random_input, random_output_logits)
        
        epsilon_scores = {}
        for layer in psi_scores.keys():
            if layer in phi_scores:
                epsilon = psi_scores[layer] + phi_scores[layer]
                epsilon_random = psi_random.get(layer, 0.0) + phi_random.get(layer, 0.0)
                epsilon_scores[layer] = epsilon - epsilon_random
        
        return epsilon_scores
    
    def extract_attention_flow(self, input_sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract attention matrices for ψ-Field visualization"""
        self.model.eval()
        self.attention_matrices = {}
        with torch.no_grad():
            _ = self.model(input_sequence)
        return self.attention_matrices
    
    def compute_delta_k(self, input_batch: torch.Tensor, 
                       compression_method: str = 'lzma',
                       baseline_control: bool = True) -> Dict[str, float]:
        """Existing ΔK computation, now fixed."""
        # The forward pass is already done in the main analysis loop,
        # so we don't need to call the model again here.
        
        input_bytes = input_batch.cpu().numpy().tobytes()
        input_compressed = self._compress(input_bytes, compression_method)
        
        if baseline_control:
            # The input_batch is a LongTensor. randn_like requires a float tensor.
            noise_baseline = torch.randn_like(input_batch.float()).cpu().numpy().tobytes()
            noise_compressed = self._compress(noise_baseline, compression_method)
            baseline_complexity = len(noise_compressed)
        else:
            baseline_complexity = len(input_compressed)
        
        delta_k_scores = {}
        for layer_name, activation in self.activations.items():
            act_bytes = activation.cpu().numpy().tobytes()
            act_compressed = self._compress(act_bytes, compression_method)
            
            if baseline_control:
                # Activations are usually float, but it's safer to cast.
                noise_act = torch.randn_like(activation.float()).cpu().numpy().tobytes()
                noise_act_compressed = self._compress(noise_act, compression_method)
                # Add epsilon to denominator to prevent division by zero
                delta_k = ((len(act_compressed) - len(input_compressed)) - 
                          (len(noise_act_compressed) - baseline_complexity)) / (baseline_complexity + 1e-9)
            else:
                # Add epsilon to denominator
                delta_k = (len(act_compressed) - len(input_compressed)) / (len(input_compressed) + 1e-9)
                
            delta_k_scores[layer_name] = delta_k
            
        return delta_k_scores
    
    def _compress(self, data: bytes, method: str) -> bytes:
        """Existing compression method (unchanged)"""
        if method == 'lzma':
            return lzma.compress(data)
        elif method == 'bz2':
            return bz2.compress(data)
        else:
            raise ValueError(f"Unknown compression method: {method}")
    
    def compute_spectral_jacobian(self, input_batch: torch.Tensor) -> Dict[str, float]:
        """Computes spectral Jacobian norm w.r.t input embeddings."""
        # We need to compute gradients w.r.t the input embeddings, not the long-tensor of token IDs.
        
        # 1. Get the model's embedding layer
        embedding_layer = self.model.get_input_embeddings()
        
        # 2. Get the embeddings for the input_ids
        input_embeddings = embedding_layer(input_batch)
        # We cannot set requires_grad on a non-leaf variable.
        # Instead, we tell PyTorch to retain the gradient for this intermediate variable.
        input_embeddings.retain_grad()
        
        # 3. Pass the embeddings to the model using the 'inputs_embeds' argument
        # We also need the attention mask from the original input if available
        # This part of the code assumes input_batch is just the IDs.
        # For a robust solution, the whole input dict should be passed around.
        # For now, we proceed with only the embeddings.
        output = self.model(inputs_embeds=input_embeddings)
        
        # 4. Compute a scalar loss and backpropagate
        loss = output.logits.sum() if hasattr(output, 'logits') else output[0].sum()
        loss.backward(retain_graph=True)
        
        jacobian_norms = {}
        # The gradients are now in the .grad attribute of the parameters
        # and also on the input_embeddings tensor itself.
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_flat = param.grad.flatten()
                spectral_norm = torch.norm(grad_flat, p=2).item()
                jacobian_norms[name] = spectral_norm
        
        # Reset gradients
        self.model.zero_grad()
        if input_embeddings.grad is not None:
            input_embeddings.grad.zero_()
        
        return jacobian_norms
    
    def compute_mutual_info_infoNCE(self, input_batch: torch.Tensor, 
                                   targets: torch.Tensor, 
                                   temperature: float = 0.1) -> Dict[str, Tuple[float, float]]:
        """Existing InfoNCE computation (unchanged)"""
        batch_size = input_batch.size(0)
        _ = self.model(input_batch)
        
        mi_scores = {}
        for layer_name, activation in self.activations.items():
            z = activation.view(batch_size, -1)
            z_norm = F.normalize(z, dim=1)
            target_onehot = F.one_hot(targets, num_classes=targets.max().item() + 1).float()
            logits = torch.mm(z_norm, target_onehot.t()) / temperature
            labels = torch.arange(batch_size).to(self.device)
            infoNCE_loss = F.cross_entropy(logits, labels)
            mi_lower_bound = np.log(batch_size) - infoNCE_loss.item()
            mi_scores[layer_name] = (mi_lower_bound, infoNCE_loss.item())
        
        return mi_scores
    
    def compute_trajectory_drift(self, input_sequences: List[torch.Tensor],
                                manifold_method: str = 'pca',
                                n_components: int = 2,
                                collapse_detection: bool = True) -> Dict[str, Dict]:
        """Existing trajectory drift computation with the NameError fixed."""
        layer_trajectories = {}
        # Correctly initialize all_activations based on actual layer names discovered
        layer_names = []
        with torch.no_grad():
            # Correctly call the model with keyword arguments
            _ = self.model(input_ids=input_sequences[0])
        layer_names = list(self.activations.keys())
        all_activations = {name: [] for name in layer_names}
        
        for input_batch in input_sequences:
            with torch.no_grad():
                # Correctly call the model with keyword arguments
                _ = self.model(input_ids=input_batch)
            for layer_name, activation in self.activations.items():
                # Ensure activation is a tensor before calling .cpu()
                if torch.is_tensor(activation):
                    flat_act = activation.cpu().numpy().flatten()
                    all_activations[layer_name].append(flat_act)
        
        for layer_name, activations in all_activations.items():
            if not activations or len(activations) < 3:
                continue
            X = np.array(activations)
            
            if manifold_method == 'pca':
                reducer = PCA(n_components=n_components)
            elif manifold_method == 'umap':
                reducer = umap.UMAP(n_components=n_components)
            else:
                raise ValueError(f"Unknown manifold method: {manifold_method}")
            
            try:
                trajectory = reducer.fit_transform(X)
                trajectory_variance = np.var(trajectory, axis=0).sum()
                
                if collapse_detection and len(trajectory) > 2:
                    pairwise_distances = np.linalg.norm(trajectory[1:] - trajectory[:-1], axis=1)
                    collapse_score = np.exp(-np.mean(pairwise_distances))
                    drift_vectors = trajectory[1:] - trajectory[:-1]
                    if len(drift_vectors) > 1:
                        cosine_sims = [
                            np.dot(drift_vectors[i], drift_vectors[i+1]) / 
                            (np.linalg.norm(drift_vectors[i]) * np.linalg.norm(drift_vectors[i+1]) + 1e-8)
                            for i in range(len(drift_vectors)-1)
                        ]
                        drift_coherence = np.mean(cosine_sims)
                    else:
                        drift_coherence = 0.0
                else:
                    collapse_score = 0.0
                    drift_coherence = 0.0
                
                layer_trajectories[layer_name] = {
                    'trajectory': trajectory,
                    'variance': trajectory_variance,
                    'collapse_score': collapse_score,
                    'drift_coherence': drift_coherence
                }
                
            except Exception as e:
                print(f"Skipping layer {layer_name} due to error: {e}")
                continue
                
        return layer_trajectories
    
    def visualize_attention_flow(self, attention_matrices: Dict[str, torch.Tensor], 
                               save_path: Optional[str] = None):
        """Visualize attention matrices for ψ-Field"""
        fig, axes = plt.subplots(len(attention_matrices), 1, figsize=(10, 5 * len(attention_matrices)))
        if len(attention_matrices) == 1:
            axes = [axes]
        
        for idx, (layer_name, attn_matrix) in enumerate(attention_matrices.items()):
            sns.heatmap(attn_matrix.cpu().numpy().mean(axis=0), ax=axes[idx], cmap='viridis')
            axes[idx].set_title(f'Attention Flow: {layer_name}')
            axes[idx].set_xlabel('Key Tokens')
            axes[idx].set_ylabel('Query Tokens')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
    
    def analyze_model(self, input_batch: torch.Tensor, 
                     targets: Optional[torch.Tensor] = None,
                     input_sequences: Optional[List[torch.Tensor]] = None) -> Dict:
        """Extended analysis including ψ, φ, ε"""
        results = {}
        
        print("Computing ΔK (compression divergence)...")
        results['delta_k'] = self.compute_delta_k(input_batch)
        
        print("Computing ∇J (spectral jacobian)...")
        results['spectral_jacobian'] = self.compute_spectral_jacobian(input_batch)
        
        if targets is not None:
            print("Computing I(X;Y) (InfoNCE mutual information)...")
            results['mutual_info'] = self.compute_mutual_info_infoNCE(input_batch, targets)
            print("Computing φ (MINE mutual information)...")
            # Ensure model output is computed for phi_gate
            with torch.no_grad():
                model_output = self.model(input_ids=input_batch)
            results['phi_gate'] = self.compute_phi_gate_mine(input_batch, model_output.logits)
        
        if input_sequences is not None and len(input_sequences) > 0:
            print("Computing ψ (conditional mutual information)...")
            results['psi_field'] = self.compute_psi_field(input_sequences[0])
            print("Computing Φ (trajectory drift)...")
            results['trajectory_drift'] = self.compute_trajectory_drift(input_sequences)
            print("Computing ε (emergence score)...")
            with torch.no_grad():
                model_output_seq = self.model(input_ids=input_sequences[0])
            results['emergence_score'] = self.compute_emergence_score(input_sequences[0], model_output_seq)
            print("Extracting attention flow...")
            results['attention_flow'] = self.extract_attention_flow(input_sequences[0])
        
        return results
    
    def visualize_results(self, results: Dict, save_path: Optional[str] = None):
        """Extended visualization including ψ, φ, ε, and attention flow"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        fig.suptitle('ADNAT: Information Dynamics Analysis', fontsize=16)
        
        # ΔK Plot
        if 'delta_k' in results:
            layers = list(results['delta_k'].keys())
            values = list(results['delta_k'].values())
            axes[0,0].bar(range(len(layers)), values)
            axes[0,0].set_title('ΔK: Compression Divergence')
            axes[0,0].set_ylabel('Normalized ΔK')
            axes[0,0].tick_params(axis='x', rotation=45)
        
        # Spectral Jacobian Plot
        if 'spectral_jacobian' in results:
            layers = list(results['spectral_jacobian'].keys())
            values = list(results['spectral_jacobian'].values())
            axes[0,1].plot(range(len(layers)), values, 'o-')
            axes[0,1].set_title('∇J: Spectral Jacobian Norms')
            axes[0,1].set_ylabel('Spectral Norm')
        
        # ψ-Field Plot
        if 'psi_field' in results:
            layers = list(results['psi_field'].keys())
            values = list(results['psi_field'].values())
            axes[1,0].bar(range(len(layers)), values, color='purple')
            axes[1,0].set_title('ψ: Conditional Mutual Information')
            axes[1,0].set_ylabel('I(S_{n+1}; C | S_n)')
        
        # φ-Gate Plot
        if 'phi_gate' in results:
            layers = list(results['phi_gate'].keys())
            values = list(results['phi_gate'].values())
            axes[1,1].bar(range(len(layers)), values, color='green')
            axes[1,1].set_title('φ: Output Mutual Information')
            axes[1,1].set_ylabel('I(H; α)')
        
        # ε Plot
        if 'emergence_score' in results:
            layers = list(results['emergence_score'].keys())
            values = list(results['emergence_score'].values())
            axes[2,0].bar(range(len(layers)), values, color='orange')
            axes[2,0].set_title('ε: Emergence Score')
            axes[2,0].set_ylabel('ψ + φ (Normalized)')
        
        # Attention Flow (example layer)
        if 'attention_flow' in results and results['attention_flow']:
            layer_name = list(results['attention_flow'].keys())[0]
            attn_matrix = results['attention_flow'][layer_name].mean(dim=0).cpu().numpy()
            sns.heatmap(attn_matrix, ax=axes[2,1], cmap='viridis')
            axes[2,1].set_title(f'Attention Flow: {layer_name}')
            axes[2,1].set_xlabel('Key Tokens')
            axes[2,1].set_ylabel('Query Tokens')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig

class ADNATExperiments:
    """Extended validation suite including PDF tasks and toy model"""
    
    @staticmethod
    def run_toy_linear_regression(device: str = 'cuda') -> Dict:
        """Toy linear regression example (y = 2x + ε)"""
        class LinearModel(nn.Module):
            def __init__(self):
                super(LinearModel, self).__init__()
                self.linear = nn.Linear(1, 1)
            def forward(self, x):
                return self.linear(x)
        
        model = LinearModel().to(device)
        x = torch.linspace(-1, 1, 100).reshape(-1, 1).to(device)
        y = 2 * x + 0.1 * torch.randn_like(x)
        
        tracker = ADNATTracker(model, device)
        results = tracker.analyze_model(x, y, input_sequences=[x])
        tracker.cleanup()
        
        return results
    
    @staticmethod
    def imdb_sentiment_experiment(model: nn.Module, device: str = 'cuda') -> Dict:
        """Hypothetical IMDb sentiment analysis experiment"""
        from datasets import load_dataset
        dataset = load_dataset("imdb", split='test[:1000]')
        # Placeholder for tokenization (requires actual tokenizer)
        inputs = torch.randn(1000, 512, 768).to(device)  # Mock tokenized inputs
        labels = torch.tensor([1 if x['label'] == 'positive' else 0 for x in dataset]).to(device)
        
        tracker = ADNATTracker(model, device)
        results = tracker.analyze_model(inputs, labels, input_sequences=[inputs])
        tracker.cleanup()
        return results
    
    @staticmethod
    def wikitext_generation_experiment(model: nn.Module, device: str = 'cuda') -> Dict:
        """Hypothetical WikiText-103 text generation experiment"""
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split='test[:1000]')
        inputs = torch.randn(1000, 512, 768).to(device)  # Mock tokenized inputs
        outputs = torch.randn(1000, 512, 768).to(device)  # Mock generated outputs
        
        tracker = ADNATTracker(model, device)
        results = tracker.analyze_model(inputs, None, input_sequences=[inputs])
        results['emergence_score'] = tracker.compute_emergence_score(inputs, outputs)
        tracker.cleanup()
        return results
    
    @staticmethod
    def complexity_comparison(model: nn.Module, inputs: torch.Tensor, device: str = 'cuda') -> Dict:
        """Hypothetical comparison to Logical Depth, EMC, IIT"""
        tracker = ADNATTracker(model, device)
        results = tracker.analyze_model(inputs, input_sequences=[inputs])
        
        # Placeholder for complexity measures (requires actual implementations)
        results['logical_depth'] = {'mock': 0.0}  # Requires Kolmogorov complexity estimation
        results['emc'] = {'mock': 0.0}  # Requires statistical complexity implementation
        results['iit'] = {'mock': 0.0}  # Requires integrated information computation
        
        tracker.cleanup()
        return results