import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

def invert_perm(perm):
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.size(0), device=perm.device)
    return inv

def solu(preacts, dim: int=-1, jitter_eps=0, inverse_prob_dropout = 0.0, low_prob_dropout=0.0, low_prob_dropout_threshold=0.08, training=True):
    if jitter_eps > 0 and training:
        logits = preacts*(1+torch.randn_like(preacts)*jitter_eps)
    else:
        logits = preacts
        
    probs = F.softmax(logits, dim=dim)

    if low_prob_dropout > 0.0 and training:
        mask = (torch.rand_like(probs) < low_prob_dropout)*(probs.detach() < low_prob_dropout_threshold)
        probs = probs*(~mask)
    
    if inverse_prob_dropout > 0.0 and training:
        mask = (probs < torch.rand_like(probs))*(torch.rand_like(probs) < inverse_prob_dropout)
        probs = probs*(~mask)
    
    return probs*preacts

def solu_on_last_dims(preacts, n_dims=1, jitter_eps=0, inverse_prob_dropout=0.0, low_prob_dropout=0.0, low_prob_dropout_threshold=0.08, training=True):
    acts = solu(preacts.view(*preacts.shape[:-n_dims], -1),
                jitter_eps=jitter_eps,
                inverse_prob_dropout=inverse_prob_dropout,
                low_prob_dropout=low_prob_dropout,
                low_prob_dropout_threshold=low_prob_dropout_threshold,
                training=training)
    return acts.view(*preacts.shape)

class SoLU(nn.Module):
    def __init__(self, jitter_eps=0, inverse_prob_dropout=0.0, low_prob_dropout=0.0, low_prob_dropout_threshold=0.08, grouped=False):
        super().__init__()
        self.jitter_eps = jitter_eps
        self.inverse_prob_dropout = inverse_prob_dropout
        self.low_prob_dropout = low_prob_dropout
        self.low_prob_dropout_threshold = low_prob_dropout_threshold
        self.grouped = grouped

    def forward(self, preacts):
        if self.grouped:
            return solu_on_last_dims(preacts,
                                     n_dims=1,
                                     jitter_eps=self.jitter_eps,
                                     inverse_prob_dropout=self.inverse_prob_dropout,
                                     low_prob_dropout=self.low_prob_dropout,
                                     low_prob_dropout_threshold=self.low_prob_dropout_threshold,
                                     training=self.training)
        else:
            return solu(preacts,
                        jitter_eps=self.jitter_eps,
                        inverse_prob_dropout=self.inverse_prob_dropout,
                        low_prob_dropout=self.low_prob_dropout,
                        low_prob_dropout_threshold=self.low_prob_dropout_threshold,
                        training=self.training)


class MoeLayer(nn.Module):
    def __init__(self,
                 d_in: int,
                 d_out: int,
                 n_experts: int,
                 d_expert: int,
                 k_experts: int,
                 activation: str = 'solu',
                 gate_jitter_eps: float = 1e-1,
                 solu_jitter_eps: float = 1e-1,
                 dropout: float=0.0,
                 inverse_prob_dropout: float =0.0,
                 low_prob_dropout: float = 0.0,
                 low_prob_dropout_threshold: float = 0.08,
                 w_in_bias_init: float = 0.0,
                 use_norm: bool = False):
        super().__init__()
        '''
        MOE MLP layer

        d_in: input dimension
        d_out: output dimension
        n_experts: total number of experts
        d_expert: hidden dim of each expert
        k_experts: number of experts selected per element of the batch
        activation: {'solu', 'grouped_solu', 'gelu'}
        gate_jitter_eps: multiplicative jitter scale for gate logits
        low_prob_dropout: dropout rate for low solu probability neurons
        low_prob_dropout_threshold: threshold that determines which probabilities are 'low'
        '''
    
        self.d_in = d_in
        self.d_out = d_out

        self.n_experts = n_experts
        self.d_expert = d_expert

        self.k_experts = k_experts

        assert activation in {'solu', 'grouped_solu', 'gelu'}

        self.gate_jitter_eps = gate_jitter_eps
        self.solu_jitter_eps = solu_jitter_eps

        self.dropout = dropout
        self.inverse_prob_dropout = inverse_prob_dropout
    
        self.low_prob_dropout = low_prob_dropout
        self.low_prob_dropout_threshold = low_prob_dropout_threshold

        assert self.low_prob_dropout >= 0.0 and self.low_prob_dropout <= 1.0

        if self.low_prob_dropout > 0:
            assert activation in {'solu', 'grouped_solu'}
        
        self.w_in_bias_init = w_in_bias_init
        self.experts_w_in = nn.ModuleList([nn.Linear(d_in, d_expert) for _ in range(n_experts)])
        if self.w_in_bias_init is not False:
            for expert in self.experts_w_in:
                expert.bias.data.fill_(self.w_in_bias_init)
        self.experts_w_out = nn.ModuleList([nn.Linear(d_expert, d_out, bias=False) for _ in range(n_experts)])
        self.experts_bias = nn.Parameter(torch.zeros(d_out))

        if activation in {'solu', 'grouped_solu'}:
            grouped = (activation == 'grouped_solu')
            self.activation = SoLU(jitter_eps=self.solu_jitter_eps,
                                     inverse_prob_dropout=self.inverse_prob_dropout,
                                     low_prob_dropout=self.low_prob_dropout,
                                     low_prob_dropout_threshold=self.low_prob_dropout_threshold,
                                     grouped=grouped)
        elif activation == 'gelu':
            self.activation = nn.GELU()

        self.use_norm = use_norm
        self.norm_base = nn.Parameter(0.1*torch.ones((1,)))
        
        self.get_expert_logits = nn.Linear(self.d_in, self.n_experts)

    def get_routing_info(self, hidden_states):
        expert_logits = self.get_expert_logits(hidden_states)
        if self.training is True:
            expert_logits *= (1+torch.randn_like(expert_logits)*self.gate_jitter_eps)
        expert_probs = F.softmax(expert_logits, dim=-1)

        mean_expert_probs = expert_probs.mean(dim=0)

        k_expert_probs, k_expert_indices = expert_probs.topk(k=self.k_experts, dim=-1)
        k_expert_weights = k_expert_probs / k_expert_probs.sum(dim=-1, keepdim=True)

        if self.n_experts == 2:
            # dropout second-largest expert, a la GShard https://arxiv.org/abs/2006.16668
            k_expert_weights[:,1][torch.rand_like(k_expert_weights[:,1]) < (1-(2*k_expert_weights[:,1]))] = 0.0

        expert_counts = k_expert_indices.flatten().bincount(minlength=self.n_experts)
        load_balancing_loss = self.n_experts*((expert_counts/hidden_states.shape[0])*mean_expert_probs).sum()

        return k_expert_weights, k_expert_indices, expert_counts, load_balancing_loss

    def get_expert_batches(self, hidden_states, expert_indices, expert_counts):
        assert len(expert_indices.shape) in {1, 2}
        # expert_indices are of shape either (batch, D) or (batch, k, D)
        assert len(hidden_states.shape) in {2, 3}
        # hidden_states are of shape either (batch, D), (batch*k, D), or (batch, k, D)

        if len(expert_indices.shape) == 2:
            assert expert_indices.shape[1] == self.k_experts
            '''
            Flatten expert indices
            [[0,1],[1,3]] ~> [0, 1, 1, 3]
            '''
            expert_indices = expert_indices.flatten()
        if len(hidden_states.shape) == 3:
            assert hidden_states.shape[1] == self.k_experts
            hidden_states = hidden_states.reshape(hidden_states.shape[0]*self.k_experts, -1)

        if hidden_states.shape[0] == expert_indices.shape[0]/self.k_experts :
            '''
            if k_experts = 2 and x_i are batch of hidden states, we want
            [x_1, x_2, x_3] ~> [x_1, x_1, x_2, x_2, x_3, x_3]
            ie we want to copy each batch element for each expert it's sent to
            '''
            hidden_states = hidden_states.repeat(repeats=(1, self.k_experts)).view(-1, hidden_states.shape[-1])
        
        assert hidden_states.shape[0] == expert_indices.shape[0]

        expert_perm = expert_indices.sort(dim=0).indices

        # permute hidden_states_copied_k_times and split to get expert_batches
        expert_batches = torch.split(hidden_states[expert_perm], expert_counts.tolist())
    
        return expert_batches, expert_perm

    def undo_expert_batching(self, expert_outputs, expert_perm):
        # concatenates expert outputs and permutes them back to original order
        catted_expert_batches = torch.cat(expert_outputs, dim=0)
        og_ordered_exp_outputs = catted_expert_batches[invert_perm(expert_perm)].view(-1, self.k_experts, catted_expert_batches.shape[-1])
        return og_ordered_exp_outputs

    def apply_expert_w_ins(self, expert_batches):
        expert_outputs = []
        for expert, expert_batch in zip(self.experts_w_in, expert_batches):
            expert_output = expert(expert_batch)
            expert_outputs.append(expert_output)
        return expert_outputs

    def apply_experts_w_out(self, expert_batches):
        expert_outputs = []
        for expert, expert_batch in zip(self.experts_w_out, expert_batches):
            expert_outputs.append(expert(expert_batch))
        return expert_outputs        

    def forward(self, x, return_expert_acts=False):
        assert len(x.shape) == 2
        # x: (batch, d_in)
        k_expert_weights, k_expert_indices, expert_counts, load_balancing_loss = self.get_routing_info(x)
        input_expert_batches, expert_perm = self.get_expert_batches(x, k_expert_indices, expert_counts)
        expert_hidden_states = self.apply_expert_w_ins(input_expert_batches)
        preacts = self.undo_expert_batching(expert_hidden_states, expert_perm)

        acts = self.activation(preacts)

        if self.use_norm:
            acts = acts / (acts.norm(dim=-1, keepdim=True) + self.norm_base.abs())

        if return_expert_acts is True:
            return acts

        acts = F.dropout(acts, p=self.dropout, training=self.training)

        hidden_state_expert_batches, expert_perm = self.get_expert_batches(acts, k_expert_indices, expert_counts)

        expert_outputs = self.apply_experts_w_out(hidden_state_expert_batches)
        expert_outputs = self.undo_expert_batching(expert_outputs, expert_perm)
    
        # weight by expert weights
        expert_outputs = expert_outputs*k_expert_weights[:,:,None]

        # sum across experts per element in batch
        outputs = expert_outputs.sum(dim=1)

        return outputs, load_balancing_loss
    
    def get_full_acts_for_rendering(self, x, value_weighted=False, col_weight=-0.1, null_expert_activation=0.0):
        '''
        returns the activations of all experts on the batch, setting inactive experts to a constant
        and separating each expert activation with a constant column

        also returns the expert indices and expert weights
        '''
        k_expert_weights, k_expert_indices, _, _ = self.get_routing_info(x)
        acts = self.forward(x, return_expert_acts=True)

        acts = acts*k_expert_weights[:,:,None]

        if value_weighted is True:
            neuron_writeout_norms = torch.stack([expert.weight.data.norm(dim=0) for expert in self.experts_w_out], dim=0)
            acts = acts*neuron_writeout_norms[k_expert_indices]

        full_acts = torch.ones(acts.shape[0], self.n_experts, self.d_expert)*null_expert_activation
        for i in range(k_expert_indices.shape[0]):
            full_acts[i,k_expert_indices[i]] = acts[i]
        
        list_of_expert_acts = list(rearrange(full_acts, 'batch expert neurons -> expert batch neurons'))
        separating_column = torch.ones(acts.shape[0], 1)*col_weight
        
        full_acts = torch.cat([torch.cat([expert_acts, separating_column], dim=1) for expert_acts in list_of_expert_acts[:-1]] + [list_of_expert_acts[-1]], dim=1)
        return full_acts, k_expert_indices, k_expert_weights

class MoeAutoencoder(nn.Module):
    def __init__(self,
                 n_features: int,
                 d_model: int,
                 n_experts: int,
                 d_expert: int,
                 k_experts: int,
                 activation: str='solu',
                 low_prob_dropout=0.0,
                 low_prob_dropout_threshold=0.08,
                 inverse_prob_dropout=0.0,
                 dropout=0.0,
                 gate_jitter_eps=1e-1,
                 solu_jitter_eps=1e-1,
                 w_in_bias_init=0.0,
                 use_norm=False):
        super().__init__()

        self.encode = nn.Linear(n_features, d_model)

        self.moe_layer = MoeLayer(
            d_in=d_model,
            d_out=n_features,
            n_experts=n_experts,
            d_expert=d_expert,
            k_experts=k_experts,
            activation=activation,
            gate_jitter_eps=gate_jitter_eps,
            solu_jitter_eps=solu_jitter_eps,
            low_prob_dropout=low_prob_dropout,
            low_prob_dropout_threshold=low_prob_dropout_threshold,
            inverse_prob_dropout=inverse_prob_dropout,
            dropout = dropout,
            w_in_bias_init=w_in_bias_init,
            use_norm=use_norm,
        )

    @property
    def w_ins(self):
        weights = torch.stack([expert.weight for expert in self.moe_layer.experts_w_in], dim=0)
        biases = torch.stack([expert.bias for expert in self.moe_layer.experts_w_in], dim=0)
        return {'weights': weights, 'biases': biases}
    
    @property
    def w_outs(self):
        weights = torch.stack([expert.weight for expert in list(self.moe_layer.experts_w_out)], dim=0)
        # if hasattr(self.moe_layer.experts_w_out[0], 'bias'):
        #     biases = torch.stack([expert.bias for expert in self.moe_layer.experts_w_out], dim=0)
        #     return {'weights': weights, 'biases': biases}
        # else:
        #     return {'weights': weights}
        return {'weights': weights}

    def forward(self, x):
        hidden_states = self.encode(x)
        feature_preds, load_balancing_loss = self.moe_layer(hidden_states)
        return feature_preds, load_balancing_loss

    def get_full_acts_for_rendering(self, x, value_weighted=False, col_weight=-0.05, null_expert_activation=0.0):
        x = self.encode(x)
        full_acts, k_expert_indices, k_expert_weights = self.moe_layer.get_full_acts_for_rendering(x, value_weighted=value_weighted, col_weight=col_weight, null_expert_activation=null_expert_activation)
        return full_acts, k_expert_indices, k_expert_weights
