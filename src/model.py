from layer import MultiHeadHerbAttention
from torch_geometric.nn import HeteroConv, SAGEConv
import torch
import torch.nn as nn


class FHPRModel(nn.Module):
    """
    Foundational herbal prescription reasoning model
    """
    def __init__(self, args, num_herbs, sym_rule, dis_rule, condition_num=3):
        super().__init__()
        self.args = args
        self.num_herbs = num_herbs
        D = args.bottle_neurons

        self.sym_rule = sym_rule
        self.dis_rule = dis_rule

        # Herb embedding
        self.herb_embedding = nn.Embedding(num_herbs, D)

        # condition attention modules (sym/ syn/ dis)
        self.condition_attentions = nn.ModuleList([
            MultiHeadHerbAttention(
                D,
                num_heads=args.head_number,
                dropout=args.dropout
            )
            for _ in range(condition_num)
        ])


        self.mlp_herb = nn.Sequential(
            nn.Linear(self.num_herbs, 128),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(128, self.num_herbs)
        )

        self.mlp_dosage = nn.Sequential(
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(128, self.num_herbs)
        )

        self.mlp_dosage_2 = nn.Sequential(
            nn.Linear(self.num_herbs, 128),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(128, self.num_herbs)
        )

        self._init_weights()

    def rule_drive(self, batch, sym_ori, dis_ori):
        """Guide rules to adjust the dosage"""

        rule_dosage = torch.zeros(batch, self.num_herbs, device=sym_ori.device)

        # Symptom guidelines
        for sym_id, herb_dict in self.sym_rule.items():
            sym_idx = int(sym_id)
            mask = (sym_ori[:, sym_idx] > 0).float().unsqueeze(1)  # [B,1]
            for herb_id, dose in herb_dict.items():
                herb_idx = int(herb_id)
                rule_dosage[:, herb_idx] = torch.maximum(
                    rule_dosage[:, herb_idx],
                    mask.squeeze() * dose
                )
        # Disease guidelines
        for dis_id, herb_dict in self.dis_rule.items():
            dis_idx = int(dis_id)
            mask = (dis_ori[:, dis_idx] > 0).float().unsqueeze(1)
            for herb_id, dose in herb_dict.items():
                herb_idx = int(herb_id)
                rule_dosage[:, herb_idx] = torch.maximum(
                    rule_dosage[:, herb_idx],
                    mask.squeeze() * dose
                )
        return rule_dosage


    def _init_weights(self):
        """Initialize the weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        sym_ori, syn_ori, dis_ori,
        sym_emb, syn_emb, dis_emb,
        sim_herb, sim_dosage
    ):

        # Current batch size
        B = sym_ori.size(0)

        # Herb embedding
        herb_emb = self.herb_embedding.weight

        # Representation enhance
        enh_sym = sym_ori @ sym_emb
        enh_syn = syn_ori @ syn_emb
        enh_dis = dis_ori @ dis_emb
        conditions = [enh_sym, enh_syn, enh_dis]

        # Comprehensive herb representations through integrating mult-condition clinical information
        attn_outputs = []
        for i, cond in enumerate(conditions):
            attn_vec, _ = self.condition_attentions[i](cond, herb_emb)
            attn_outputs.append(attn_vec)

        herb_agg = torch.add(attn_outputs[0],attn_outputs[1])
        herb_agg = torch.add(herb_agg, attn_outputs[2])

        # Similar herb mean representation
        basic_herb = sim_herb.mean(dim=0)
        weight = self.mlp_herb(basic_herb)
        sim_herb_emb = herb_emb * weight.unsqueeze(1)

        # Herb prediction
        herb_logits = torch.mm(herb_agg, sim_herb_emb.T)
        herb_probs = torch.sigmoid(herb_logits)  # π(a|s)

        # Dosage prediction & optimization
        pre_dosage = torch.mm(herb_probs,herb_emb)
        pre_dosage = self.mlp_dosage(pre_dosage)
        rule_raw = self.rule_drive(B, sym_ori, dis_ori)
        opt_dosage = self.mlp_dosage_2(pre_dosage + rule_raw + sim_dosage)

        return herb_probs, opt_dosage


class TCMHeteroGNN(nn.Module):
    """ Graph representation network"""
    def __init__(self, args):
        super().__init__()
        hidden_dim = args.hidden_dim

        self.conv = HeteroConv({
            ('Herb', 'Treats', 'Disease'):
                SAGEConv((-1, -1), hidden_dim),
            ('Disease', 'Treats_by', 'Herb'):
                SAGEConv((-1, -1), hidden_dim),

            ('Herb', 'Relieves', 'Symptom'):
                SAGEConv((-1, -1), hidden_dim),
            ('Symptom', 'Relieves_by', 'Herb'):
                SAGEConv((-1, -1), hidden_dim),

            ('Herb', 'Synergizes with', 'Herb'):
                SAGEConv((-1, -1), hidden_dim),

            ('Disease', 'Present with', 'Symptom'):
                SAGEConv((-1, -1), hidden_dim),
            ('Symptom', 'Present by', 'Disease'):
                SAGEConv((-1, -1), hidden_dim),

            ('Disease', 'Manifests as', 'Syndrome'):
                SAGEConv((-1, -1), hidden_dim),
            ('Syndrome', 'be Manifested as', 'Disease'):
                SAGEConv((-1, -1), hidden_dim),

            ('Syndrome', 'Has', 'Symptom'):
                SAGEConv((-1, -1), hidden_dim),
            ('Symptom', 'comprise', 'Syndrome'):
                SAGEConv((-1, -1), hidden_dim)
        }, aggr='sum')

        self.linear = nn.Sequential(
            nn.Linear(hidden_dim, args.linear_dim1),
            nn.LayerNorm(args.linear_dim1),
            nn.ReLU(),
            nn.Linear(args.linear_dim1, args.linear_dim2),
            nn.LayerNorm(args.linear_dim2),
            nn.ReLU()
        )

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv(x_dict, edge_index_dict)
        x_dict['Symptom'] = self.linear(x_dict['Symptom'])
        x_dict['Syndrome'] = self.linear(x_dict['Syndrome'])
        x_dict['Disease'] = self.linear(x_dict['Disease'])
        x_dict['Herb'] = self.linear(x_dict['Herb'])

        return x_dict



