import numpy as np
import torch
import json
from dataloader import processDatasetTCMGCD
from param_parser import parameter_parser
from training_and_testing import FHPRTrainer, EGHPRTrainer


def main():
    DataPath = 'data/TCM-GCD-example.xlsx'
    SimPre = 'data/SimilarPrescription.json'
    args = parameter_parser()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 2026
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

    # Symptom guidelines and disease guidelines
    with open("data/SymGuidelineExample.json", 'r') as f:
        sym_guidelines = json.load(f)

    with open("data/DisGuidelineExample.json", 'r') as f:
        dis_guidelines = json.load(f)

    # Preprocessing dataset
    train_dataloader, test_dataloader, herb_numbers, global_dosage_min, global_dosage_range = (
        processDatasetTCMGCD(args, DataPath, sim_prescription=SimPre, seed=seed, device=device))

    # knowledge graph data
    graph_data = torch.load('data/TCMgraph.pt', weights_only=False)

    # Stage 1: Foundational herbal prescription prediction (SL)
    foundation = FHPRTrainer(args, herb_numbers, sym_guidelines, dis_guidelines)
    foundation.train(train_dataloader, test_dataloader, graph_data,
                     global_dosage_min, global_dosage_range)

    # Stage 2: Herbal prescription refinement (RL / EGHPR)
    refinement = EGHPRTrainer(args, herb_numbers, sym_guidelines, dis_guidelines)
    refinement.train(train_dataloader, test_dataloader, graph_data)


if __name__ == '__main__':
    main()
