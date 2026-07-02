import json
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

def processDatasetTCMGCD(args, data_file, sim_prescription, seed, device):
    # Data load
    df = pd.read_excel(data_file)
    with open(sim_prescription, 'r') as f:
        sim_pre = json.load(f)

    # Multi-hot encode
    p_dict = multi_hotEncode(data_file,sim_pre)

    # Patient clinical information in patient EHRs (i.e., symptom, disease, and syndrome)
    sym_ori = torch.tensor(np.array(p_dict['symptom']),device=device,dtype=torch.float32)
    syn_ori = torch.tensor(np.array(p_dict['syndrome']),device=device,dtype=torch.float32)
    dis_ori = torch.tensor(np.array(p_dict['disease']),device=device,dtype=torch.float32)

    # Similar patients' prescription, which includes herbs and dosage
    herb_sim = torch.tensor(np.array(p_dict['SimHerb']),device=device,dtype=torch.float32)
    dose_vec = np.array(p_dict['SimDosage'])

    # Ground truth
    true_herb = torch.tensor(np.array(p_dict['herb']),device=device,dtype=torch.float32)
    true_dosage = np.array(p_dict['dosage'])

    # Types of herbs
    herb_numbers = p_dict['HerbNumbers']

    # Normalize dosage
    nor_true_dosage, nor_dosage_vector,global_dosage_min,global_dosage_range = normalize_dosage(true_dosage,dose_vec)
    nor_true_dosage = torch.tensor(nor_true_dosage,device=device,dtype=torch.float32)
    nor_dosage_vector = torch.tensor(nor_dosage_vector,device=device,dtype=torch.float32)

    # Dataset splitting
    id_list = [x for x in range(len(df))]
    x_train, x_test = train_test_split(
        id_list, test_size=args.test_ratio, shuffle=False, random_state=seed)

    # Training set
    train_symOri = sym_ori[x_train]
    train_synOri = syn_ori[x_train]
    train_diseOri = dis_ori[x_train]
    train_herbSim = herb_sim[x_train]
    train_dosageV = nor_dosage_vector[x_train]
    train_true_herb = true_herb[x_train]
    train_true_dosage = nor_true_dosage[x_train]

    # Test set
    test_symOri = sym_ori[x_test]
    test_synOri = syn_ori[x_test]
    test_diseOri = dis_ori[x_test]
    test_herbSim = herb_sim[x_test]
    test_dosageV = nor_dosage_vector[x_test]
    test_true_herb = true_herb[x_test]
    test_true_dosage = nor_true_dosage[x_test]

    train_dataset = PreDatasetTCMGCD(train_symOri, train_synOri, train_diseOri, train_herbSim, train_dosageV,
                                     train_true_herb, train_true_dosage)
    test_dataset = PreDatasetTCMGCD(test_symOri, test_synOri, test_diseOri, test_herbSim, test_dosageV,
                                    test_true_herb, test_true_dosage)

    train_dataloader = DataLoader(dataset=train_dataset,batch_size=args.batch_size,drop_last=True)
    test_dataloader = DataLoader(dataset=test_dataset,batch_size=args.batch_size,drop_last=False)

    return train_dataloader,test_dataloader,herb_numbers,global_dosage_min,global_dosage_range

def normalize_dosage(true_dosage,dose_vec):
    """ normalize dosage in similar prescription and true prescription"""
    all_dosage_values = []

    def flatten_value(v):
        if isinstance(v, list):
            for item in v:
                flatten_value(item)
        else:
            if isinstance(v, (int, float)):
                all_dosage_values.append(v)

    for v in true_dosage:
        flatten_value(v)

    dosage_vector_flat = dose_vec.flatten()
    all_dosage_values.extend(dosage_vector_flat.tolist())

    all_dosage_values = np.array(all_dosage_values)
    global_dosage_min = all_dosage_values.min()
    global_dosage_max = all_dosage_values.max()
    global_dosage_range = global_dosage_max - global_dosage_min + 1e-8  # 避免除0

    nor_true_dosage = []
    for v in true_dosage:
        normalized_v = [(x - global_dosage_min) / global_dosage_range for x in v]

        nor_true_dosage.append(normalized_v)

    nor_true_dosage = np.array(nor_true_dosage)
    nor_dosage_vector = (dose_vec - global_dosage_min) / global_dosage_range

    return nor_true_dosage,nor_dosage_vector,global_dosage_min,global_dosage_range

def multi_hotEncode(data_file,sim_matrix):
    # Data load
    df = pd.read_excel(data_file, sheet_name='Data')
    sym_ids = pd.read_excel(data_file, sheet_name='Symptom Dictionary')
    syn_ids = pd.read_excel(data_file, sheet_name='Syndrome Dictionary')
    disease_ids = pd.read_excel(data_file, sheet_name='Disease Dictionary')
    herb_ids = pd.read_excel(data_file, sheet_name='Herb Dictionary')

    count_disease = len(disease_ids)
    count_symptom = len(sym_ids)
    count_syndrome = len(syn_ids)
    count_herb = len(herb_ids)

    dis_bol = []
    sym_bol = []
    syn_bol = []
    herb_bol = []
    dosage_bol = []
    Sim_herbs = []
    Sim_dosage = []
    for patient in df.values:
        disease = [patient[1]]
        symptom = [int(i) for i in patient[2].split(',') if i] if pd.notna(patient[2]) and patient[2] != '' else []
        syndrome = [int(i) for i in patient[5].split(',') if i] if pd.notna(patient[5]) and patient[5] != '' else []
        herbs = [int(i) for i in patient[3].split(',') if i] if pd.notna(patient[3]) and patient[3] != '' else []
        dosage = [float(i) for i in patient[4].split(',') if i] if pd.notna(patient[4]) and patient[4] != '' else []
        herb_dosage = dict(zip(herbs, dosage))

        dis_bol.append([1 if d in disease else 0 for d in range(1,count_disease+1)])
        sym_bol.append([1 if sy in symptom else 0 for sy in range(1,count_symptom+1)])
        syn_bol.append([1 if syn in syndrome else 0 for syn in range(1,count_syndrome+1)])
        herb_bol.append([1 if h in herbs else 0 for h in range(1,count_herb+1)])
        dosage_bol.append([herb_dosage[h] if h in herb_dosage.keys() else 0.0 for h in range(1,count_herb+1)])
        Sim_herbs.append([1 if str(h) in sim_matrix[patient[0]].keys() else 0 for h in range(1,count_herb+1)])
        Sim_dosage.append([float(sim_matrix[patient[0]][str(h)]) if str(h) in sim_matrix[patient[0]].keys() else 0.0 for h in range(1,count_herb+1)])

    return {
        'disease': dis_bol,
        'symptom': sym_bol,
        'syndrome': syn_bol,
        'herb': herb_bol,
        'dosage': dosage_bol,
        'SimHerb':Sim_herbs,
        'SimDosage':Sim_dosage,
        'HerbNumbers':count_herb,
        'DiseaseNumbers':count_disease,
        'SymptomNumbers':count_symptom,
        'SyndromeNumbers':count_syndrome
    }


class PreDatasetTCMGCD(Dataset):
    def __init__(self, a,b,c,d,e,f,g):
        self.sym_ori = a
        self.syn_ori = b
        self.dis_ori = c
        self.herb_sim = d
        self.dosage_vector = e
        self.true_herb = f
        self.true_dosage = g

    # Return idx-th sample
    def __getitem__(self, idx):
        sym_ori = self.sym_ori[idx]
        syn_ori = self.syn_ori[idx]
        dis_ori = self.dis_ori[idx]
        herb_sim = self.herb_sim[idx]
        dosage_vector = self.dosage_vector[idx]
        true_herb = self.true_herb[idx]
        true_dosage = self.true_dosage[idx]
        return sym_ori,syn_ori,dis_ori,herb_sim,dosage_vector,true_herb,true_dosage

    # Return length of dataset
    def __len__(self):
        return len(self.syn_ori)


