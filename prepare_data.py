import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from collections import Counter
from itertools import groupby
import random
from Bio.Align import substitution_matrices  # 加载BLOSUM62

def load_sequences(filepath, label):
    with open(filepath, 'r') as f:
        sequences = [line.strip() for line in f if line.strip()]
    return pd.DataFrame({'sequence': sequences, 'label': label})

def calculate_physicochemical_properties(sequence):
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    
    standard_aa = "ACDEFGHIKLMNPQRSTVWY"
    cleaned_seq = ''.join([aa for aa in sequence if aa in standard_aa])
    
    if not cleaned_seq or len(cleaned_seq) < 3:
        return np.zeros(9)
    
    try:
        X = ProteinAnalysis(cleaned_seq)
        
        properties = [
            X.molecular_weight(),
            X.aromaticity(),
            X.instability_index(),
            X.isoelectric_point(),
            X.gravy(),
            sum(1 for aa in cleaned_seq if aa in ['A', 'V', 'L', 'I']) / len(cleaned_seq) * 100,
            X.secondary_structure_fraction()[0],  # helix fraction
            X.secondary_structure_fraction()[1],  # turn fraction
            X.secondary_structure_fraction()[2]   # sheet fraction
        ]
        
        return np.array(properties)
        
    except Exception as e:
        print(f"Error calculating properties for sequence {cleaned_seq}: {str(e)}")
        return np.zeros(9)

PHYSCHEM_GROUPS = {
    'hydrophobic': list('AVLIMFWP'),
    'polar_uncharged': list('STCYNQ'),
    'acidic': list('DE'),
    'basic': list('KRH'),
    'glycine': ['G']
}

def get_group(aa):
    for group, aas in PHYSCHEM_GROUPS.items():
        if aa in aas:
            return group
    return None

# 基于BLOSUM62的替换增强
def augment_sequence_blosum(seq, blosum_matrix, num_replacements=2):
    aa_list = list("ACDEFGHIKLMNPQRSTVWY")
    seq_list = list(seq)
    positions = random.sample(range(len(seq)), min(num_replacements, len(seq)))
    for pos in positions:
        original_aa = seq_list[pos]
        candidates = [aa for aa in aa_list if blosum_matrix[original_aa, aa] > 0 and aa != original_aa]
        if candidates:
            seq_list[pos] = random.choice(candidates)
    return ''.join(seq_list)

# 基于物理化学性质分组的替换增强
def augment_sequence_physchem(seq, num_replacements=2):
    seq_list = list(seq)
    positions = random.sample(range(len(seq)), min(num_replacements, len(seq)))
    for pos in positions:
        original_aa = seq_list[pos]
        group = get_group(original_aa)
        if group:
            candidates = [aa for aa in PHYSCHEM_GROUPS[group] if aa != original_aa]
            if candidates:
                seq_list[pos] = random.choice(candidates)
    return ''.join(seq_list)

def extract_kmer_features(sequences, k=3):
    kmer_counts = Counter()
    
    for seq in sequences:
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            kmer_counts[kmer] += 1
    
    return kmer_counts

def get_important_kmers(X_train, y_train, k=3, top_n=20):
    healing_peptides = [seq for seq, label in zip(X_train, y_train) if label == 1]
    non_healing_peptides = [seq for seq, label in zip(X_train, y_train) if label == 0]
    
    # 提取k-mer
    healing_kmers = extract_kmer_features(healing_peptides, k)
    non_healing_kmers = extract_kmer_features(non_healing_peptides, k)
    
    # 计算每个k-mer的相对频率比
    freq_ratio = {}
    total_healing = sum(healing_kmers.values())
    total_non_healing = sum(non_healing_kmers.values())
    
    all_kmers = set(healing_kmers.keys()) | set(non_healing_kmers.keys())
    for kmer in all_kmers:
        healing_freq = healing_kmers.get(kmer, 0) / (total_healing + 1e-10)
        non_healing_freq = non_healing_kmers.get(kmer, 0) / (total_non_healing + 1e-10)
        
        ratio = (healing_freq + 0.01) / (non_healing_freq + 0.01)
        freq_ratio[kmer] = ratio
    
    sorted_kmers = sorted(freq_ratio.items(), key=lambda x: x[1], reverse=True)
    important_kmers = [kmer for kmer, ratio in sorted_kmers[:top_n]]
    
    return important_kmers

def create_kmer_features(sequences, important_kmers, k=3):
    kmer_features = []
    
    for seq in sequences:
        features = []
        for kmer in important_kmers:
            count = 0
            for i in range(len(seq) - k + 1):
                if seq[i:i+k] == kmer:
                    count += 1
            features.append(count)
        kmer_features.append(features)
    
    return np.array(kmer_features)

wound_healing_df = load_sequences("wound_healing_sequences.txt", 1)
uniprot_df = load_sequences("uniprot_sequences.txt", 0)

blosum62 = substitution_matrices.load("BLOSUM62")

augmentation_factor = 4
augmented_sequences = []
augmented_labels = []

for _, row in wound_healing_df.iterrows():
    original_seq = row['sequence']
    augmented_sequences.append(original_seq)
    augmented_labels.append(1)
    
    for _ in range(2):
        aug_seq = augment_sequence_blosum(original_seq, blosum62, num_replacements=random.randint(1, 3))
        augmented_sequences.append(aug_seq)
        augmented_labels.append(1)
    
    for _ in range(2):
        aug_seq = augment_sequence_physchem(original_seq, num_replacements=random.randint(1, 3))
        augmented_sequences.append(aug_seq)
        augmented_labels.append(1)

wound_healing_df = pd.DataFrame({'sequence': augmented_sequences, 'label': augmented_labels})
print(f"Augmented wound healing sequences: {len(wound_healing_df)}")

data = pd.concat([wound_healing_df, uniprot_df]).sample(frac=1, random_state=42).reset_index(drop=True)

data = data[data['sequence'].apply(len) <= 45]

data.to_csv("processed_sequences.csv", index=False)
print(f"Total sequences after filtering (<=45 aa): {len(data)}")
print(f"Wound healing sequences: {data['label'].sum()}")
print(f"Non-wound healing sequences: {len(data) - data['label'].sum()}")

X = data['sequence'].tolist()
y = data['label'].tolist()

important_kmers_3 = get_important_kmers(X, y, k=3, top_n=20)
important_kmers_4 = get_important_kmers(X, y, k=4, top_n=15)
important_kmers_5 = get_important_kmers(X, y, k=5, top_n=10)

print("Top 20 important 3-mers:", important_kmers_3)
print("Top 15 important 4-mers:", important_kmers_4)
print("Top 10 important 5-mers:", important_kmers_5)

with open("important_kmers_3.txt", "w") as f:
    for kmer in important_kmers_3:
        f.write(kmer + "\n")

with open("important_kmers_4.txt", "w") as f:
    for kmer in important_kmers_4:
        f.write(kmer + "\n")

with open("important_kmers_5.txt", "w") as f:
    for kmer in important_kmers_5:
        f.write(kmer + "\n")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)

with open("X_train.txt", "w") as f:
    for seq in X_train:
        f.write(seq + "\n")
        
with open("X_val.txt", "w") as f:
    for seq in X_val:
        f.write(seq + "\n")
        
with open("X_test.txt", "w") as f:
    for seq in X_test:
        f.write(seq + "\n")
        
np.savetxt("y_train.txt", y_train, fmt='%d')
np.savetxt("y_val.txt", y_val, fmt='%d')
np.savetxt("y_test.txt", y_test, fmt='%d')

physchem_train = np.array([calculate_physicochemical_properties(seq) for seq in X_train])
physchem_val = np.array([calculate_physicochemical_properties(seq) for seq in X_val])
physchem_test = np.array([calculate_physicochemical_properties(seq) for seq in X_test])

scaler = StandardScaler()
physchem_train_scaled = scaler.fit_transform(physchem_train)
physchem_val_scaled = scaler.transform(physchem_val)
physchem_test_scaled = scaler.transform(physchem_test)

np.save("physchem_scaler_mean.npy", scaler.mean_)
np.save("physchem_scaler_scale.npy", scaler.scale_)

np.save("physchem_train.npy", physchem_train_scaled)
np.save("physchem_val.npy", physchem_val_scaled)
np.save("physchem_test.npy", physchem_test_scaled)

X_train_kmer3 = create_kmer_features(X_train, important_kmers_3, k=3)
X_val_kmer3 = create_kmer_features(X_val, important_kmers_3, k=3)
X_test_kmer3 = create_kmer_features(X_test, important_kmers_3, k=3)

X_train_kmer4 = create_kmer_features(X_train, important_kmers_4, k=4)
X_val_kmer4 = create_kmer_features(X_val, important_kmers_4, k=4)
X_test_kmer4 = create_kmer_features(X_test, important_kmers_4, k=4)

X_train_kmer5 = create_kmer_features(X_train, important_kmers_5, k=5)
X_val_kmer5 = create_kmer_features(X_val, important_kmers_5, k=5)
X_test_kmer5 = create_kmer_features(X_test, important_kmers_5, k=5)

np.save("X_train_kmer3.npy", X_train_kmer3)
np.save("X_val_kmer3.npy", X_val_kmer3)
np.save("X_test_kmer3.npy", X_test_kmer3)

np.save("X_train_kmer4.npy", X_train_kmer4)
np.save("X_val_kmer4.npy", X_val_kmer4)
np.save("X_test_kmer4.npy", X_test_kmer4)

np.save("X_train_kmer5.npy", X_train_kmer5)
np.save("X_val_kmer5.npy", X_val_kmer5)
np.save("X_test_kmer5.npy", X_test_kmer5)

print("Data split into training, validation, and test sets and saved.")