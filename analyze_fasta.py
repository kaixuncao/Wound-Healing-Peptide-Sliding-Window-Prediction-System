from Bio import SeqIO
import re

def analyze_fasta(filepath):
    """分析FASTA文件，安全处理非标准氨基酸"""
    lengths = []
    sequences = []
    for record in SeqIO.parse(filepath, "fasta"):
        seq = str(record.seq).upper()
        seq = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', seq)  # 只保留标准氨基酸
        if seq:  # 只添加非空序列
            lengths.append(len(seq))
            sequences.append(seq)
    return lengths, sequences

wound_healing_lengths, wound_healing_sequences = analyze_fasta("./wound_healing_peptides.fasta")
uniprot_lengths, uniprot_sequences = analyze_fasta("./uniprot_sprot_filter.fasta")

print(f"Wound healing peptides count: {len(wound_healing_lengths)}")
print(f"Wound healing peptides min length: {min(wound_healing_lengths)}")
print(f"Wound healing peptides max length: {max(wound_healing_lengths)}")
print(f"Wound healing peptides average length: {sum(wound_healing_lengths) / len(wound_healing_lengths):.2f}")
print(f"Uniprot peptides count: {len(uniprot_lengths)}")
print(f"Uniprot peptides min length: {min(uniprot_lengths)}")
print(f"Uniprot peptides max length: {max(uniprot_lengths)}")
print(f"Uniprot peptides average length: {sum(uniprot_lengths) / len(uniprot_lengths):.2f}")

# 保存序列和标签
with open("wound_healing_sequences.txt", "w") as f:
    for seq in wound_healing_sequences:
        f.write(seq + "\n")
        
with open("uniprot_sequences.txt", "w") as f:
    for seq in uniprot_sequences:
        f.write(seq + "\n")
