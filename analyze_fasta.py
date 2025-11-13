from Bio import SeqIO
import re

def analyze_fasta(filepath):
    lengths = []
    sequences = []
    for record in SeqIO.parse(filepath, "fasta"):
        seq = str(record.seq).upper()
        seq = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', seq)
        if seq:
            lengths.append(len(seq))
            sequences.append(seq)
    return lengths, sequences

wound_healing_lengths, wound_healing_sequences = analyze_fasta("./wound_healing_peptides.fasta")
uniprot_lengths, uniprot_sequences = analyze_fasta("./uniprot_sprot_filter.fasta")

print(f"Wound healing peptides count (before augmentation): {len(wound_healing_lengths)}")
print(f"Wound healing peptides min length: {min(wound_healing_lengths)}")
print(f"Wound healing peptides max length: {max(wound_healing_lengths)}")
print(f"Wound healing peptides average length: {sum(wound_healing_lengths) / len(wound_healing_lengths):.2f}")
print(f"Uniprot peptides count: {len(uniprot_lengths)}")
print(f"Uniprot peptides min length: {min(uniprot_lengths)}")
print(f"Uniprot peptides max length: {max(uniprot_lengths)}")
print(f"Uniprot peptides average length: {sum(uniprot_lengths) / len(uniprot_lengths):.2f}")

with open("wound_healing_sequences.txt", "w") as f:
    for seq in wound_healing_sequences:
        f.write(seq + "\n")
        
with open("uniprot_sequences.txt", "w") as f:
    for seq in uniprot_sequences:
        f.write(seq + "\n")



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

with open("wound_healing_sequences.txt", "w") as f:
    for seq in wound_healing_sequences:
        f.write(seq + "\n")
        
with open("uniprot_sequences.txt", "w") as f:
    for seq in uniprot_sequences:
        f.write(seq + "\n")