import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import warnings
warnings.filterwarnings("ignore")


with open("X_train.txt", "r") as f:
    X_train = [line.strip() for line in f if line.strip()]
with open("X_val.txt", "r") as f:
    X_val = [line.strip() for line in f if line.strip()]
with open("X_test.txt", "r") as f:
    X_test = [line.strip() for line in f if line.strip()]

y_train = np.loadtxt("y_train.txt", dtype=int)
y_val = np.loadtxt("y_val.txt", dtype=int)
y_test = np.loadtxt("y_test.txt", dtype=int)

physchem_train = np.load("physchem_train.npy")
physchem_val = np.load("physchem_val.npy")
physchem_test = np.load("physchem_test.npy")

X_train_kmer3 = np.load("X_train_kmer3.npy")
X_val_kmer3 = np.load("X_val_kmer3.npy")
X_test_kmer3 = np.load("X_test_kmer3.npy")
X_train_kmer4 = np.load("X_train_kmer4.npy")
X_val_kmer4 = np.load("X_val_kmer4.npy")
X_test_kmer4 = np.load("X_test_kmer4.npy")
X_train_kmer5 = np.load("X_train_kmer5.npy")
X_val_kmer5 = np.load("X_val_kmer5.npy")
X_test_kmer5 = np.load("X_test_kmer5.npy")

print(f"Training set: {len(X_train)} sequences (positive: {sum(y_train)}, negative: {len(y_train) - sum(y_train)})")

# Tokenizer
VOCAB = ["<pad>", "<unk>"] + list("ACDEFGHIKLMNPQRSTVWY")
char_to_int = {char: i for i, char in enumerate(VOCAB)}
max_sequence_length = 45
vocab_size = len(VOCAB)

def tokenize_sequence(sequence, max_len=45):
    tokenized = [char_to_int.get(char, char_to_int["<unk>"]) for char in sequence]
    if len(tokenized) < max_len:
        tokenized += [char_to_int["<pad>"]] * (max_len - len(tokenized))
    return tokenized[:max_len]

X_train_tokenized = np.array([tokenize_sequence(seq) for seq in X_train])
X_val_tokenized = np.array([tokenize_sequence(seq) for seq in X_val])
X_test_tokenized = np.array([tokenize_sequence(seq) for seq in X_test])


def create_enhanced_model():
    sequence_input = layers.Input(shape=(max_sequence_length,), name="sequence_input")
    embedding = layers.Embedding(vocab_size, 8, input_length=max_sequence_length)(sequence_input)

    conv3 = layers.Conv1D(8, 3, activation="relu")(embedding)
    conv5 = layers.Conv1D(8, 5, activation="relu")(embedding)
    conv7 = layers.Conv1D(8, 7, activation="relu")(embedding)
    pool3 = layers.GlobalMaxPooling1D()(conv3)
    pool5 = layers.GlobalMaxPooling1D()(conv5)
    pool7 = layers.GlobalMaxPooling1D()(conv7)
    merged_conv = layers.Concatenate()([pool3, pool5, pool7])

    kmer3_input = layers.Input(shape=(X_train_kmer3.shape[1],))
    kmer4_input = layers.Input(shape=(X_train_kmer4.shape[1],))
    kmer5_input = layers.Input(shape=(X_train_kmer5.shape[1],))
    physchem_input = layers.Input(shape=(physchem_train.shape[1],))

    kmer3_dense = layers.Dense(8, activation="relu")(kmer3_input)
    kmer4_dense = layers.Dense(8, activation="relu")(kmer4_input)
    kmer5_dense = layers.Dense(8, activation="relu")(kmer5_input)
    physchem_dense = layers.Dense(8, activation="relu")(physchem_input)

    merged = layers.Concatenate()([merged_conv, kmer3_dense, kmer4_dense, kmer5_dense, physchem_dense])
    x = layers.Dense(8, activation="relu")(merged)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(
        inputs=[sequence_input, kmer3_input, kmer4_input, kmer5_input, physchem_input],
        outputs=output
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc"),
                 keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")]
    )
    return model


kfold = KFold(n_splits=5, shuffle=True, random_state=42)
fold_no = 1
accuracies, auc_scores, precision_scores, recall_scores = [], [], [], []
models_list = []

for train_idx, val_idx in kfold.split(X_train_tokenized):
    print(f"\n===== Training Fold {fold_no}/5 =====")
    X_train_seq, X_val_seq = X_train_tokenized[train_idx], X_train_tokenized[val_idx]
    X_train_k3, X_val_k3 = X_train_kmer3[train_idx], X_train_kmer3[val_idx]
    X_train_k4, X_val_k4 = X_train_kmer4[train_idx], X_train_kmer4[val_idx]
    X_train_k5, X_val_k5 = X_train_kmer5[train_idx], X_train_kmer5[val_idx]
    X_train_phys, X_val_phys = physchem_train[train_idx], physchem_train[val_idx]
    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

    model = create_enhanced_model()
    log_csv = keras.callbacks.CSVLogger(f"training_log_fold{fold_no}.csv", append=False)
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    history = model.fit(
        [X_train_seq, X_train_k3, X_train_k4, X_train_k5, X_train_phys],
        y_train_fold,
        epochs=100,
        batch_size=8,
        validation_data=([X_val_seq, X_val_k3, X_val_k4, X_val_k5, X_val_phys], y_val_fold),
        callbacks=[log_csv, early_stop, reduce_lr],
        verbose=1
    )

    hist_df = pd.DataFrame(history.history)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(hist_df['loss'], label='Train Loss')
    plt.plot(hist_df['val_loss'], label='Val Loss')
    plt.title(f'Fold {fold_no} - Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(alpha=0.3)

    plt.subplot(1,2,2)
    plt.plot(hist_df['accuracy'], label='Train Acc')
    plt.plot(hist_df['val_accuracy'], label='Val Acc')
    plt.title(f'Fold {fold_no} - Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"training_curve_fold{fold_no}.png", dpi=300)
    plt.close()

    model.save(f"wound_healing_model_fold{fold_no}.h5")
    models_list.append(model)

    results = model.evaluate(
        [X_val_seq, X_val_k3, X_val_k4, X_val_k5, X_val_phys],
        y_val_fold, verbose=0
    )
    accuracies.append(results[1])
    auc_scores.append(results[2])
    precision_scores.append(results[3])
    recall_scores.append(results[4])
    print(f"Fold {fold_no} Results: Acc={results[1]:.3f}, AUC={results[2]:.3f}")
    fold_no += 1


metrics_df = pd.DataFrame({
    'Fold': range(1, 6),
    'Accuracy': accuracies,
    'AUC': auc_scores,
    'Precision': precision_scores,
    'Recall': recall_scores
})
metrics_df.to_csv("cv_metrics.csv", index=False)

melted = metrics_df.melt(id_vars='Fold', var_name='Metric', value_name='Score')
plt.figure(figsize=(8,6))
sns.boxplot(x='Metric', y='Score', data=melted, palette='Set2')
sns.stripplot(x='Metric', y='Score', data=melted, color='black', alpha=0.6)
plt.title("Cross-Validation Performance (5-Fold)")
plt.ylim(0,1)
plt.tight_layout()
plt.savefig("cross_validation_boxplots.png", dpi=300)
plt.close()


rf = RandomForestClassifier(n_estimators=200, random_state=42)
svm = SVC(kernel='rbf', probability=True, random_state=42)

X_features_train = np.hstack([physchem_train, X_train_kmer3, X_train_kmer4, X_train_kmer5])
X_features_val = np.hstack([physchem_val, X_val_kmer3, X_val_kmer4, X_val_kmer5])

rf.fit(X_features_train, y_train)
svm.fit(X_features_train, y_train)

plt.figure(figsize=(8,6))
for name, model in {'Deep Model (Ours)': models_list[0], 'Random Forest': rf, 'SVM': svm}.items():
    if "Deep" in name:
        y_pred = model.predict([X_val_tokenized, X_val_kmer3, X_val_kmer4, X_val_kmer5, physchem_val]).ravel()
    else:
        y_pred = model.predict_proba(X_features_val)[:,1]
    fpr, tpr, _ = roc_curve(y_val, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.3f})')

plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend(); plt.tight_layout()
plt.savefig("roc_comparison.png", dpi=300)
plt.close()


y_pred_prob = models_list[0].predict(
    [X_val_tokenized, X_val_kmer3, X_val_kmer4, X_val_kmer5, physchem_val]
).ravel()
precisions, recalls, thresholds = precision_recall_curve(y_val, y_pred_prob)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)

fig, ax1 = plt.subplots(figsize=(8,6))
ax1.plot(thresholds, precisions[:-1], label='Precision', color='tab:blue')
ax1.plot(thresholds, recalls[:-1], label='Recall', color='tab:orange')
ax1.set_xlabel("Probability Threshold"); ax1.set_ylabel("Precision / Recall")
ax1.legend(loc='upper right'); ax1.grid(alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(thresholds, f1_scores[:-1], color='tab:green', linestyle='--', label='F1 Score')
ax2.set_ylabel("F1 Score")
fig.suptitle("Threshold Analysis: Precision-Recall Tradeoff")
fig.tight_layout()
plt.savefig("threshold_analysis.png", dpi=300)
plt.close()

print("\n✅ Training complete.")
print("Generated files:")
print("  - training_curve_fold*.png (each fold's learning curve)")
print("  - training_log_fold*.csv (each fold's log)")
print("  - cross_validation_boxplots.png")
print("  - roc_comparison.png")
print("  - threshold_analysis.png")
print("  - cv_metrics.csv")