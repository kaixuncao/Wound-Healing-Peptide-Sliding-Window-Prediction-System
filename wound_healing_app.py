import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io, base64
from tensorflow.keras.models import load_model
from Bio import SeqIO

st.set_page_config(page_title="伤口修复多肽预测软件", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #f7f9fb; }
    .main-title { text-align:center; font-size:28px; color:#2c3e50; font-weight:700; }
    .sub-title { text-align:center; font-size:14px; color:#5f6c7b; margin-bottom:10px; }
    .card { background: white; padding:12px; border-radius:8px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
    hr { border: none; height: 1px; background-color: #e6e9ee; margin: 18px 0; }
    </style>
""", unsafe_allow_html=True)

LANGS = {
    'zh': {
        'title': '🧬 伤口修复多肽预测软件',
        'subtitle': '支持FASTA上传、任意长度片段组合预测与中英界面切换（默认中文）',
        'input_seq': '粘贴单条氨基酸序列（可选）:',
        'upload_fasta': '或上传FASTA文件（支持批量）',
        'threshold': '预测阈值（高亮/判定阈值）',
        'window_prompt': '预测片段长度组合（逗号分隔，例如：6,9,12,18）',
        'run_btn': '🚀 运行预测',
        'invalid_chars': '含有非法字符，请只使用标准 20 种氨基酸字母。',
        'seq_heading': '🧩 序列：`{sid}` （长度: {ln}）',
        'top_frag': '— 前 {k} 个高概率片段（窗口长度={w}） —',
        'download_frag': '💾 下载此序列的 fragment 结果 (CSV)',
        'download_pos': '💾 下载此序列的 per-position 结果 (CSV)',
        'all_download': '📊 下载所有序列的汇总 fragment 结果 (CSV)',
        'n_pos_msg': '预测结果：{n}/{L} 个氨基酸位点高于阈值（潜在活性）',
        'explain': """说明：
- 支持多长度片段联合预测（例如 6,9,12,18）。
- 对长序列自动滑动预测并融合结果。
- 热图颜色表示预测概率：蓝（低）→ 红（高）。
- 页面同时输出 per-position 与 fragment-level 结果，可下载 CSV。"""
    },
    'en': {
        'title': '🧬 Wound-Healing Peptide Predictor',
        'subtitle': 'FASTA upload, multi-length sliding-window prediction and English/Chinese toggle (default: Chinese)',
        'input_seq': 'Paste a single amino-acid sequence (optional):',
        'upload_fasta': 'Or upload a FASTA file (batch supported)',
        'threshold': 'Prediction threshold (for highlighting)',
        'window_prompt': 'Window lengths, comma-separated (e.g. 6,9,12,18)',
        'run_btn': '🚀 Run prediction',
        'invalid_chars': 'Sequence contains invalid characters. Use the 20 standard amino acid letters only.',
        'seq_heading': '🧩 Sequence: `{sid}` (length: {ln})',
        'top_frag': '— Top {k} fragments (window={w}) —',
        'download_frag': '💾 Download fragment results (CSV)',
        'download_pos': '💾 Download per-position results (CSV)',
        'all_download': '📊 Download all sequences fragments (CSV)',
        'n_pos_msg': '{n}/{L} positions above threshold (potential active sites)',
        'explain': """Notes:
- Supports multi-window combined prediction (e.g. 6,9,12,18).
- Automatically slides over long sequences and aggregates results.
- Colors show predicted probability: blue (low) → red (high).
- Both per-position and fragment-level CSV outputs are provided."""
    }
}

lang_choice = st.radio("Language / 语言", options=["中文", "English"], index=0)
lang = 'zh' if lang_choice == "中文" else 'en'
T = LANGS[lang]

st.markdown(f"<div class='main-title'>{T['title']}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='sub-title'>{T['subtitle']}</div>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

@st.cache_resource
def load_trained_model(path="wound_healing_model_fold1.h5"):
    model = load_model(path)
    return model

model = load_trained_model()

# model.inputs[0] is sequence input; remaining inputs are kmer3,kmer4,kmer5,physchem
try:
    input_dims = [int(inp.shape[1]) for inp in model.inputs[1:]]
    # defensive: if length mismatch, pad zeros
    while len(input_dims) < 4:
        input_dims.append(0)
except Exception:
    # fallback to safe defaults (if cannot read shapes)
    input_dims = [20, 40, 60, 9]
kmer3_dim, kmer4_dim, kmer5_dim, physchem_dim = input_dims

# =========================
# Tokenizer & helper
# =========================
VOCAB = ["<pad>", "<unk>"] + list("ACDEFGHIKLMNPQRSTVWY")
char_to_int = {c: i for i, c in enumerate(VOCAB)}
MAX_SEQ_LEN = 45  # model was trained to accept 45-length tokens

def tokenize_sequence(seq, max_len=MAX_SEQ_LEN):
    seq = seq.upper()
    tokenized = [char_to_int.get(c, char_to_int["<unk>"]) for c in seq]
    if len(tokenized) < max_len:
        tokenized += [char_to_int["<pad>"]] * (max_len - len(tokenized))
    return np.array(tokenized[:max_len])

def sliding_window_prediction(sequence, model, window_sizes):
    seq = sequence.strip().upper()
    n = len(seq)
    # per-position accumulators
    pos_scores = np.zeros(n, dtype=float)
    pos_counts = np.zeros(n, dtype=float)
    # fragment-level store: dict window -> list of dicts
    fragments_by_window = {w: [] for w in window_sizes}

    # prepare dummy inputs matching model expected dims
    dummy_k3 = np.zeros((1, kmer3_dim)) if kmer3_dim>0 else np.zeros((1,1))
    dummy_k4 = np.zeros((1, kmer4_dim)) if kmer4_dim>0 else np.zeros((1,1))
    dummy_k5 = np.zeros((1, kmer5_dim)) if kmer5_dim>0 else np.zeros((1,1))
    dummy_phys = np.zeros((1, physchem_dim)) if physchem_dim>0 else np.zeros((1,1))

    for w in window_sizes:
        if w <= 0:
            continue
        # slide over sequence
        for i in range(0, n - w + 1):
            seg = seq[i:i+w]
            token = np.expand_dims(tokenize_sequence(seg), axis=0)  # shape (1,MAX_SEQ_LEN)
            # predict using model inputs in same order as training
            try:
                pred = model.predict([token, dummy_k3, dummy_k4, dummy_k5, dummy_phys], verbose=0)[0][0]
            except Exception:
                # some models may expect different number of inputs; try minimal
                pred = model.predict([token], verbose=0)[0][0]
            # record fragment
            fragments_by_window[w].append({
                'start': i+1,               # 1-based
                'end': i+w,
                'fragment': seg,
                'prob': float(pred),
                'window': w
            })
            # accumulate per-position
            pos_scores[i:i+w] += pred
            pos_counts[i:i+w] += 1

    # compute averaged per-position scores
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_pos = np.divide(pos_scores, pos_counts, out=np.zeros_like(pos_scores), where=pos_counts>0)
    return avg_pos, fragments_by_window

def plot_heatmap(seq, scores):
    seq_len = len(seq)
    y = np.array(scores)
    cmap = plt.cm.get_cmap('coolwarm')
    colors = cmap(y)

    fig, ax = plt.subplots(figsize=(min(14, max(6, seq_len/4)), 1.6))
    ax.bar(range(seq_len), np.ones_like(y), width=1.0, color=colors, edgecolor='none')
    # write AA letters if not too long
    if seq_len <= 120:
        for i, aa in enumerate(seq):
            ax.text(i, -0.12, aa, fontsize=6.0, ha='center', va='top', rotation=90)
    ax.axis('off')
    ax.set_xlim(0, seq_len)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=180)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_b64

with st.container():
    st.markdown(f"**{T['input_seq']}**")
    seq_input = st.text_area("", height=120, placeholder="ACDEFGHIKLMNPQRSTVWY...")
    st.markdown(f"**{T['upload_fasta']}**")
    uploaded = st.file_uploader("", type=['fasta','fa'])
    st.markdown("---")

col1, col2 = st.columns([1,2])
with col1:
    threshold = st.slider(T['threshold'], 0.05, 0.95, 0.5, 0.05)
with col2:
    window_input = st.text_input(T['window_prompt'], value="6,9,12,18")

# parse window sizes
try:
    window_list = [int(x.strip()) for x in window_input.split(",") if x.strip()!='']
    # filter unrealistic sizes (>0 and <=MAX_SEQ_LEN_allowed)
    window_list = [w for w in window_list if 1 <= w <= 1000]
    if len(window_list)==0:
        window_list=[9]
except:
    window_list=[9]

st.markdown("<hr>", unsafe_allow_html=True)

if st.button(T['run_btn']):
    # collect sequences
    seqs = []
    if uploaded:
        try:
            fasta = SeqIO.parse(uploaded, "fasta")
            for rec in fasta:
                seqs.append((rec.id, str(rec.seq)))
        except Exception as e:
            st.error(f"FASTA 解析失败: {e}")
            st.stop()
    elif seq_input.strip():
        seqs.append(("User_Sequence", seq_input.strip()))
    else:
        st.warning(T['invalid_chars'])
        st.stop()

    all_fragment_dfs = []
    per_position_dfs = []

    for sid, seq in seqs:
        seq = seq.strip().upper().replace(" ", "").replace("\n","")
        # validate chars
        if not set(seq).issubset(set("ACDEFGHIKLMNPQRSTVWY")):
            st.error(f"{sid} - {T['invalid_chars']}")
            continue

        st.markdown(f"<div class='card'><h4>{T['seq_heading'].format(sid=sid, ln=len(seq))}</h4></div>", unsafe_allow_html=True)

        # run sliding prediction
        avg_pos_scores, fragments_by_window = sliding_window_prediction(seq, model, window_list)

        # show heatmap
        img_b64 = plot_heatmap(seq, avg_pos_scores)
        st.markdown(f'<img src="data:image/png;base64,{img_b64}" width="900"/>', unsafe_allow_html=True)

        # per-position table + download
        pos_df = pd.DataFrame({
            "Sequence_ID": sid,
            "Position": np.arange(1, len(seq)+1),
            "AminoAcid": list(seq),
            "PredictedScore": avg_pos_scores
        })
        per_position_dfs.append(pos_df)
        st.download_button(T['download_pos'], data=pos_df.to_csv(index=False).encode('utf-8'), file_name=f"{sid}_per_position.csv", mime="text/csv")

        # for each window size: show top-5 fragments
        for w in window_list:
            frags = fragments_by_window.get(w, [])
            if not frags:
                continue
            frags_sorted = sorted(frags, key=lambda x: x['prob'], reverse=True)
            topk = frags_sorted[:5]
            st.markdown(f"**{T['top_frag'].format(k=len(topk), w=w)}**")
            # display a small table
            frag_table = pd.DataFrame([{
                "start": f['start'],
                "end": f['end'],
                "fragment": f['fragment'],
                "probability": round(f['prob'], 6)
            } for f in topk])
            st.table(frag_table)
            # also collect full fragment list to save/download
            frag_df_full = pd.DataFrame([{
                "Sequence_ID": sid,
                "window": f['window'],
                "start": f['start'],
                "end": f['end'],
                "fragment": f['fragment'],
                "probability": f['prob']
            } for f in frags_sorted])
            all_fragment_dfs.append(frag_df_full)
            # allow download of full fragment list for this window+sequence
            st.download_button(f"{T['download_frag']} ({w}aa)", data=frag_df_full.to_csv(index=False).encode('utf-8'), file_name=f"{sid}_fragments_w{w}.csv", mime="text/csv")

        # count high positions
        n_high = int((avg_pos_scores > threshold).sum())
        st.success(T['n_pos_msg'].format(n=n_high, L=len(seq)))

        st.markdown("<hr>", unsafe_allow_html=True)

    # if multiple sequences, provide combined download of fragments
    if len(all_fragment_dfs) > 0:
        combined_frag_df = pd.concat(all_fragment_dfs, ignore_index=True)
        st.download_button(T['all_download'], data=combined_frag_df.to_csv(index=False).encode('utf-8'), file_name="all_fragments_combined.csv", mime="text/csv")

    # also combined per-position
    if len(per_position_dfs) > 1:
        comb_pos = pd.concat(per_position_dfs, ignore_index=True)
        st.download_button("📥 Download combined per-position CSV", data=comb_pos.to_csv(index=False).encode('utf-8'), file_name="all_per_position.csv", mime="text/csv")

    st.markdown(T['explain'])