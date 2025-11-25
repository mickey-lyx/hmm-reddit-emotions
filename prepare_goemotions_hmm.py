#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data processing pipeline for GoEmotions + HMM.

Inputs:
    goemotions_1.csv, goemotions_2.csv, goemotions_3.csv

Outputs:
    observations.txt  # cluster IDs for each sequence
    labels.txt        # emotion label IDs for each sequence
    map.txt           # original comment IDs for each sequence

Each line corresponds to one thread-level sequence (all comments under a link_id,
sorted by time). Columns are space-separated.

BERT sentence embeddings are cached to:
    embeddings.npy
    embedding_ids.txt
"""

import os
import math
from collections import defaultdict

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans

# --------------------------
# CONFIG
# --------------------------
CSV_FILES = ["data/full_dataset/goemotions_1.csv", "data/full_dataset/goemotions_2.csv", "data/full_dataset/goemotions_3.csv"]

# Emotion columns in the CSV (order defines label IDs 0..27)
EMOTION_COLUMNS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise", "neutral"
]

# 4-way sentiment mapping (from GoEmotions docs/blog)
SENTIMENT_GROUPS = {
    "positive": [
        "admiration", "amusement", "approval", "caring",
        "desire", "excitement", "gratitude", "joy",
        "love", "optimism", "pride", "relief",
    ],
    "negative": [
        "anger", "annoyance", "disappointment", "disapproval",
        "disgust", "embarrassment", "fear", "grief",
        "nervousness", "remorse", "sadness",
    ],
    "ambiguous": [
        "confusion", "curiosity", "realization", "surprise",
    ],
    "neutral": [
        "neutral",
    ],
}

SENTIMENT_TO_ID = {
    "positive": 0,
    "negative": 1,
    "ambiguous": 2,
    "neutral": 3,
}

NUM_SENTIMENTS = len(SENTIMENT_TO_ID)

BERT_MODEL_NAME = "bert-base-uncased"
MAX_SEQ_LEN = 128
K_CLUSTERS = 32          # 超参数：KMeans 聚类数
MIN_SEQ_LEN = 1          # 允许长度 >=1 的序列（单评论也作为一条序列）

OBS_PATH = "observations.txt"
LABELS_PATH = "labels.txt"
MAP_PATH = "map.txt"

# Embedding cache
EMB_CACHE_NPY = "embeddings.npy"
EMB_CACHE_IDS = "embedding_ids.txt"


# --------------------------
# STEP 1: Load & aggregate CSVs
# --------------------------

def load_and_aggregate(csv_files):
    print("Loading CSV files...")
    dfs = []
    for path in csv_files:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found")
        # GoEmotions 原始有 tsv 版本，这里简单兼容
        if path.endswith(".tsv"):
            df = pd.read_csv(path, sep="\t")
        else:
            df = pd.read_csv(path)
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    print(f"Total rows (with rater_id): {len(data)}")

    # 保留我们需要的列
    cols_needed = [
        "id", "text", "subreddit", "link_id", "parent_id", "created_utc"
    ] + EMOTION_COLUMNS
    missing = [c for c in cols_needed if c not in data.columns]
    if missing:
        raise ValueError(f"Missing expected columns in CSV: {missing}")

    data = data[cols_needed]

    # 按 id 聚合（同一条评论可能有多个标注者）
    print("Aggregating by comment id...")
    agg_dict = {col: "sum" for col in EMOTION_COLUMNS}
    agg_dict.update({
        "text": "first",
        "subreddit": "first",
        "link_id": "first",
        "parent_id": "first",
        "created_utc": "first",
    })

    grouped = data.groupby("id", as_index=False).agg(agg_dict)
    print(f"Unique comments after aggregation: {len(grouped)}")

    # 把 created_utc 转成 float / numeric（如果有的话）
    grouped["created_utc"] = pd.to_numeric(grouped["created_utc"], errors="coerce")

    return grouped


# --------------------------
# STEP 2: Build sequences by thread (flatten by post)
# --------------------------

def build_thread_sequences_by_time(df):
    """
    每个 link_id 下的所有评论按时间排序，作为一条序列。
    df: DataFrame after aggregation, one row per comment.
    Returns: list of sequences, each sequence is a list of comment ids (strings).
    """
    print("Building thread-level sequences (flattened by post)...")
    sequences = []

    # 按 link_id 分组，每个帖子一条序列
    for link_id, group in tqdm(df.groupby("link_id"), desc="link_id groups"):
        # 按 created_utc 排序；如果有 NaN，用 0.0 处理
        group_sorted = group.copy()
        group_sorted["created_utc_"] = group_sorted["created_utc"].fillna(0.0)
        group_sorted = group_sorted.sort_values("created_utc_")

        seq = group_sorted["id"].tolist()
        if len(seq) >= MIN_SEQ_LEN:
            sequences.append(seq)

    print(f"Total thread-level sequences: {len(sequences)}")
    return sequences


# --------------------------
# STEP 3: Compute / load BERT embeddings
# --------------------------

def init_bert():
    print("Loading BERT model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = AutoModel.from_pretrained(BERT_MODEL_NAME)

    # 选择设备：优先 MPS (Apple GPU) 其次 CUDA, 最后 CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)
    model.eval()

    print(f"Using device: {device}")
    return tokenizer, model, device


def load_cached_embeddings():
    if os.path.exists(EMB_CACHE_NPY) and os.path.exists(EMB_CACHE_IDS):
        print(f"Loading cached embeddings from {EMB_CACHE_NPY} / {EMB_CACHE_IDS} ...")
        embs = np.load(EMB_CACHE_NPY)
        with open(EMB_CACHE_IDS, "r", encoding="utf-8") as f:
            ids = [line.strip() for line in f if line.strip()]
        if embs.shape[0] != len(ids):
            print("Warning: embeddings.npy and embedding_ids.txt size mismatch, ignoring cache.")
            return None
        id_to_emb = {cid: emb for cid, emb in zip(ids, embs)}
        print(f"Loaded cached embeddings for {len(id_to_emb)} comments.")
        return id_to_emb
    return None


def save_cached_embeddings(id_to_emb):
    print(f"Saving embeddings cache to {EMB_CACHE_NPY} / {EMB_CACHE_IDS} ...")
    ids = list(id_to_emb.keys())
    embs = np.stack([id_to_emb[cid] for cid in ids], axis=0)
    np.save(EMB_CACHE_NPY, embs)
    with open(EMB_CACHE_IDS, "w", encoding="utf-8") as f:
        for cid in ids:
            f.write(cid + "\n")
    print("Embeddings cache saved.")


def compute_sentence_embeddings(df, tokenizer, model, device, batch_size=32):
    """
    Returns: dict id -> embedding (np.array of shape [hidden_size])
    如果本地已有缓存，则直接加载。
    """
    # 先尝试加载缓存
    cached = load_cached_embeddings()
    if cached is not None:
        # 注意：如果你换了 CSV / 过滤规则，记得手动删除缓存文件再跑
        return cached

    print("Computing sentence embeddings with BERT (first run, will be cached)...")

    id_list = df["id"].tolist()
    text_list = df["text"].fillna("").tolist()
    id_to_emb = {}

    with torch.no_grad():
        for i in tqdm(range(0, len(id_list), batch_size), desc="BERT encoding"):
            batch_ids = id_list[i:i+batch_size]
            batch_texts = text_list[i:i+batch_size]

            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_SEQ_LEN,
                return_tensors="pt"
            ).to(device)

            outputs = model(**inputs)
            # 使用 CLS token 作为句子向量
            cls_emb = outputs.last_hidden_state[:, 0, :]  # [batch, hidden_size]
            cls_emb = cls_emb.cpu().numpy()

            for cid, emb in zip(batch_ids, cls_emb):
                id_to_emb[cid] = emb

    print(f"Computed embeddings for {len(id_to_emb)} comments")

    # 保存缓存
    save_cached_embeddings(id_to_emb)

    return id_to_emb


# --------------------------
# STEP 4: KMeans clustering
# --------------------------

def run_kmeans(id_to_emb, n_clusters=K_CLUSTERS):
    print(f"Running KMeans with K={n_clusters} on all embeddings...")
    ids = list(id_to_emb.keys())
    embs = np.stack([id_to_emb[cid] for cid in ids], axis=0)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    cluster_ids = kmeans.fit_predict(embs)

    id_to_cluster = {cid: int(clust) for cid, clust in zip(ids, cluster_ids)}
    print("KMeans clustering done.")
    return id_to_cluster


# --------------------------
# STEP 5: Compute main emotion label per comment
# --------------------------

def compute_main_labels(df):
    """
    For each comment id, collapse the 28 fine-grained emotions into
    4 sentiment classes: positive / negative / ambiguous / neutral.

    We sum rater counts within each sentiment group and take argmax.
    Returns: dict id -> sentiment_id (0..3)
    """
    print("Computing main SENTIMENT labels per comment (4 classes)...")

    # 预先把 emotion -> sentiment 映射好，加速遍历
    emotion_to_sentiment = {}
    for sent_name, emo_list in SENTIMENT_GROUPS.items():
        for emo in emo_list:
            emotion_to_sentiment[emo] = sent_name

    id_to_label = {}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="sentiment labels"):
        cid = row["id"]

        # 累加每个 sentiment 的总票数
        sent_scores = {s: 0.0 for s in SENTIMENT_TO_ID.keys()}
        for emo in EMOTION_COLUMNS:
            count = float(row[emo])
            if count <= 0:
                continue
            sent = emotion_to_sentiment[emo]
            sent_scores[sent] += count

        total = sum(sent_scores.values())

        if total == 0:
            # 没有任何情绪标签，默认归为 neutral
            label_sent = "neutral"
        else:
            # 取票数最高的 sentiment
            label_sent = max(sent_scores.items(), key=lambda kv: kv[1])[0]

        label_id = SENTIMENT_TO_ID[label_sent]
        id_to_label[cid] = label_id

    print(f"Computed sentiment labels for {len(id_to_label)} comments")
    print("Sentiment id mapping:", SENTIMENT_TO_ID)
    return id_to_label


# --------------------------
# STEP 6: Write observations.txt, labels.txt, map.txt
# --------------------------

def write_sequences(
    sequences,
    id_to_cluster,
    id_to_label,
    obs_path=OBS_PATH,
    labels_path=LABELS_PATH,
    map_path=MAP_PATH
):
    print("Writing output files...")

    with open(obs_path, "w", encoding="utf-8") as f_obs, \
         open(labels_path, "w", encoding="utf-8") as f_lab, \
         open(map_path, "w", encoding="utf-8") as f_map:

        kept = 0
        for seq in tqdm(sequences, desc="Writing sequences"):
            # 过滤掉有缺失 cluster / label 的序列
            if any(cid not in id_to_cluster or cid not in id_to_label for cid in seq):
                continue

            obs_line = " ".join(str(id_to_cluster[cid]) for cid in seq)
            lab_line = " ".join(str(id_to_label[cid]) for cid in seq)
            map_line = " ".join(seq)

            f_obs.write(obs_line + "\n")
            f_lab.write(lab_line + "\n")
            f_map.write(map_line + "\n")
            kept += 1

    print(f"Done. Sequences written: {kept}")
    print(f"observations: {obs_path}")
    print(f"labels:       {labels_path}")
    print(f"map:          {map_path}")


# --------------------------
# MAIN
# --------------------------

def main():
    # 1) Load & aggregate
    df = load_and_aggregate(CSV_FILES)

    # 2) Build thread-level sequences (flatten by post)
    sequences = build_thread_sequences_by_time(df)

    # 3) Init BERT & compute/load embeddings
    tokenizer, model, device = init_bert()
    id_to_emb = compute_sentence_embeddings(df, tokenizer, model, device, batch_size=32)

    # 4) KMeans clustering
    id_to_cluster = run_kmeans(id_to_emb, n_clusters=K_CLUSTERS)

    # 5) Compute main emotion labels (0..27)
    id_to_label = compute_main_labels(df)

    # 6) Write out sequences
    write_sequences(sequences, id_to_cluster, id_to_label)


if __name__ == "__main__":
    main()