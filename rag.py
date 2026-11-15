# 2025/12/26に実行成功
# 事前にOPENAI_API_KEYの環境変数を事前に設定する。仮想環境有効化.txt参照。

import requests
from bs4 import BeautifulSoup
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import httpx
import os

# ===== OpenAI（リランク・回答生成用） =====
#os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
client = OpenAI(http_client=httpx.Client())

# ===== SentenceTransformer（Embedding用） =====
# 日本語対応モデル（精度と軽さのバランスが良い）
embedding_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# -----------------------------
# スクレイピング
# -----------------------------
def scrape_article(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    text_nodes = soup.find_all("div")
    joined_text = "".join(
        t.get_text(strip=True)
        for t in text_nodes
    )
    return joined_text

# -----------------------------
# チャンク分割
# -----------------------------
def chunk_text(text, chunk_size, overlap):
    chunks = []
    start = 0
    while start + chunk_size <= len(text):
        chunks.append(text[start:start + chunk_size])
        start += (chunk_size - overlap)
    if start < len(text):
        chunks.append(text[-chunk_size:])
    return chunks

# -----------------------------
# sentence-transformersでベクトル化
# -----------------------------
def vectorize_texts(texts):
    """
    texts: List[str]
    return: np.ndarray (N, dim)
    """
    return embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True  # cosine類似度に最適
    )

# -----------------------------
# ベクトル検索
# -----------------------------
def find_top_k_similar(question_vector, vectors, documents, k=5):
    similarities = cosine_similarity(
        question_vector.reshape(1, -1),
        vectors
    )[0]

    ranked_indices = similarities.argsort()[::-1][:k]
    return [documents[i] for i in ranked_indices]

# -----------------------------
# LLMリランク
# -----------------------------
def rerank_with_llm(question, documents):
    scored_docs = []

    for doc in documents:
        prompt = f"""
あなたは検索結果のリランカーです。
次の文章が、質問にどれだけ関連しているかを0〜10で評価してください。

[質問]
{question}

[文章]
{doc}

数字のみで回答してください。
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0
        )

        try:
            score = float(response.choices[0].message.content.strip())
        except:
            score = 0

        scored_docs.append((score, doc))

    scored_docs.sort(reverse=True, key=lambda x: x[0])
    return [doc for score, doc in scored_docs]

# -----------------------------
# コンテキスト生成
# -----------------------------
def build_context(documents, max_docs=2):
    return "\n\n".join(documents[:max_docs])

# -----------------------------
# 質問応答
# -----------------------------
def ask_question(question, context):
    prompt = f"""
以下の質問に以下の情報をベースにして答えて下さい。

[質問]
{question}

[情報]
{context}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200
    )

    return response.choices[0].message.content

# =============================
# 実行
# =============================
# OpenAI → SentenceTransformerに変更したので、chunk_size400→200に変更
url = "https://www.kuronekoyamato.co.jp/ytc/customer/send/members/nyanpay/"
chunk_size = 200
overlap = 50

article_text = scrape_article(url)
text_chunks = chunk_text(article_text, chunk_size, overlap)

# Embedding（まとめて実行するのが高速）
# OpenAI → SentenceTransformerに変更したのでquestionをキーワード寄りに変更
doc_vectors = vectorize_texts(text_chunks)

question = "残高 失効条件"
question_vector = vectorize_texts([question])[0]

# ① ベクトル検索　OpenAI → SentenceTransformerに変更したので、k=5→10に変更
top_k_docs = find_top_k_similar(
    question_vector,
    doc_vectors,
    text_chunks,
    k=10
)

# ② LLMリランク
reranked_docs = rerank_with_llm(question, top_k_docs)

# ③ コンテキスト作成
context = build_context(reranked_docs, max_docs=2)

# ④ 回答生成
answer = ask_question(question, context)
print(answer)
