# 2025/12/26に実行成功 → LangChain v0.2+対応に改善
# Ingestion（1回）と Query（何回）を分離したアーキテクチャ
# 事前にOPENAI_API_KEYの環境変数を事前に設定する。仮想環境有効化.txt参照。

import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin, urlparse
from pathlib import Path

# ===== LangChain関連 =====
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import RecursiveUrlLoader
import pickle
import shutil

# ===== LLM セットアップ（OpenAI） =====
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=200)
llm_rerank = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=5)

# ===== Embeddings セットアップ =====
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
)

# ===== テキスト分割 =====
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "、", " ", ""]
)

# ===== VectorDB設定 =====
VECTORDB_PATH = "./faiss_db"  # ローカル保存パス
CHUNKS_INFO_FILE = "./chunks_info.txt"  # チャンク情報ファイル

# =============================
# Ingestion処理（1回だけ実行）
# =============================
def ingest_documents(base_url, max_depth=2, force_refresh=False):
    """
    ドキュメントを取得・処理してVectorDBに保存
    
    Args:
        base_url: 開始URL
        max_depth: 再帰の最大深さ
        force_refresh: Trueで既存DBを削除して再取得
    """
    
    # 既存DBを削除する場合
    if force_refresh and Path(VECTORDB_PATH).exists():
        import shutil
        shutil.rmtree(VECTORDB_PATH)
        print("[OK] 既存VectorDBを削除しました\n")
    
    # 既にVectorDBが存在する場合はスキップ
    if Path(VECTORDB_PATH).exists() and not force_refresh:
        print("[OK] VectorDBが既に存在します。Queryモードに進みます。\n")
        return
    
    print("=" * 70)
    print("【Ingestion処理開始】")
    print("=" * 70)
    print()
    
    # ========== ステップ1: 再帰的URL取得 ==========
    print("[1] URL配下のサイトを再帰的に取得中...")
    raw_documents = load_recursive_urls(base_url, max_depth=max_depth)
    
    if not raw_documents:
        print("[ERROR] ドキュメントの取得に失敗しました")
        return False
    
    print(f"[OK] 初期ドキュメント数: {len(raw_documents)}個\n")
    
    # ========== ステップ2: ドキュメント前処理 ==========
    print("[2] ドキュメント前処理中...")
    documents = preprocess_documents(raw_documents, min_length=100)
    print(f"[OK] フィルタリング後: {len(documents)}個\n")
    
    # ========== ステップ3: テキストチャンク分割 ==========
    print("[3] テキストをチャンク分割中...")
    text_chunks = []
    for doc in documents:
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            chunk_doc = Document(
                page_content=chunk,
                metadata={
                    "source": doc.metadata.get("source", "unknown"),
                }
            )
            text_chunks.append(chunk_doc)
    
    print(f"[OK] {len(text_chunks)}個のチャンクに分割完了\n")
    
    # ========== ステップ4: VectorDB構築・保存 ==========
    print("[4] VectorDB構築中（ディスクに保存）...")
    vectorstore = FAISS.from_documents(
        documents=text_chunks,
        embedding=embeddings
    )
    
    # FAISSをディスクに保存
    vectorstore.save_local(VECTORDB_PATH)
    print(f"[OK] VectorDB保存完了: {VECTORDB_PATH}\n")
    
    # ========== ステップ5: チャンク情報をファイルに保存 ==========
    print("[5] チャンク情報をファイルに保存中...")
    save_chunks_to_file(text_chunks)
    print(f"[OK] チャンク情報保存完了: {CHUNKS_INFO_FILE}\n")
    
    print("=" * 70)
    print("【Ingestion処理完了】")
    print("=" * 70)
    print()
    
    return True


# =============================
# チャンク情報をファイルに保存
# =============================
def save_chunks_to_file(text_chunks):
    """
    チャンク内容をテキストファイルに保存
    
    Args:
        text_chunks: List[Document]
    """
    with open(CHUNKS_INFO_FILE, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("【チャンク情報リスト】\n")
        f.write(f"合計チャンク数: {len(text_chunks)}個\n")
        f.write("=" * 80 + "\n\n")
        
        for idx, chunk in enumerate(text_chunks, 1):
            source = chunk.metadata.get("source", "unknown")
            content = chunk.page_content
            
            f.write(f"【チャンク ID: {idx}】\n")
            f.write(f"ソースURL: {source}\n")
            f.write(f"文字数: {len(content)}\n")
            f.write("-" * 80 + "\n")
            f.write(f"内容:\n{content}\n")
            f.write("=" * 80 + "\n\n")


# =============================
# Query処理実行ログをファイルに保存
# =============================
def save_query_results(question, answer, used_docs, result_file="./query_results.txt"):
    """
    Query処理の結果を詳細にファイルに保存
    
    Args:
        question: 質問文
        answer: 生成された回答
        used_docs: 使用されたドキュメント
        result_file: 出力ファイルパス
    """
    with open(result_file, "a", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"【質問】\n{question}\n\n")
        f.write(f"【回答】\n{answer}\n\n")
        f.write(f"【参考ドキュメント数】{len(used_docs)}件\n\n")
        
        for i, doc in enumerate(used_docs, 1):
            source = doc.metadata.get("source", "unknown")
            f.write(f"参考資料 {i}:\n")
            f.write(f"  ソース: {source}\n")
            f.write(f"  内容: {doc.page_content[:500]}...\n\n")
        
        f.write("=" * 80 + "\n\n")


# =============================
# Query処理（何回でも実行）
# =============================
def query_vectordb(question, k_initial=10, k_final=2):
    """
    VectorDBに対して質問し、回答を生成
    
    Args:
        question: 質問文
        k_initial: 初期検索での取得数
        k_final: 最終的に使用するドキュメント数
    
    Returns:
        回答文とリランク済みドキュメント
    """
    
    # ========== VectorDBをロード ==========
    print("[LOAD] VectorDBをロード中...")
    if not Path(VECTORDB_PATH).exists():
        print("❌ VectorDBが見つかりません。先にIngestionを実行してください。")
        return None, None
    
    vectorstore = FAISS.load_local(
        VECTORDB_PATH,
        embeddings,
        allow_dangerous_deserialization=True  # 自作のファイルなので安全
    )
    print("[OK] ドキュメント読み込み完了\n")
    
    # ========== ベクトル検索 ==========
    print(f"[SEARCH] ベクトル検索中（質問: '{question}'）...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": k_initial})
    initial_docs = retriever.invoke(question)
    print(f"[OK] {len(initial_docs)}件の関連ドキュメントを検索\n")
    
    # ========== LLMリランク ==========
    print("[RANK] LLMでリランク中...")
    reranked_docs = rerank_with_llm(question, initial_docs, top_k=k_final)
    print(f"[OK] リランク完了: 上位{len(reranked_docs)}件を選定\n")
    
    # ========== 回答生成 ==========
    print("[GEN] 回答生成中...")
    answer = generate_answer(question, reranked_docs)
    print(f"[OK] 回答生成完了\n")
    
    return answer, reranked_docs


# =============================
# 回答生成（新規関数）
# =============================
def generate_answer(question, documents):
    """
    ドキュメントを基に回答を生成
    
    Args:
        question: 質問文
        documents: 関連ドキュメント
    
    Returns:
        生成された回答
    """
    
    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""以下の質問に以下の情報をベースにして答えてください。

[質問]
{question}

[情報]
{context}

回答:"""
    )
    
    context = "\n\n".join([doc.page_content for doc in documents])
    qa_chain = qa_prompt | llm | StrOutputParser()
    answer = qa_chain.invoke({
        "context": context,
        "question": question
    })
    
    return answer

# =============================
# 再帰的URLローダー
# =============================
def load_recursive_urls(base_url, max_depth=2, timeout=10):
    """
    指定されたベースURL配下のサイトを再帰的に取得
    
    Args:
        base_url: 開始URL
        max_depth: 再帰の最大深さ
        timeout: リクエストタイムアウト（秒）
    
    Returns:
        List[Document]: 抽出されたドキュメント
    """
    try:
        def html_to_text(html):
            """HTMLからテキストを効果的に抽出"""
            soup = BeautifulSoup(html, "html.parser")
            
            # script と style タグを削除
            for script in soup(["script", "style"]):
                script.decompose()
            
            # テキストを抽出
            text = soup.get_text()
            
            # 複数の空行を1行に、先頭・末尾の空白を削除
            lines = (line.strip() for line in text.splitlines())
            text = '\n'.join(line for line in lines if line)
            
            return text
        
        loader = RecursiveUrlLoader(
            url=base_url,
            max_depth=max_depth,
            extractor=html_to_text,
            prevent_outside=True,
            timeout=timeout,
            check_response_status=True,
        )
        
        docs = loader.load()
        return docs
        
    except Exception as e:
        print(f"❌ URLローディングエラー: {e}")
        return []

# =============================
# ドキュメント前処理
# =============================
def preprocess_documents(docs, min_length=100):
    """短すぎるドキュメントを除外し、HTMLタグをさらにクリーニング"""
    filtered_docs = []
    
    for doc in docs:
        # テキストをクリーニング（HTMLタグの残骸を削除）
        content = doc.page_content
        
        # 残っているHTMLタグを削除
        import re
        content = re.sub(r'<[^>]+>', '', content)
        
        # 複数の空白・改行を正規化
        content = re.sub(r'\s+', ' ', content).strip()
        
        # 最小文字数チェック
        if len(content) >= min_length:
            # クリーニング済みコンテンツでドキュメントを更新
            doc.page_content = content
            filtered_docs.append(doc)
    
    print(f"  [OK] {len(filtered_docs)}個のドキュメントをフィルタリング完了")
    return filtered_docs

# =============================
# LLMリランク
# =============================
def rerank_with_llm(question, documents, top_k=5):
    """LLMを使用してドキュメントを関連度でリランク"""
    
    rerank_prompt = PromptTemplate(
        input_variables=["question", "document"],
        template="""あなたは検索結果のリランカーです。
次の文章が、質問にどれだけ関連しているかを0〜10で評価してください。

[質問]
{question}

[文章]
{document}

数字のみで回答してください。"""
    )
    
    rerank_chain = rerank_prompt | llm_rerank | StrOutputParser()
    
    scored_docs = []
    
    for doc in documents:
        try:
            score_str = rerank_chain.invoke({
                "question": question,
                "document": doc.page_content
            })
            score = float(score_str.strip())
        except (ValueError, Exception):
            score = 0
        
        scored_docs.append((score, doc))
    
    scored_docs.sort(reverse=True, key=lambda x: x[0])
    return [doc for score, doc in scored_docs[:top_k]]

# =============================
# 実行
# =============================
if __name__ == "__main__":
    # パラメータ設定
    base_url = "https://www.faq.mizuhobank.co.jp/category/index?site_domain=jcoin"
    max_depth = 2  # 深さを1に制限
    
    # ========== 処理選択 ==========
    print()
    print("=" * 70)
    print("【RAG: Ingestion/Query 分離モード】")
    print("=" * 70)
    print()
    print("【処理を選択してください】")
    print("1. Ingestion（Webサイト取得 → VectorDB保存）")
    print("2. Query（質問 → 回答生成）")
    print()
    
    choice = input("選択してください (1 or 2): ").strip()
    
    if choice == "1":
        # ========== Ingestion処理 ==========
        confirm = input(f"\n{base_url}\nをIngestionしてもよろしいですか？ (yes/no): ").strip().lower()
        
        if confirm == "yes":
            force_refresh = input("既存VectorDBを削除して再取得しますか？ (yes/no): ").strip().lower() == "yes"
            success = ingest_documents(base_url, max_depth=max_depth, force_refresh=force_refresh)
            
            if success:
                print("[DONE] Ingestion完了。次回はQuery（選択肢2）を実行できます。")
        else:
            print("キャンセルされました。")
    
    elif choice == "2":
        # ========== Query処理 ==========
        print()
        print("=" * 70)
        print("【Query処理開始】")
        print("=" * 70)
        print()
        
        while True:
            question = input("\n質問を入力してください (終了は 'exit'): ").strip()
            
            if question.lower() == "exit":
                print("終了します。")
                break
            
            if not question:
                print("質問を入力してください。")
                continue
            
            print()
            answer, used_docs = query_vectordb(question, k_initial=10, k_final=2)
            
            if answer is None:
                break
            
            # Query結果をファイルに保存
            save_query_results(question, answer, used_docs)
            
            print("=" * 70)
            print("【回答】")
            print("=" * 70)
            print(answer)
            print("\n" + "=" * 70)
            print("【参考ドキュメント】")
            print("=" * 70)
            for i, doc in enumerate(used_docs, 1):
                source = doc.metadata.get("source", "unknown")
                print(f"\n{i}. {doc.page_content[:300]}...")
                print(f"   📄 ソース: {source}")
            print()
    
    else:
        print("❌ 無効な選択です。")

