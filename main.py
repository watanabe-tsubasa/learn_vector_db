import chromadb
from sentence_transformers import SentenceTransformer

# 文章をベクトルに変換するモデル
model = SentenceTransformer("all-MiniLM-L6-v2")

# # ベクトルDB作成（メモリ上）
# client = chromadb.Client()
# ベクトルDB作成（ローカル永続化）
client = chromadb.PersistentClient(path="./.chromadb_data")

# コレクション（テーブルみたいなもの）
collection = client.get_or_create_collection(name="demo")

# 保存する文章
docs = [
    "I enjoy programming in Rust",
    "JavaScript is used in web development",
    "Python is popular for data science",
    "Coding is fun",
    "I love pizza",
    "Sushi is delicious"
]

# ベクトルDBに登録
if collection.count() == 0:
    # 初回のみベクトル化して登録
    embeddings = model.encode(docs).tolist()
    collection.add(
        documents=docs,
        embeddings=embeddings,
        ids=[str(i) for i in range(len(docs))]
    )

    print("登録完了！")

# 検索クエリ
query = input("検索クエリを入力してください: ")
# クエリをベクトル化
query_embedding = model.encode([query]).tolist()
print("クエリのベクトル:", query_embedding)

# 意味検索
results = collection.query(
    query_embeddings=query_embedding,
    n_results=4
)

print("\n🔍 検索クエリ:", query)
print("🎯 検索結果:", results["documents"][0])
# print("🆔 検索結果の項目:", results.keys())
for key, value in results.items():
    print(f"{key}: {value}")
    print("-----")