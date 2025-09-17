# ベースとなるPostgreSQLイメージ
FROM postgres:16

# pgvectorのインストール
RUN apt-get update && apt-get install -y postgresql-16-pgvector && rm -rf /var/lib/apt/lists/*