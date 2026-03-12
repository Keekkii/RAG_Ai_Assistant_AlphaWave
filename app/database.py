import psycopg2
from psycopg2.extras import RealDictCursor
from app.embeddings import generate_embedding
from flashrank import Ranker, RerankRequest

reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp/flashrank")


DB_CONFIG = {
    "host": "localhost",
    "port": 5434,
    "dbname": "alphawave_comp",
    "user": "postgres",
    "password": "postgres"
}


def get_connection():
    return psycopg2.connect(
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"],
        dbname=DB_CONFIG["dbname"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        cursor_factory=RealDictCursor
    )


# -------------------------
# Insert document
# -------------------------
def insert_parent_chunk(url: str, title: str, content: str, source_id: int = None) -> int:
    """Insert a parent chunk without an embedding (used for context retrieval only)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO documents (source_id, url, title, content)
        VALUES (%s, %s, %s, %s)
        RETURNING id;
    """, (source_id, url, title, content))
    new_id = cursor.fetchone()["id"]
    conn.commit()
    cursor.close()
    conn.close()
    return new_id


def insert_document(url: str, title: str, content: str, source_id: int = None, parent_id: int = None):
    """Insert a child chunk with an embedding."""
    embedding = generate_embedding(content)
    vector_str = "[" + ",".join(map(str, embedding)) + "]"

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO documents (source_id, parent_id, url, title, content, embedding)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id;
    """, (source_id, parent_id, url, title, content, vector_str))

    new_id = cursor.fetchone()["id"]
    conn.commit()

    cursor.close()
    conn.close()

    return new_id


# -------------------------
# Vector search (parent-child)
# -------------------------
def search_similar_documents(query: str, limit: int = 5):
    embedding = generate_embedding(query)
    vector_str = "[" + ",".join(map(str, embedding)) + "]"

    conn = get_connection()
    cursor = conn.cursor()

    # 1. Search child chunks (small, precise)
    cursor.execute("""
        SELECT id, parent_id, url, title, content,
               (1 - (embedding <=> %s)) as score
        FROM documents
        WHERE embedding IS NOT NULL AND parent_id IS NOT NULL
        ORDER BY embedding <=> %s
        LIMIT 20;
    """, (vector_str, vector_str))
    child_results = cursor.fetchall()

    # 2. Collect unique parent IDs in score order
    seen_parents = set()
    parent_ids = []
    for r in child_results:
        pid = r["parent_id"]
        if pid not in seen_parents:
            seen_parents.add(pid)
            parent_ids.append(pid)

    if not parent_ids:
        cursor.close()
        conn.close()
        return []

    # 3. Fetch parent chunks (full context for LLM)
    cursor.execute(
        "SELECT id, url, title, content FROM documents WHERE id = ANY(%s);",
        (parent_ids,)
    )
    parent_map = {r["id"]: dict(r) for r in cursor.fetchall()}
    cursor.close()
    conn.close()

    # 4. Build candidates: one per unique parent, scored by best child match
    candidates = []
    seen = set()
    for child in child_results:
        pid = child["parent_id"]
        if pid in parent_map and pid not in seen:
            seen.add(pid)
            parent = dict(parent_map[pid])
            parent["score"] = child["score"]
            parent["rrf_score"] = child["score"]
            parent["distance"] = 1 - child["score"]
            candidates.append(parent)

    # 5. Rerank by parent content
    passages = [{"id": i, "text": doc["content"]} for i, doc in enumerate(candidates)]
    rerank_request = RerankRequest(query=query, passages=passages)
    reranked = reranker.rerank(rerank_request)

    final_results = []
    for item in reranked[:limit]:
        doc = candidates[item["id"]]
        doc["rerank_score"] = float(item["score"])
        final_results.append(doc)

    return final_results


# -------------------------
# Source management
# -------------------------
def create_source(filename: str, title: str) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO sources (filename, title, status) VALUES (%s, %s, 'pending') RETURNING id;",
        (filename, title)
    )
    source_id = cursor.fetchone()["id"]
    conn.commit()
    cursor.close()
    conn.close()
    return source_id


def finalize_source(source_id: int, chunk_count: int, status: str = "done"):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE sources SET chunk_count = %s, status = %s WHERE id = %s;",
        (chunk_count, status, source_id)
    )
    conn.commit()
    cursor.close()
    conn.close()


def source_already_ingested(filename: str) -> bool:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id FROM sources WHERE filename = %s AND status = 'done';",
        (filename,)
    )
    exists = cursor.fetchone() is not None
    cursor.close()
    conn.close()
    return exists


# -------------------------
# Manual test
# -------------------------
if __name__ == "__main__":
    try:
        query = "What is the DPP-Compliant Asset Management Platform?"

        results = search_similar_documents(query)

        print("\nSearch Results:\n")
        for r in results:
            print(f"ID: {r['id']}")
            print(f"Title: {r['title']}")
            print(f"Distance: {r['distance']}")
            print(f"Content Preview: {r['content'][:150]}...")
            print("-" * 50)

    except Exception as e:
        print("Operation failed:")
        print(e)