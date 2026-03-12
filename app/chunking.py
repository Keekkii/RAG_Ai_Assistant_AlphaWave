from langchain_text_splitters import RecursiveCharacterTextSplitter

PARENT_CHUNK_SIZE = 600
PARENT_OVERLAP = 80
CHILD_CHUNK_SIZE = 150
CHILD_OVERLAP = 20


def chunk_text_parent_child(text: str) -> list[dict]:
    """
    Splits text into parent chunks, then splits each parent into smaller child chunks.
    Returns: [{"parent": str, "children": [str, ...]}, ...]
    """
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_OVERLAP,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_OVERLAP,
    )

    result = []
    for parent in parent_splitter.split_text(text):
        children = child_splitter.split_text(parent)
        if children:
            result.append({"parent": parent, "children": children})
    return result
