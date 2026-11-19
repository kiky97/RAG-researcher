"""Deterministic retrieval and explicitly labelled offline measurements."""
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def chunks(text: str, size: int = 360, overlap: int = 60) -> list[str]:
    text = text.replace("\x00", "").strip()
    if not text:
        return []
    result = []
    for start in range(0, len(text), size - overlap):
        result.append(text[start:start + size])
        if start + size >= len(text):
            break
    return result


def retrieve(question: str, passages: list[dict], top_k: int, strategy: str):
    if not passages:
        return []
    # Character n-grams support both Chinese and English without a tokenizer model.
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), sublinear_tf=True)
    matrix = vectorizer.fit_transform([p["text"] for p in passages])
    query = vectorizer.transform([question])
    relevance = cosine_similarity(query, matrix).ravel()
    candidates = [int(i) for i in relevance.argsort()[::-1] if relevance[i] > 0]
    selected = []
    while candidates and len(selected) < top_k:
        if strategy == "mmr" and selected:
            diversity = cosine_similarity(matrix[candidates], matrix[selected]).max(axis=1)
            best = max(range(len(candidates)), key=lambda i: .7 * relevance[candidates[i]] - .3 * diversity[i])
        else:
            best = 0
        selected.append(candidates.pop(best))
    return [{**passages[i], "score": round(float(relevance[i]), 4)} for i in selected]


def terms(text: str) -> set[str]:
    text = text.lower()
    latin = re.findall(r"[a-z0-9]+", text)
    han = re.findall(r"[\u4e00-\u9fff]+", text)
    return set(latin + [word[i:i+2] for word in han for i in range(max(1, len(word)-1))])


def offline_metrics(answer: str, expected: str, contexts: list[dict], relevant_sources: list[str]):
    actual, reference = terms(answer), terms(expected)
    intersection = len(actual & reference)
    precision = intersection / len(actual) if actual else 0
    recall = intersection / len(reference) if reference else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    retrieved = {p["source"] for p in contexts}
    return {
        "answer_token_f1": round(f1, 4),
        "source_recall": round(len(retrieved & set(relevant_sources)) / len(set(relevant_sources)), 4) if relevant_sources else None,
        "context_token_coverage": round(len(actual & terms(" ".join(p["text"] for p in contexts))) / len(actual), 4) if actual else 0,
    }


def extract_answer(question: str, contexts: list[dict]) -> str:
    if not contexts:
        return "知识库中未找到相关依据，无法回答。"
    sentences = []
    q = terms(question)
    for index, context in enumerate(contexts, 1):
        for sentence in re.split(r"(?<=[。！？.!?\n])", context["text"]):
            sentence = sentence.strip()
            if sentence:
                sentences.append((len(q & terms(sentence)), sentence, index))
    chosen = sorted(sentences, key=lambda row: row[0], reverse=True)[:3]
    return "\n".join(f"{sentence} [{index}]" for _, sentence, index in chosen)
