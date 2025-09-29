import json
from typing import List, Dict, Tuple

def load_docai_json(response_content: str) -> Tuple[str, List[dict]]:
    doc = json.loads(response_content)
    output = doc.get("output", {})
    return output.get("text", ""), output.get("pages", [])

def extract_text_from_anchor_with_context(text_anchor: Dict, full_text: str) -> str:
    segments = text_anchor.get("textSegments", [])
    return " ".join(full_text[int(s.get("startIndex", 0)):int(s.get("endIndex", 0))] for s in segments if int(s.get("endIndex", 0)) > int(s.get("startIndex", 0)))

def merge_text_from_docai_blocks(full_text: str, pages: List[dict]) -> List[str]:
    merged_texts = []
    for page in pages:
        for block in page.get("blocks", []):
            layout = block.get("layout", {})
            anchor = layout.get("textAnchor", {})
            block_text = extract_text_from_anchor_with_context(anchor, full_text)
            header = block.get("sectionHeader", {}).get("text", "")
            if header and header not in block_text:
                block_text = f"{header}\n{block_text}"
            if block_text:
                merged_texts.append(block_text)
    return merged_texts
