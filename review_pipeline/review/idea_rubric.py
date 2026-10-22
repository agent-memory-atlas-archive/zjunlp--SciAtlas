from __future__ import annotations

import concurrent.futures
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from kg_search.service_hosted import hosted_kg_search
from .common import (
    LEGACY_SECTION_ALIASES,
    REVIEW_SECTIONS,
    RUBRIC_STANDARD_KEYS,
    ensure_directory,
    first_non_empty,
    load_env_values,
    normalize_whitespace,
    slugify,
    write_json,
)
from .llm_retry import call_llm_json_with_retry


DEFAULT_LLM_BASE_URL = "https://www.dmxapi.cn/v1"
DEFAULT_LLM_MODEL_NAME = "DeepSeek-V3.2"
DEFAULT_LLM_TEMPERATURE = 0.3
DEFAULT_SEARCH_TOP_K = 50
DEFAULT_SEARCH_FINAL_K = 15
DEFAULT_MAX_WORKERS = 8
DEFAULT_RERANK_BATCH_SIZE = 5
PERSONA_PROMPTS_PATH = Path(__file__).with_name("persona_prompts.json")
RUBRIC_SCHEMA_VERSION = 5
RERANK_SYSTEM_PROMPT = (
    "You are a rigorous academic literature relevance judge. "
    "Score each candidate paper independently and return strict JSON only."
)
ABSTRACT_REVIEW_NORMS_SYSTEM_PROMPT = """You are a rigorous senior reviewer for a top journal in this paper's field.
You are given only a candidate paper's title, abstract, and publication metadata, plus reviewer-persona notes describing how demanding reviewers evaluate submissions.
Write the review norms such reviewers would apply when judging work like this paper: the concrete checks, evidence demands, and methodological expectations.
Rules:
1. Ground every norm in what the abstract and persona notes actually support; do not hallucinate paper details.
2. Each norm must be a single actionable demand, phrased in professional reviewer language.
3. Prefer field-specific terminology over generic labels such as "novelty" or "clarity" alone.
4. Produce 8 to 12 norms.
5. If a norm mentions a formula, variable, or metric, use Markdown math notation such as `$x$` or `$$ ... $$`, never backticks.
All output MUST be strictly in English."""
ABSTRACT_DIMENSIONS_SYSTEM_PROMPT = """You are an expert reviewer.
You are given a candidate paper's title, abstract summary, and a set of review norms describing how demanding reviewers in this field judge submissions.
Extract reviewer evaluation standards for four modules: Introduction, Methods, Results, and Discussion.
Rules:
1. Ground each standard in what the abstract and review norms actually support; do not hallucinate specifics.
2. Each standard must be concrete and anchored to this paper's problem framing, method claims, evaluation expectations, or limitations.
3. Use professional reviewer language; avoid broad standalone labels such as "novelty", "clarity", or "effectiveness" alone.
4. Produce 2 to 4 standards per module.
5. If a standard mentions a formula, variable, or metric, use Markdown math notation such as `$x$` or `$$ ... $$`, never backticks.
All output MUST be strictly in English."""
GENERAL_RUBRIC_SYSTEM_PROMPT = """You are a senior academic reviewer for a top journal.
CREATE A PROFESSIONAL EVALUATION RUBRIC divided into Motivation, Method, Result, and Discussion.
Synthesize historical standards and adjust them tightly to the user's idea. The rubric should be balanced across the four sections and should not read like generic journal-review criteria.
Each section should contain 3 to 5 technically specific standards. Prefer standards grounded in the idea's concrete problem framing, mechanism, assumptions, data, evaluation plan, expected result claims, limitations, or deployment context.
Avoid broad standalone dimension names such as "Novelty", "Clarity", "Significance", "Validity", or "Effectiveness". If such a concept is needed, make it domain- and idea-specific.
If a rubric item mentions a formula, variable, metric definition, or symbolic mechanism, express it using Markdown math notation such as `$x$`, `$R = a/b$`, or `$$ ... $$` rather than plain-text equations.
Never wrap math expressions in backticks or code formatting. Use `$...$` or `$$...$$` only.
Because the response must be JSON, every backslash inside a string must be doubled. For example, write `\\\\alpha` or `\\\\hat{x}`, never `\\alpha` or `\\hat{x}`.
All output MUST be strictly in English."""

DETAILED_RUBRIC_SYSTEM_PROMPT = """You are an expert reviewer.
The user provided a Research Idea text. Create a strict list of idea-specific review rules.
IMPORTANT RULES:
1. Only use what is clearly written in the text.
2. DO NOT divide into sections. Put ALL rules into a single list.
3. Use professional, domain-appropriate reviewer language instead of generic or simplified wording.
4. Prefer technical terms, model/component names, optimization objectives, evaluation protocols, theoretical assumptions, and metric names that are explicitly present in the idea.
5. Each rule must be tightly anchored to a concrete mechanism, claim, experimental setup, or failure mode in the idea. Avoid generic criteria that could apply to almost any paper.
6. When possible, name the rule using specialist terminology rather than broad labels such as "novelty", "clarity", or "effectiveness" alone.
7. If the idea contains formulas, variable names, metrics, or symbolic expressions, preserve them using Markdown math notation such as `$x$`, `$dE/dW$`, or `$$ ... $$`.
8. Prefer plain-English names for symbolic expressions when possible, for example "z-hat" instead of a LaTeX command.
9. Never wrap math expressions in backticks or code formatting. Use `$...$` or `$$...$$` only.
10. Because the response must be JSON, every backslash inside a string must be doubled. For example, write `\\\\alpha` or `\\\\hat{x}`, never `\\alpha` or `\\hat{x}`.
All output MUST be strictly in English."""

SECTIONED_SYNTHESIS_SYSTEM_PROMPT = """You are a Senior Reviewer and Area Expert for a top science journal.
You are provided with a user's Research Idea and TWO preliminary review rubrics:
1. [General Rubric]: classified by Motivation, Method, Result, and Discussion.
2. [Detailed Rubric]: a strict, non-hallucinated list of idea-specific rules.

Your task: create the final selected review standards for the report rubric, grouped into exactly four sections: Motivation, Method, Result, and Discussion.

CRITICAL RULES:
1. Use the Detailed Rubric as the main source for idea-specific constraints.
2. Use the General Rubric only when it contains a clearly idea-relevant and technically meaningful standard.
3. Refine and deduplicate overlapping standards.
4. For every final standard, assign source_tag strictly as "General" or "Detailed".
   If a standard merges overlapping concepts from both rubrics, tag it "Detailed".
5. For every final standard, assign target_section strictly as one of "Motivation", "Method", "Result", or "Discussion".
6. All Detailed standards must be assigned to one of the four sections; there is no Idea-Specific section.
7. Keep the four sections balanced. Produce 3 to 5 final standards per section unless the idea lacks enough evidence for a section.
8. The dimension_name MUST NOT contain the "&" symbol and MUST focus on one single concept.
9. Do not hallucinate concepts not present in the Research Idea.
10. Standards must be professional and idea-specific. Avoid generic journal-review clichés.
11. Avoid broad standalone dimension names such as "Novelty", "Clarity", "Significance", "Validity", or "Effectiveness". If such a concept is necessary, narrow it to a concrete mechanism, claim, experimental protocol, or literature contrast in the idea.
12. Motivation standards should focus on problem framing, literature positioning, domain need, and the specificity of the claimed contribution.
13. Method standards should focus on mechanism, algorithmic design, assumptions, privacy/safety constraints, implementation details, and theoretical consistency.
14. Result standards should focus on evaluation protocols, datasets, baselines, metrics, ablations, statistical reliability, and whether the expected evidence can support the claims.
15. Discussion standards should focus on limitations, boundary conditions, failure modes, generalizability, deployment implications, ethical/societal risks, and future work.
16. If required_evidence mentions formulas, variables, metrics, or symbolic definitions, write them using Markdown math notation rather than plain-text equations.
17. Never wrap math expressions in backticks or code formatting. Use `$...$` or `$$...$$` only.
18. Because the response must be JSON, every backslash inside a string must be doubled. For example, write `\\\\alpha` or `\\\\hat{x}`, never `\\alpha` or `\\hat{x}`.
All output MUST be strictly in English."""


class PaperSummary(BaseModel):
    paper_title: str = Field(description="The full title of the paper. Provide 'Unknown' if not explicitly stated.")
    abstract_content: list[str] = Field(description="Factual summary of the Abstract.")
    introduction_content: list[str] = Field(description="Factual summary of the Introduction section.")
    methods_content: list[str] = Field(description="Factual summary of the Methods section.")
    results_content: list[str] = Field(description="Factual summary of the Results section.")
    discussion_content: list[str] = Field(description="Factual summary of the Discussion/Conclusion section.")


class ExtractedDimension(BaseModel):
    dimension_name: str = Field(description="A simple and clear name for this rule.")
    core_reason: str = Field(description="Why does the reviewer care about this?")
    required_evidence: list[str] = Field(description="What exact proof did the reviewer ask for?")


class ModuleStandards(BaseModel):
    standards: list[ExtractedDimension] = Field(description="List of evaluation standards for this module.")


class ReviewStandard(BaseModel):
    dimension_name: str = Field(description="A concise evaluation dimension.")
    core_philosophy: str = Field(description="Why reviewers will care about this dimension.")
    required_evidence: str = Field(description="The evidence needed to justify this dimension.")


class GeneralReviewRubric(BaseModel):
    Idea_Summary: str = Field(description="A brief one-sentence summary of the proposed idea.")
    Motivation_Standards: list[ReviewStandard] = Field(description="Professional criteria for Motivation.")
    Method_Standards: list[ReviewStandard] = Field(description="Professional criteria for Method.")
    Result_Standards: list[ReviewStandard] = Field(description="Professional criteria for Result.")
    Discussion_Standards: list[ReviewStandard] = Field(description="General criteria for Discussion/Conclusion.")


class DetailedUnifiedRubric(BaseModel):
    Evaluation_Standards: list[ReviewStandard] = Field(
        description="A strict flat list of idea-specific review rules. DO NOT divide into paper sections."
    )


class IdeaBreakdown(BaseModel):
    motivation_and_problem: str = Field(
        description="The main problem the idea wants to solve. If not found, write 'Not mentioned'."
    )
    proposed_method: str = Field(
        description="The exact method or model in the idea. If not found, write 'Not mentioned'."
    )
    experiment_and_data: str = Field(
        description="The dataset or tests clearly written in the idea. If not found, write 'Not mentioned'."
    )


class RefinedStandard(BaseModel):
    dimension_name: str = Field(
        description="A clear rule name. DO NOT use the '&' symbol. Each name must focus on one single concept."
    )
    core_philosophy: str = Field(description="Why this rule is important.")
    required_evidence: str = Field(description="What the author must do to pass this rule.")
    source_tag: str = Field(description="The origin of this standard. Must be strictly 'General' or 'Detailed'.")
    target_section: str = Field(
        description=(
            "The section this standard belongs to. Must be strictly one of "
            "'Motivation', 'Method', 'Result', or 'Discussion'."
        )
    )


class SectionedSynthesisRubric(BaseModel):
    Idea_Summary: str = Field(description="A brief one-sentence summary of the proposed idea.")
    Idea_Breakdown: IdeaBreakdown = Field(description="Break down the idea to understand it clearly.")
    Motivation_Standards: list[RefinedStandard] = Field(description="Final standards for Motivation.")
    Method_Standards: list[RefinedStandard] = Field(description="Final standards for Method.")
    Result_Standards: list[RefinedStandard] = Field(description="Final standards for Result.")
    Discussion_Standards: list[RefinedStandard] = Field(description="Final standards for Discussion.")


@dataclass
class IdeaRubricConfig:
    target_idea: str
    artifact_root: Path
    output_path: Path
    sources_root: Path | None = None
    env_path: Path | None = None
    llm_api_key: str | None = None
    llm_base_url: str = ""
    llm_model_name: str = ""
    llm_temperature: float = DEFAULT_LLM_TEMPERATURE
    llm_timeout_seconds: int = 300
    search_top_k: int = DEFAULT_SEARCH_TOP_K
    search_final_k: int = DEFAULT_SEARCH_FINAL_K
    max_workers: int = DEFAULT_MAX_WORKERS
    rerank_batch_size: int = DEFAULT_RERANK_BATCH_SIZE


class RerankPaperScore(BaseModel):
    paper_index: int = Field(description="1-based index of the candidate paper in the batch.")
    score: float = Field(description="Relevance score in [0, 10].")
    reason: str = Field(default="", description="Short reason for the score.")


class RerankBatchScores(BaseModel):
    papers: list[RerankPaperScore] = Field(description="One entry per candidate paper in the batch.")


class ReviewNorms(BaseModel):
    norms: list[str] = Field(description="Review norms, each a single actionable reviewer demand.")


class PaperModuleDimensions(BaseModel):
    introduction_standards: list[ExtractedDimension] = Field(description="Standards for the Introduction module.")
    methods_standards: list[ExtractedDimension] = Field(description="Standards for the Methods module.")
    results_standards: list[ExtractedDimension] = Field(description="Standards for the Results module.")
    discussion_standards: list[ExtractedDimension] = Field(description="Standards for the Discussion module.")


def _compact_text(value: Any) -> str:
    return normalize_whitespace(value)


def _env_lookup(env_values: dict[str, str], *names: str) -> str:
    for name in names:
        value = normalize_whitespace(os.environ.get(name))
        if value:
            return value
        value = normalize_whitespace(env_values.get(name))
        if value:
            return value
    return ""


def _summarize_general_rubric_for_synthesis(general_result: dict[str, Any]) -> str:
    section_keys = [(section, RUBRIC_STANDARD_KEYS[section]) for section in REVIEW_SECTIONS]
    lines: list[str] = []
    summary = _compact_text(general_result.get("Idea_Summary"))
    if summary:
        lines.append(f"Idea Summary: {summary}")
    for section_name, key in section_keys:
        lines.append(f"{section_name}:")
        standards = general_result.get(key) if isinstance(general_result.get(key), list) else []
        if not standards:
            lines.append("- No standards.")
            continue
        for standard in standards:
            if not isinstance(standard, dict):
                continue
            dimension = _compact_text(standard.get("dimension_name")) or "Unnamed dimension"
            philosophy = _compact_text(standard.get("core_philosophy"))
            evidence = _compact_text(standard.get("required_evidence"))
            line = f"- {dimension}"
            if philosophy:
                line += f": {philosophy}"
            if evidence:
                line += f" | Evidence: {evidence}"
            lines.append(line)
    return "\n".join(lines)


def _summarize_detailed_rubric_for_synthesis(detailed_result: dict[str, Any]) -> str:
    lines = ["Detailed standards:"]
    standards = detailed_result.get("Evaluation_Standards") if isinstance(detailed_result.get("Evaluation_Standards"), list) else []
    if not standards:
        lines.append("- No standards.")
        return "\n".join(lines)
    for standard in standards:
        if not isinstance(standard, dict):
            continue
        dimension = _compact_text(standard.get("dimension_name")) or "Unnamed dimension"
        philosophy = _compact_text(standard.get("core_philosophy"))
        evidence = _compact_text(standard.get("required_evidence"))
        line = f"- {dimension}"
        if philosophy:
            line += f": {philosophy}"
        if evidence:
            line += f" | Evidence: {evidence}"
        lines.append(line)
    return "\n".join(lines)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


GENERAL_SECTION_KEYS = tuple(RUBRIC_STANDARD_KEYS[section] for section in REVIEW_SECTIONS)
GENERAL_SECTION_NAME_TO_KEY = dict(RUBRIC_STANDARD_KEYS)


def _canonical_rubric_section(value: Any) -> str:
    text = normalize_whitespace(value)
    if not text:
        return ""
    for section in REVIEW_SECTIONS:
        if text.casefold() == section.casefold():
            return section
    for legacy, canonical in LEGACY_SECTION_ALIASES.items():
        if text.casefold() == legacy.casefold():
            return canonical
    return ""


def _standards_with_source(payload: dict[str, Any], key: str, source_tag: str) -> list[dict[str, Any]]:
    standards = payload.get(key) if isinstance(payload.get(key), list) else []
    tagged: list[dict[str, Any]] = []
    for standard in standards:
        if not isinstance(standard, dict):
            continue
        item = dict(standard)
        item["source_tag"] = source_tag
        tagged.append(item)
    return tagged


def _clean_refined_standard(raw: dict[str, Any]) -> dict[str, Any]:
    standard = dict(raw)
    source_tag = normalize_whitespace(standard.get("source_tag"))
    if source_tag not in {"General", "Detailed"}:
        source_tag = "Detailed"
    standard["source_tag"] = source_tag
    target_section = _canonical_rubric_section(standard.get("target_section"))
    standard["target_section"] = target_section
    dimension_name = normalize_whitespace(standard.get("dimension_name")).replace("&", "and")
    if dimension_name:
        standard["dimension_name"] = dimension_name
    return standard


def _standard_dedup_key(raw: dict[str, Any]) -> str:
    return normalize_whitespace(
        " | ".join(
            [
                normalize_whitespace(raw.get("dimension_name")).lower(),
                normalize_whitespace(raw.get("core_philosophy")).lower(),
                normalize_whitespace(raw.get("required_evidence")).lower(),
            ]
        )
    )


def _build_canonical_rubric(
    *,
    target_idea: str,
    general_result: dict[str, Any],
    synthesis_result: dict[str, Any],
) -> dict[str, Any]:
    min_per_section = 3
    max_per_section = 5
    rubric: dict[str, Any] = {
        "schema_version": RUBRIC_SCHEMA_VERSION,
        "Idea_Summary": normalize_whitespace(synthesis_result.get("Idea_Summary"))
        or normalize_whitespace(general_result.get("Idea_Summary"))
        or normalize_whitespace(target_idea)[:500],
        "Idea_Breakdown": synthesis_result.get("Idea_Breakdown")
        if isinstance(synthesis_result.get("Idea_Breakdown"), dict)
        else None,
    }
    candidates_by_key: dict[str, list[dict[str, Any]]] = {}
    for key in GENERAL_SECTION_KEYS:
        candidates_by_key[key] = _standards_with_source(general_result, key, "General")
        rubric[key] = []

    seen_keys_by_section: dict[str, set[str]] = {key: set() for key in GENERAL_SECTION_KEYS}

    for section, key in GENERAL_SECTION_NAME_TO_KEY.items():
        raw_standards = synthesis_result.get(key) if isinstance(synthesis_result.get(key), list) else []
        for standard in raw_standards:
            if not isinstance(standard, dict):
                continue
            cleaned = _clean_refined_standard({**standard, "target_section": section})
            dedup_key = _standard_dedup_key(cleaned)
            if not dedup_key or dedup_key in seen_keys_by_section[key]:
                continue
            if len(rubric[key]) >= max_per_section:
                continue
            seen_keys_by_section[key].add(dedup_key)
            rubric[key].append(cleaned)

    for key in GENERAL_SECTION_KEYS:
        if len(rubric[key]) >= min_per_section:
            continue
        for standard in candidates_by_key[key]:
            if len(rubric[key]) >= min_per_section:
                break
            if not isinstance(standard, dict):
                continue
            dedup_key = _standard_dedup_key(standard)
            if not dedup_key or dedup_key in seen_keys_by_section[key]:
                continue
            section_name = next((section for section, section_key in GENERAL_SECTION_NAME_TO_KEY.items() if section_key == key), "")
            rubric[key].append(_clean_refined_standard({**standard, "target_section": section_name}))
            seen_keys_by_section[key].add(dedup_key)

    for key in GENERAL_SECTION_KEYS:
        rubric[key] = rubric[key][:max_per_section]
    return rubric


def _build_rerank_prompt(idea: str, batch: list[dict[str, Any]]) -> str:
    paper_blocks = []
    for slot, doc in enumerate(batch, start=1):
        abstract = normalize_whitespace(doc.get("Abstract")) or "[missing abstract]"
        paper_blocks.append(f"[{slot}] Title: {normalize_whitespace(doc.get('Title'))}\nAbstract: {abstract}")
    return "\n\n".join(
        [
            "Evaluate how substantively relevant each candidate paper is to the research idea below.",
            "",
            "Scoring target:",
            "- Judge each paper independently, not comparatively.",
            "- Focus on overlap in the core research problem, evaluation target, inputs/outputs, and methodological purpose.",
            "- A paper can score high even if the method differs, as long as it directly addresses the same problem.",
            "- Penalize papers that only share broad themes such as LLMs, scientific discovery, benchmarks, or evaluation.",
            "",
            "Rubric:",
            "- 9-10: Extremely close neighbor; essentially the same problem.",
            "- 7-8: Directly relevant; strong task or evaluation overlap.",
            "- 5-6: Meaningful partial overlap, but not the same core problem.",
            "- 3-4: Weak topical relation only.",
            "- 0-2: Effectively irrelevant.",
            "",
            'Return ONLY JSON in this format: {"papers":[{"paper_index":1,"score":8.7,"reason":"short reason"}]}',
            "",
            "Requirements for output:",
            "- Include every paper exactly once.",
            "- score must be a number in [0, 10].",
            "- reason must be concise and mention the main match or mismatch.",
            "",
            f"Research idea:\n{normalize_whitespace(idea)}",
            "Candidate papers:\n" + "\n\n".join(paper_blocks),
        ]
    )


def _load_general_persona_notes(limit: int = 8) -> list[str]:
    try:
        payload = json.loads(PERSONA_PROMPTS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    entries = payload.get("groups", {}).get("general", []) if isinstance(payload.get("groups"), dict) else []
    notes: list[str] = []
    for item in entries[:limit]:
        if isinstance(item, dict):
            note = normalize_whitespace(item.get("text"))
            if note:
                notes.append(note)
    return notes


class KgRubricSearch:
    """Hosted KG retrieval plus LLM relevance reranking for rubric sources.

    Retrieval always goes through the hosted SciAtlas KG (/v1/search); relevance
    reranking is done by the pipeline LLM in small batches, so no local
    embedding or cross-encoder models are required.
    """

    def __init__(self, config: IdeaRubricConfig, *, call_llm: Any) -> None:
        self.config = config
        self._call_llm = call_llm

    def search(self, idea: str, *, top_k: int, final_k: int) -> list[dict[str, Any]]:
        search = hosted_kg_search(query_text=idea, top_k=top_k, env_path=self.config.env_path)
        docs = self._map_papers(idea, search["papers"])
        if not docs:
            raise RuntimeError("hosted KG search returned no candidate papers")
        return self._rerank(idea, docs)[:final_k]

    def _map_papers(self, idea: str, papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        idea_lower = idea.strip().lower()
        docs: list[dict[str, Any]] = []
        seen_titles: set[str] = set()
        for paper in papers:
            title = normalize_whitespace(paper.get("title"))
            if not title or title.casefold() in seen_titles:
                continue
            if title.lower() in idea_lower:
                continue
            seen_titles.add(title.casefold())
            docs.append(
                {
                    "Title": title,
                    "Abstract": normalize_whitespace(paper.get("abstract")),
                    "DOI": normalize_whitespace(paper.get("doi")),
                    "venue": normalize_whitespace(paper.get("venue")),
                    "publication_year": paper.get("publication_year"),
                    "authors": paper.get("authors") or [],
                    "paper_id": normalize_whitespace(paper.get("id")),
                    "url": normalize_whitespace(paper.get("url")),
                    "pdf_url": normalize_whitespace(paper.get("pdf_url")),
                    "cited_by_count": paper.get("cited_by_count"),
                    "kg_score": paper.get("score"),
                    "source": "kg_hosted",
                }
            )
        return docs

    def _rerank(self, idea: str, docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        batch_size = max(2, int(self.config.rerank_batch_size or DEFAULT_RERANK_BATCH_SIZE))
        batches = [docs[start : start + batch_size] for start in range(0, len(docs), batch_size)]
        max_parallel = max(1, min(int(self.config.max_workers or DEFAULT_MAX_WORKERS), len(batches)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
            futures = [executor.submit(self._score_batch, idea, batch) for batch in batches]
            for future in concurrent.futures.as_completed(futures):
                future.result()

        def sort_key(doc: dict[str, Any]) -> tuple[int, float, float]:
            llm_score = doc.get("llm_score")
            if llm_score is None:
                return (0, 0.0, float(doc.get("kg_score") or 0.0))
            return (1, float(llm_score), 0.0)

        return sorted(docs, key=sort_key, reverse=True)

    def _score_batch(self, idea: str, batch: list[dict[str, Any]]) -> None:
        prompt = _build_rerank_prompt(idea, batch)
        result = self._call_llm(
            system_prompt=RERANK_SYSTEM_PROMPT,
            user_content=prompt,
            response_model_schema=RerankBatchScores,
            temperature=0.1,
            label="rubric_rerank",
        )
        if not isinstance(result, dict):
            print("[KgRubricSearch] rerank batch failed; keeping KG order for its papers", file=sys.stderr)
            return
        for item in result.get("papers", []):
            if not isinstance(item, dict):
                continue
            try:
                slot = int(item.get("paper_index"))
                score = float(item.get("score"))
            except (TypeError, ValueError):
                continue
            if slot < 1 or slot > len(batch):
                continue
            batch[slot - 1]["llm_score"] = round(max(0.0, min(10.0, score)), 4)
            batch[slot - 1]["rerank_reason"] = normalize_whitespace(item.get("reason"))


class IdeaRubricRunner:
    def __init__(self, config: IdeaRubricConfig) -> None:
        self.config = self._resolve_config(config)
        self._timing_lock = threading.Lock()
        self.timing_log_path = self.config.artifact_root / "timing.log"
        if not self.config.target_idea:
            raise ValueError("target_idea must not be empty")
        if not self.config.llm_api_key:
            raise ValueError("LLM API key is required for idea rubric generation")

    def _log_timing(self, event: str, **fields: Any) -> None:
        payload = {"ts": round(time.time(), 3), "event": event, **fields}
        line = json.dumps(payload, ensure_ascii=False)
        with self._timing_lock:
            with self.timing_log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def _resolve_config(self, config: IdeaRubricConfig) -> IdeaRubricConfig:
        env_values = load_env_values(config.env_path)
        llm_api_key = first_non_empty(
            config.llm_api_key,
            _env_lookup(env_values, "LLM_API_KEY", "DMX-API-KEY", "DMX_API_KEY", "OPENAI_API_KEY"),
        )
        return IdeaRubricConfig(
            target_idea=normalize_whitespace(config.target_idea),
            artifact_root=Path(config.artifact_root).expanduser().resolve(),
            output_path=Path(config.output_path).expanduser().resolve(),
            sources_root=Path(config.sources_root).expanduser().resolve() if config.sources_root else None,
            env_path=Path(config.env_path).expanduser().resolve() if config.env_path else None,
            llm_api_key=llm_api_key or None,
            llm_base_url=first_non_empty(
                config.llm_base_url,
                _env_lookup(env_values, "LLM_BASE_URL", "OPENAI_BASE_URL"),
                DEFAULT_LLM_BASE_URL,
            ),
            llm_model_name=first_non_empty(
                config.llm_model_name,
                _env_lookup(env_values, "LLM_MODEL", "OPENAI_MODEL"),
                DEFAULT_LLM_MODEL_NAME,
            ),
            llm_temperature=float(config.llm_temperature),
            llm_timeout_seconds=int(config.llm_timeout_seconds),
            search_top_k=int(config.search_top_k),
            search_final_k=int(config.search_final_k),
            max_workers=int(config.max_workers),
            rerank_batch_size=int(config.rerank_batch_size),
        )

    @property
    def sources_root(self) -> Path:
        return self.config.sources_root or self.config.artifact_root

    def call_llm_with_retry(
        self,
        *,
        system_prompt: str,
        user_content: str,
        response_model_schema: Any,
        max_retries: int = 3,
        temperature: float | None = None,
        label: str | None = None,
    ) -> dict[str, Any] | None:
        effective_temperature = self.config.llm_temperature if temperature is None else temperature
        return call_llm_json_with_retry(
            api_key=self.config.llm_api_key or "",
            base_url=self.config.llm_base_url,
            model_name=self.config.llm_model_name,
            system_prompt=system_prompt,
            user_content=user_content,
            response_model_schema=response_model_schema,
            timeout_seconds=self.config.llm_timeout_seconds,
            max_retries=max_retries,
            temperature=effective_temperature,
            max_tokens=8192,
            helper_path=Path(__file__).with_name("llm_json_call_worker.py"),
            label=label,
            debug_dir=self.config.artifact_root / "llm_failures",
            log_attempt=lambda **kwargs: self._log_timing("llm_call", **kwargs),
        )

    def _build_doc_artifact_dir(self, index: int, doc: dict[str, Any]) -> Path:
        title = normalize_whitespace(doc.get("Title")) or normalize_whitespace(doc.get("title")) or f"paper-{index}"
        return ensure_directory(self.sources_root / "papers" / f"{index:02d}_{slugify(title, limit=60)}")

    def _build_review_norms_content(self, doc: dict[str, Any]) -> str:
        parts = [f"Paper title: {normalize_whitespace(doc.get('Title'))}"]
        venue = normalize_whitespace(doc.get("venue"))
        year = doc.get("publication_year")
        if venue:
            parts.append(f"Venue: {venue}")
        if year not in (None, ""):
            parts.append(f"Publication year: {year}")
        cited_by = doc.get("cited_by_count")
        if cited_by not in (None, ""):
            parts.append(f"Citation count: {cited_by}")
        abstract = normalize_whitespace(doc.get("Abstract")) or "[missing abstract]"
        parts.append(f"Paper abstract:\n{abstract}")
        persona_notes = _load_general_persona_notes()
        if persona_notes:
            parts.append("Reviewer-persona notes (how demanding reviewers evaluate submissions):\n" + "\n".join(f"- {note}" for note in persona_notes))
        return "\n\n".join(parts)

    def process_single_retrieved_paper_abstract(self, doc: dict[str, Any]) -> dict[str, Any]:
        paper_started_at = time.perf_counter()
        artifact_dir = Path(doc["artifact_dir"])
        title = normalize_whitespace(doc.get("Title"))
        timings_ms: dict[str, Any] = {}

        def finish(status: str, reason: str | None = None) -> dict[str, Any]:
            elapsed_ms = round((time.perf_counter() - paper_started_at) * 1000, 1)
            payload = {
                "title": title,
                "artifact_dir": str(artifact_dir),
                "status": status,
                "elapsed_ms": elapsed_ms,
                "timings_ms": timings_ms,
            }
            if reason:
                payload["reason"] = reason
            self._log_timing(
                "paper_done",
                title=title,
                artifact_dir=str(artifact_dir),
                status=status,
                elapsed_ms=elapsed_ms,
                reason=reason,
            )
            return payload

        abstract = normalize_whitespace(doc.get("Abstract"))
        sections_path = artifact_dir / "extracted_sections.json"
        review_text_path = artifact_dir / "review_text.txt"
        source_meta_path = artifact_dir / "source_meta.json"

        write_json(sections_path, {"abstract": abstract, "full_text_available": False})

        if not review_text_path.exists():
            step_started_at = time.perf_counter()
            norms_result = self.call_llm_with_retry(
                system_prompt=ABSTRACT_REVIEW_NORMS_SYSTEM_PROMPT,
                user_content=self._build_review_norms_content(doc),
                response_model_schema=ReviewNorms,
                temperature=0.2,
                label=f"norms:{title}",
            )
            if isinstance(norms_result, dict) and norms_result.get("norms"):
                norms = [normalize_whitespace(item) for item in norms_result["norms"] if normalize_whitespace(item)]
            else:
                norms = []
            if not norms:
                # Deterministic degradation: reuse the general reviewer personas
                # shipped with the repo instead of skipping the paper entirely.
                norms = _load_general_persona_notes(limit=10)
                timings_ms["norms_source"] = "persona_fallback"
            if not norms:
                return finish("skipped", "review norms generation failed")
            _write_text(review_text_path, "\n".join(f"- {norm}" for norm in norms))
            timings_ms["review_norms"] = round((time.perf_counter() - step_started_at) * 1000, 1)

        if not source_meta_path.exists():
            write_json(
                source_meta_path,
                {
                    "title": title,
                    "doi": normalize_whitespace(doc.get("DOI")),
                    "venue": normalize_whitespace(doc.get("venue")),
                    "publication_year": doc.get("publication_year"),
                    "paper_id": normalize_whitespace(doc.get("paper_id")),
                    "url": normalize_whitespace(doc.get("url")),
                    "pdf_url": normalize_whitespace(doc.get("pdf_url")),
                    "kg_score": doc.get("kg_score"),
                    "llm_score": doc.get("llm_score"),
                    "folder_name": normalize_whitespace(doc.get("folder_name")),
                    "source": "kg_hosted",
                    "artifact_dir": str(artifact_dir.resolve()),
                },
            )

        return finish("ok")
    def process_single_retrieved_paper_llm(self, doc: dict[str, Any]) -> dict[str, Any]:
        paper_started_at = time.perf_counter()
        artifact_dir = Path(doc["artifact_dir"])
        title = normalize_whitespace(doc.get("Title") or doc.get("title"))
        timings_ms: dict[str, Any] = {}

        def finish(status: str, reason: str | None = None) -> dict[str, Any]:
            elapsed_ms = round((time.perf_counter() - paper_started_at) * 1000, 1)
            payload = {
                "title": title,
                "artifact_dir": str(artifact_dir),
                "status": status,
                "elapsed_ms": elapsed_ms,
                "timings_ms": timings_ms,
            }
            if reason:
                payload["reason"] = reason
            self._log_timing(
                "paper_llm_done",
                title=title,
                artifact_dir=str(artifact_dir),
                status=status,
                elapsed_ms=elapsed_ms,
                reason=reason,
            )
            return payload

        sections_path = Path(normalize_whitespace(doc.get("sections_path"))) if normalize_whitespace(doc.get("sections_path")) else artifact_dir / "extracted_sections.json"
        review_text_path = Path(normalize_whitespace(doc.get("review_text_path"))) if normalize_whitespace(doc.get("review_text_path")) else artifact_dir / "review_text.txt"
        summary_path = artifact_dir / "summary.json"
        dimensions_path = artifact_dir / "dimensions.json"

        if summary_path.exists() and dimensions_path.exists():
            return finish("ok")

        if not sections_path.exists():
            return finish("skipped", "sections file missing")
        if not review_text_path.exists():
            return finish("skipped", "review text file missing")

        try:
            sections_payload = json.loads(sections_path.read_text(encoding="utf-8"))
        except Exception:
            sections_payload = {}
        abstract_only = sections_payload.get("full_text_available") is False if isinstance(sections_payload, dict) else False

        if not summary_path.exists():
            step_started_at = time.perf_counter()
            if abstract_only:
                # KG candidates carry only the abstract; the deterministic summary
                # avoids a per-paper LLM round trip.
                abstract_text = normalize_whitespace(sections_payload.get("abstract")) if isinstance(sections_payload, dict) else ""
                summary_data = {
                    "paper_title": title or "Unknown",
                    "abstract_content": [abstract_text] if abstract_text else [],
                    "introduction_content": [],
                    "methods_content": [],
                    "results_content": [],
                    "discussion_content": [],
                }
            else:
                sections_text = sections_path.read_text(encoding="utf-8")
                summary_data = self.call_llm_with_retry(
                    system_prompt="Extract factual summaries for EXACTLY 5 specific sections based on the provided JSON.",
                    user_content=sections_text,
                    response_model_schema=PaperSummary,
                    temperature=0.1,
                    label=f"summary:{title}",
                )
                if not summary_data:
                    timings_ms["summary"] = round((time.perf_counter() - step_started_at) * 1000, 1)
                    return finish("skipped", "summary LLM failed")
            write_json(summary_path, summary_data)
            timings_ms["summary"] = round((time.perf_counter() - step_started_at) * 1000, 1)

        if not dimensions_path.exists():
            step_started_at = time.perf_counter()
            summary_dict = json.loads(summary_path.read_text(encoding="utf-8"))
            clean_review = review_text_path.read_text(encoding="utf-8")

            if abstract_only:
                abstract_text = "\n".join(summary_dict.get("abstract_content", []))
                user_content = (
                    f"### Paper title:\n{title}\n\n"
                    f"### Abstract summary:\n{abstract_text}\n\n"
                    f"### Review norms:\n{clean_review}"
                )
                result = self.call_llm_with_retry(
                    system_prompt=ABSTRACT_DIMENSIONS_SYSTEM_PROMPT,
                    user_content=user_content,
                    response_model_schema=PaperModuleDimensions,
                    temperature=0.1,
                    label=f"dimensions:{title}",
                )
                if not isinstance(result, dict):
                    timings_ms["dimensions"] = round((time.perf_counter() - step_started_at) * 1000, 1)
                    return finish("skipped", "dimension extraction LLM failed")
                final_criteria = {
                    "introduction_standards": result.get("introduction_standards") or [],
                    "methods_standards": result.get("methods_standards") or [],
                    "results_standards": result.get("results_standards") or [],
                    "discussion_standards": result.get("discussion_standards") or [],
                }
                timings_ms["dimensions"] = round((time.perf_counter() - step_started_at) * 1000, 1)
            else:
                def extract_module(name: str, json_key: str, text: str) -> tuple[str, list[dict[str, Any]], float]:
                    module_started_at = time.perf_counter()
                    if not text.strip():
                        return json_key, [], 0.0
                    result = self.call_llm_with_retry(
                        system_prompt=f"Extract reviewer standards specifically for the {name} section.",
                        user_content=f"### Summary:\n{text}\n### Review:\n{clean_review}",
                        response_model_schema=ModuleStandards,
                        temperature=0.1,
                        label=f"module:{title}:{name}",
                    )
                    standards = result.get("standards") if isinstance(result, dict) else None
                    elapsed_ms = round((time.perf_counter() - module_started_at) * 1000, 1)
                    self._log_timing(
                        "module_done",
                        title=title,
                        module=name,
                        status="ok" if isinstance(standards, list) else "skipped",
                        elapsed_ms=elapsed_ms,
                        standard_count=len(standards) if isinstance(standards, list) else 0,
                    )
                    return json_key, standards if isinstance(standards, list) else [], elapsed_ms

                intro_text = "\n".join(
                    summary_dict.get("abstract_content", []) + summary_dict.get("introduction_content", [])
                )
                module_inputs = [
                    ("Introduction", "introduction_standards", intro_text),
                    ("Methods", "methods_standards", "\n".join(summary_dict.get("methods_content", []))),
                    ("Results", "results_standards", "\n".join(summary_dict.get("results_content", []))),
                    ("Discussion", "discussion_standards", "\n".join(summary_dict.get("discussion_content", []))),
                ]
                final_criteria = {key: [] for _, key, _ in module_inputs}
                module_timings_ms: dict[str, float] = {}
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(module_inputs)) as executor:
                    future_map = {
                        executor.submit(extract_module, name, json_key, text): name
                        for name, json_key, text in module_inputs
                    }
                    for future in concurrent.futures.as_completed(future_map):
                        json_key, standards, elapsed_ms = future.result()
                        final_criteria[json_key] = standards
                        module_timings_ms[json_key] = elapsed_ms
                timings_ms["dimensions"] = round((time.perf_counter() - step_started_at) * 1000, 1)
                timings_ms["dimension_modules"] = module_timings_ms
            if not any(final_criteria.values()):
                return finish("skipped", "dimension extraction returned no standards")
            write_json(dimensions_path, final_criteria)

        return finish("ok")

    def extract_criteria_context(self, retrieved_docs: list[dict[str, Any]]) -> str:
        extracted_data: dict[str, set[str]] = {
            "Motivation": set(),
            "Method": set(),
            "Result": set(),
            "Discussion": set(),
        }
        key_mapping = {
            "introduction_standards": "Motivation",
            "methods_standards": "Method",
            "results_standards": "Result",
            "discussion_standards": "Discussion",
        }

        for doc in retrieved_docs:
            artifact_dir = Path(doc["artifact_dir"])
            dim_file = artifact_dir / "dimensions.json"
            if not dim_file.exists():
                continue
            try:
                data = json.loads(dim_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            for json_key, module_name in key_mapping.items():
                for standard in data.get(json_key, []):
                    if standard.get("dimension_name") and standard.get("core_reason"):
                        extracted_data[module_name].add(
                            f"- Dimension: {standard.get('dimension_name')}\n"
                            f"  Reason: {standard.get('core_reason')}"
                        )

        context_blocks: list[str] = []
        for module_name, items in extracted_data.items():
            if items:
                context_blocks.append(
                    f"=== {module_name} Historical Standards ===\n" + "\n\n".join(sorted(items))
                )
            else:
                context_blocks.append(f"=== {module_name} Historical Standards ===\nNo data.")
        return "\n\n".join(context_blocks)

    def build_sources(self) -> dict[str, Any]:
        ensure_directory(self.sources_root)
        engine = KgRubricSearch(self.config, call_llm=self.call_llm_with_retry)
        self._log_timing(
            "search_engine",
            backend="hosted_kg",
            top_k=self.config.search_top_k,
            final_k=self.config.search_final_k,
            rerank_batch_size=self.config.rerank_batch_size,
        )
        retrieved_docs = engine.search(
            self.config.target_idea,
            top_k=self.config.search_top_k,
            final_k=self.config.search_final_k,
        )
        if not retrieved_docs:
            raise RuntimeError("Idea rubric search returned no candidate papers")

        for index, doc in enumerate(retrieved_docs, start=1):
            doc["artifact_dir"] = str(self._build_doc_artifact_dir(index, doc).resolve())
            doc["folder_name"] = Path(doc["artifact_dir"]).name

        retrieved_papers_path = self.sources_root / "retrieved_papers.json"
        write_json(retrieved_papers_path, {"papers": retrieved_docs})

        processing_results: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_map = {executor.submit(self.process_single_retrieved_paper_abstract, doc): doc for doc in retrieved_docs}
            for future in concurrent.futures.as_completed(future_map):
                processing_results.append(future.result())

        processing_index_path = self.sources_root / "processing_index.json"
        write_json(processing_index_path, {"papers": processing_results})

        paper_contexts_path = self.sources_root / "paper_contexts.json"
        write_json(
            paper_contexts_path,
            {
                "papers": [
                    {
                        "title": normalize_whitespace(doc.get("Title")),
                        "doi": normalize_whitespace(doc.get("DOI")),
                        "folder_name": normalize_whitespace(doc.get("folder_name")),
                        "artifact_dir": doc["artifact_dir"],
                        "sections_path": str((Path(doc["artifact_dir"]) / "extracted_sections.json").resolve()),
                        "review_text_path": str((Path(doc["artifact_dir"]) / "review_text.txt").resolve()),
                        "source_meta_path": str((Path(doc["artifact_dir"]) / "source_meta.json").resolve()),
                    }
                    for doc in retrieved_docs
                ]
            },
        )
        return {
            "status": "ok",
            "retrieved_paper_count": len(retrieved_docs),
            "retrieved_papers_path": str(retrieved_papers_path.resolve()),
            "processing_index_path": str(processing_index_path.resolve()),
            "paper_contexts_path": str(paper_contexts_path.resolve()),
            "timing_log_path": str(self.timing_log_path.resolve()),
            "sources_root": str(self.sources_root.resolve()),
            "result_path": str(paper_contexts_path.resolve()),
        }

    def run_llm(self) -> dict[str, Any]:
        ensure_directory(self.config.artifact_root)
        ensure_directory(self.config.output_path.parent)
        paper_contexts_path = self.sources_root / "paper_contexts.json"
        if not paper_contexts_path.exists():
            raise RuntimeError("Rubric sources are missing paper_contexts.json")

        paper_contexts_payload = json.loads(paper_contexts_path.read_text(encoding="utf-8"))
        retrieved_docs = paper_contexts_payload.get("papers") if isinstance(paper_contexts_payload, dict) else None
        if not isinstance(retrieved_docs, list) or not retrieved_docs:
            raise RuntimeError("Rubric sources returned no paper contexts")

        llm_docs: list[dict[str, Any]] = []
        for item in retrieved_docs:
            if not isinstance(item, dict):
                continue
            source_artifact_dir = Path(normalize_whitespace(item.get("artifact_dir")))
            title = normalize_whitespace(item.get("title")) or source_artifact_dir.name
            llm_docs.append(
                {
                    "Title": title,
                    "DOI": normalize_whitespace(item.get("doi")),
                    "folder_name": normalize_whitespace(item.get("folder_name")),
                    "artifact_dir": str(
                        ensure_directory(self.config.artifact_root / "papers" / source_artifact_dir.name).resolve()
                    ),
                    "sections_path": normalize_whitespace(item.get("sections_path")),
                    "review_text_path": normalize_whitespace(item.get("review_text_path")),
                }
            )

        processing_results: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_map = {executor.submit(self.process_single_retrieved_paper_llm, doc): doc for doc in llm_docs}
            for future in concurrent.futures.as_completed(future_map):
                processing_results.append(future.result())

        processing_index_path = self.config.artifact_root / "processing_index.json"
        write_json(processing_index_path, {"papers": processing_results})

        # dimensions.json lives under the llm-stage artifact dirs (llm_docs), not the
        # sources-stage dirs carried in paper_contexts.json; reading the latter silently
        # produced an empty "No data." historical context in split-worker runs.
        historical_context = self.extract_criteria_context(llm_docs)
        historical_context_path = self.config.artifact_root / "historical_context.txt"
        _write_text(historical_context_path, historical_context)
        general_user_content = (
            f"=== IDEA ===\n{self.config.target_idea}\n\n"
            f"=== HISTORICAL STANDARDS ===\n{historical_context}"
        )
        detailed_user_content = (
            f"=== USER IDEA ===\n{self.config.target_idea}\n\n"
            f"=== HISTORICAL STANDARDS ===\n{historical_context}"
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_general = executor.submit(
                self.call_llm_with_retry,
                system_prompt=GENERAL_RUBRIC_SYSTEM_PROMPT,
                user_content=general_user_content,
                response_model_schema=GeneralReviewRubric,
                label="general_rubric",
            )
            future_detailed = executor.submit(
                self.call_llm_with_retry,
                system_prompt=DETAILED_RUBRIC_SYSTEM_PROMPT,
                user_content=detailed_user_content,
                response_model_schema=DetailedUnifiedRubric,
                label="detailed_rubric",
            )
            general_result = future_general.result()
            detailed_result = future_detailed.result()

        if not general_result:
            raise RuntimeError("Failed to generate general rubric")
        if not detailed_result:
            raise RuntimeError("Failed to generate detailed rubric")

        general_rubric_path = self.config.artifact_root / "general_rubric.json"
        detailed_rubric_path = self.config.artifact_root / "detailed_rubric.json"
        write_json(general_rubric_path, general_result)
        write_json(detailed_rubric_path, detailed_result)

        synthesis_user_content = "\n".join(
            [
                "=== USER RESEARCH IDEA ===",
                self.config.target_idea,
                "",
                "=== PRELIMINARY RUBRIC 1 (General) ===",
                _summarize_general_rubric_for_synthesis(general_result),
                "",
                "=== PRELIMINARY RUBRIC 2 (Detailed) ===",
                _summarize_detailed_rubric_for_synthesis(detailed_result),
            ]
        )
        synthesis_result = self.call_llm_with_retry(
            system_prompt=SECTIONED_SYNTHESIS_SYSTEM_PROMPT,
            user_content=synthesis_user_content,
            response_model_schema=SectionedSynthesisRubric,
            label="sectioned_rubric_synthesis",
        )
        if not synthesis_result:
            raise RuntimeError("Failed to synthesize sectioned rubric")

        synthesis_rubric_path = self.config.artifact_root / "sectioned_rubric_synthesis.json"
        write_json(synthesis_rubric_path, synthesis_result)

        rubric = _build_canonical_rubric(
            target_idea=self.config.target_idea,
            general_result=general_result,
            synthesis_result=synthesis_result,
        )
        if not any(rubric.get(key) for key in GENERAL_SECTION_KEYS):
            raise RuntimeError("Sectioned rubric synthesis returned no standards")

        write_json(self.config.output_path, rubric)
        return {
            "status": "ok",
            "rubric_schema_version": RUBRIC_SCHEMA_VERSION,
            "processing_index_path": str(processing_index_path.resolve()),
            "historical_context_path": str(historical_context_path.resolve()),
            "general_rubric_path": str(general_rubric_path.resolve()),
            "detailed_rubric_path": str(detailed_rubric_path.resolve()),
            "synthesis_rubric_path": str(synthesis_rubric_path.resolve()),
            "timing_log_path": str(self.timing_log_path.resolve()),
            "result_path": str(self.config.output_path.resolve()),
        }

    def run(self) -> dict[str, Any]:
        self.build_sources()
        return self.run_llm()


def run_idea_rubric(config: IdeaRubricConfig) -> dict[str, Any]:
    runner = IdeaRubricRunner(config)
    return runner.run()


def run_rubric_sources(config: IdeaRubricConfig) -> dict[str, Any]:
    runner = IdeaRubricRunner(config)
    return runner.build_sources()


def run_rubric_llm(config: IdeaRubricConfig) -> dict[str, Any]:
    runner = IdeaRubricRunner(config)
    return runner.run_llm()
