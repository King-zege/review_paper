from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from llm_client import HelloAgentsLLM
from paper_loader import PaperLoader, DocumentIndex, DocumentChunk


# =========================
# Schema
# =========================


@dataclass
class IssueLocation:
    chunk_id: str
    chapter: str
    section: str
    page_start: Optional[int]
    page_end: Optional[int]
    paragraph_start: int
    paragraph_end: int


@dataclass
class ReviewIssue:
    issue_id: str
    issue_type: str
    severity: str
    title: str
    location: IssueLocation
    evidence_quote: str
    problem: str
    impact: str
    suggestion: str
    source_agent: str
    verified: Optional[bool] = None
    verifier_comment: Optional[str] = None


@dataclass
class LocalLogicSummary:
    chunk_id: str
    chapter: str
    section: str
    main_points: List[str]
    claims: List[str]
    evidence_summary: List[str]
    local_logic_risks: List[str]
    dependencies: List[str]
    importance: str


@dataclass
class ChapterSummary:
    chapter: str
    sections: List[str]
    main_objective: str
    core_methods_or_arguments: List[str]
    key_strengths: List[str]
    chapter_risks: List[str]
    cross_chapter_dependencies: List[str]
    representative_chunk_ids: List[str]


# =========================
# Prompts
# =========================


FORMAT_REVIEW_PROMPT = """
你是一名严格的硕士论文格式与写作问题审查专家。你只审查当前文本块的局部问题，不要对全篇下结论。
你的任务是找出当前文本块中可以通过局部证据确认的问题，并输出 JSON 数组。

【定位元数据】
chunk_id={chunk_id}
chapter={chapter_title}
section={section_title}
pages={page_start}-{page_end}
paragraphs={paragraph_start}-{paragraph_end}

【当前文本块】
{text}

请只输出 JSON 数组，每个元素必须严格包含以下字段：
- issue_id: 字符串，格式如 "fmt_c0001_01"
- issue_type: 只能取 "format" "writing" "citation" "terminology" "figure_table" "ai_tone"
- severity: 只能取 "low" "medium" "high"
- title: 简短问题标题
- location: 对象，必须包含 chunk_id chapter section page_start page_end paragraph_start paragraph_end
- evidence_quote: 原文中的短句，必须来自当前文本块，尽量短且可核查
- problem: 具体问题描述
- impact: 问题危害
- suggestion: 可执行修改建议
- source_agent: 固定写 "local_format_reviewer"

要求：
1. 只保留有明确证据的问题，不要泛泛而谈。
2. 没有问题时输出 []。
3. 一般返回 0-6 条，不要过多。
4. 不要输出 markdown，不要加解释。
"""


LOGIC_SUMMARY_PROMPT = """
你是一名负责局部逻辑梳理的论文审查专家。你不负责给整篇论文下总评，只总结当前文本块在整篇论文逻辑中的作用与风险。

【定位元数据】
chunk_id={chunk_id}
chapter={chapter_title}
section={section_title}
pages={page_start}-{page_end}
paragraphs={paragraph_start}-{paragraph_end}

【当前文本块】
{text}

请只输出一个 JSON 对象，字段严格如下：
- chunk_id
- chapter
- section
- main_points: 列表，概括本块在说什么
- claims: 列表，作者在本块显式或隐式提出了哪些论断
- evidence_summary: 列表，本块提供了哪些证据/描述来支撑这些论断
- local_logic_risks: 列表，本块内部或与上下游衔接方面有哪些逻辑风险；没有则写 []
- dependencies: 列表，本块依赖哪些前文/后文内容才完整成立；没有则写 []
- importance: 只能取 "low" "medium" "high"

要求：
1. 不要写整篇论文总结。
2. 不要使用空泛措辞。
3. 只输出 JSON 对象。
"""


CHAPTER_SYNTHESIS_PROMPT = """
你是一名章节主审。你将看到同一章的多个局部逻辑摘要，以及该章中已发现的局部问题清单。
请整合它们，生成本章的结构化摘要，供全局审查使用。

【章名】
{chapter}

【该章局部逻辑摘要】
{logic_summaries}

【该章局部问题清单】
{issues}

请只输出一个 JSON 对象，字段严格如下：
- chapter
- sections: 本章涉及的节标题列表
- main_objective: 本章的主要目的
- core_methods_or_arguments: 本章核心方法或核心论述点列表
- key_strengths: 本章相对扎实之处列表，没有则 []
- chapter_risks: 本章存在的主要问题列表，优先写会影响全局结论的问题
- cross_chapter_dependencies: 本章与其他章节的依赖或承接关系列表
- representative_chunk_ids: 最能代表本章内容的 chunk_id 列表

只输出 JSON 对象。
"""


GLOBAL_REVIEW_PROMPT = """
你是一名极其严格的高校硕士论文盲审专家。你现在需要对整篇论文给出全局评审。

重要约束：
1. 你现在只被允许直接阅读两段原文：摘要与结论。
2. 除摘要与结论外，你对全文的判断只能依据“各章结构化摘要”和“已提取的局部问题清单”。
3. 你必须检查：摘要、结论、各章内容是否彼此一致；是否存在结论拔高、研究问题未闭环、创新点支撑不足等全局问题。
4. 所有重大问题都必须带具体位置；位置优先引用已给出的局部问题 location，或引用章节代表性 chunk 的位置。

【论文标题】
{title}

【摘要原文】
{abstract_text}

【结论原文】
{conclusion_text}

【各章结构化摘要】
{chapter_summaries}

【高优先级局部问题】
{priority_issues}

【代表性位置映射】
{chapter_anchor_map}

请只输出一个 JSON 对象，字段严格如下：
- overall_assessment: 字符串
- strengths: 列表，1-4条
- fatal_weaknesses: 列表，1-4条
- dimension_reviews: 对象，键必须是 "整体逻辑与结构" "选题意义与文献综述" "方法与严谨性" "实验与结果分析" "规范与写作"；每个值都是字符串
- top_issues: 数组，每个元素字段为 issue_id title severity location evidence_quote problem impact suggestion source_agent
- scores: 对象，键必须是 "选题与意义" "文献与理论" "方法与设计" "实验与严谨性" "规范与写作" "总分"
- blind_review_conclusion: 只能取 "优秀" "良好" "中等" "较弱" "不建议直接通过"
- review_letter: 字符串，可直接用于评阅书

要求：
1. top_issues 控制在 6-10 条。
2. 所有重大问题必须带 location 和 evidence_quote。
3. 必须显式检查摘要与结论是否和各章摘要一致。
4. 不要虚构 location；如果是全局性问题，优先借用最相关的代表性 chunk 位置。
5. 只输出 JSON。
"""


ISSUE_VERIFICATION_PROMPT = """
你是一名证据核查员。你的任务不是重新审稿，而是核查某条审稿意见是否确实有原文依据。

【待核查问题】
{issue}

【该位置的原文片段及邻近上下文】
{context}

请只输出一个 JSON 对象，字段严格如下：
- issue_id
- verdict: 只能取 "confirmed" "needs_revision" "rejected"
- is_supported: true 或 false
- corrected_evidence_quote: 如果原引用不准，给出更合适的短引文；否则保持原文
- verifier_comment: 说明为何确认、为何需要修订、或为何否决
- revised_problem: 若需要修订，给出更准确的问题表述；否则返回原问题
- revised_suggestion: 若需要修订，给出更准确建议；否则返回原建议

要求：
1. 只根据给定上下文核查，不要脑补。
2. 只输出 JSON。
"""


# =========================
# Utilities
# =========================


def extract_json_payload(text: str) -> Any:
    text = (text or "").strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        obj_match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if obj_match:
            return json.loads(obj_match.group(1))
        raise


def compact_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def issue_to_dict(issue: ReviewIssue) -> Dict[str, Any]:
    return asdict(issue)


def normalize_issue(raw: Dict[str, Any]) -> ReviewIssue:
    loc_raw = raw["location"]
    return ReviewIssue(
        issue_id=raw["issue_id"],
        issue_type=raw.get("issue_type", "writing"),
        severity=raw.get("severity", "medium"),
        title=raw["title"],
        location=IssueLocation(
            chunk_id=loc_raw["chunk_id"],
            chapter=loc_raw["chapter"],
            section=loc_raw["section"],
            page_start=loc_raw.get("page_start"),
            page_end=loc_raw.get("page_end"),
            paragraph_start=loc_raw["paragraph_start"],
            paragraph_end=loc_raw["paragraph_end"],
        ),
        evidence_quote=raw.get("evidence_quote", ""),
        problem=raw.get("problem", ""),
        impact=raw.get("impact", ""),
        suggestion=raw.get("suggestion", ""),
        source_agent=raw.get("source_agent", "unknown"),
        verified=raw.get("verified"),
        verifier_comment=raw.get("verifier_comment"),
    )


class JsonLLM:
    def __init__(self, llm: HelloAgentsLLM):
        self.llm = llm

    def call_json(self, prompt: str) -> Any:
        raw = self.llm.think(messages=[{"role": "user", "content": prompt}], temperature=0) or ""
        return extract_json_payload(raw)


# =========================
# Agents
# =========================


class LocalFormatReviewer:
    def __init__(self, llm: JsonLLM):
        self.llm = llm

    def review(self, chunk: DocumentChunk) -> List[ReviewIssue]:
        data = self.llm.call_json(FORMAT_REVIEW_PROMPT.format(**chunk.to_dict()))
        if not isinstance(data, list):
            raise ValueError("LocalFormatReviewer 期望返回 JSON 数组")
        return [normalize_issue(item) for item in data]


class LocalLogicSummarizer:
    def __init__(self, llm: JsonLLM):
        self.llm = llm

    def summarize(self, chunk: DocumentChunk) -> LocalLogicSummary:
        data = self.llm.call_json(LOGIC_SUMMARY_PROMPT.format(**chunk.to_dict()))
        return LocalLogicSummary(**data)


class ChapterSynthesizer:
    def __init__(self, llm: JsonLLM):
        self.llm = llm

    def synthesize(self, chapter: str, logic_summaries: List[LocalLogicSummary], issues: List[ReviewIssue]) -> ChapterSummary:
        data = self.llm.call_json(
            CHAPTER_SYNTHESIS_PROMPT.format(
                chapter=chapter,
                logic_summaries=compact_json([asdict(x) for x in logic_summaries]),
                issues=compact_json([issue_to_dict(x) for x in issues]),
            )
        )
        return ChapterSummary(**data)


class GlobalReviewAgent:
    def __init__(self, llm: JsonLLM):
        self.llm = llm

    def review(self, document_index: DocumentIndex, chapter_summaries: List[ChapterSummary], issues: List[ReviewIssue]) -> Dict[str, Any]:
        abstract_text = "\n\n".join(c.text for c in document_index.get_chunks_by_flag(abstract=True))
        conclusion_text = "\n\n".join(c.text for c in document_index.get_chunks_by_flag(conclusion=True))
        if not abstract_text:
            abstract_text = "[未显式识别到摘要，请人工核查文档结构]"
        if not conclusion_text:
            conclusion_text = "[未显式识别到结论，请人工核查文档结构]"

        priority_issues = select_priority_issues(issues, top_n=25)
        chapter_anchor_map = build_chapter_anchor_map(document_index, chapter_summaries)

        data = self.llm.call_json(
            GLOBAL_REVIEW_PROMPT.format(
                title=document_index.title,
                abstract_text=abstract_text,
                conclusion_text=conclusion_text,
                chapter_summaries=compact_json([asdict(x) for x in chapter_summaries]),
                priority_issues=compact_json([issue_to_dict(x) for x in priority_issues]),
                chapter_anchor_map=compact_json(chapter_anchor_map),
            )
        )
        return data


class EvidenceVerifier:
    def __init__(self, llm: JsonLLM):
        self.llm = llm

    def verify_issue(self, document_index: DocumentIndex, issue: ReviewIssue) -> ReviewIssue:
        context = document_index.context_for_location(issue.location.chunk_id, radius=1)
        payload = {
            "issue_id": issue.issue_id,
            "title": issue.title,
            "severity": issue.severity,
            "location": asdict(issue.location),
            "evidence_quote": issue.evidence_quote,
            "problem": issue.problem,
            "suggestion": issue.suggestion,
            "source_agent": issue.source_agent,
        }
        data = self.llm.call_json(
            ISSUE_VERIFICATION_PROMPT.format(issue=compact_json(payload), context=context)
        )
        verdict = data["verdict"]
        issue.verified = data.get("is_supported", False)
        issue.verifier_comment = data.get("verifier_comment")
        issue.evidence_quote = data.get("corrected_evidence_quote", issue.evidence_quote)
        if verdict == "needs_revision":
            issue.problem = data.get("revised_problem", issue.problem)
            issue.suggestion = data.get("revised_suggestion", issue.suggestion)
        return issue


# =========================
# Selection / Mapping
# =========================


def select_priority_issues(issues: List[ReviewIssue], top_n: int = 25) -> List[ReviewIssue]:
    severity_rank = {"high": 0, "medium": 1, "low": 2}
    sorted_issues = sorted(
        issues,
        key=lambda x: (
            severity_rank.get(x.severity, 9),
            x.location.chapter,
            x.location.paragraph_start,
            x.issue_id,
        ),
    )
    return sorted_issues[:top_n]



def build_chapter_anchor_map(document_index: DocumentIndex, chapter_summaries: List[ChapterSummary]) -> List[Dict[str, Any]]:
    result = []
    for summary in chapter_summaries:
        chunk_id = None
        if summary.representative_chunk_ids:
            chunk_id = summary.representative_chunk_ids[0]
        else:
            for c in document_index.chunks:
                if c.chapter_title == summary.chapter:
                    chunk_id = c.chunk_id
                    break
        if chunk_id is None:
            continue
        chunk = document_index.get_chunk(chunk_id)
        result.append(
            {
                "chapter": summary.chapter,
                "chunk_id": chunk.chunk_id,
                "section": chunk.section_title,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "paragraph_start": chunk.paragraph_start,
                "paragraph_end": chunk.paragraph_end,
            }
        )
    return result


# =========================
# Pipeline
# =========================


class ThesisReviewPipeline:
    def __init__(self, llm_client: HelloAgentsLLM, chunk_size: int = 3500, overlap: int = 400):
        self.loader = PaperLoader(chunk_size=chunk_size, overlap=overlap)
        self.llm = JsonLLM(llm_client)
        self.local_format_reviewer = LocalFormatReviewer(self.llm)
        self.local_logic_summarizer = LocalLogicSummarizer(self.llm)
        self.chapter_synthesizer = ChapterSynthesizer(self.llm)
        self.global_reviewer = GlobalReviewAgent(self.llm)
        self.evidence_verifier = EvidenceVerifier(self.llm)

    def run(self, paper_path: str, verify_top_n: int = 20) -> Dict[str, Any]:
        document_index = self.loader.load(paper_path)
        print(f"已加载文档: {document_index.title}")
        print(f"文本块数量: {len(document_index.chunks)}")

        all_issues: List[ReviewIssue] = []
        all_logic_summaries: List[LocalLogicSummary] = []

        for i, chunk in enumerate(document_index.chunks, start=1):
            print(f"\n[{i}/{len(document_index.chunks)}] 审查 {chunk.chunk_id} | {chunk.chapter_title} | {chunk.section_title}")
            issues = self.local_format_reviewer.review(chunk)
            logic_summary = self.local_logic_summarizer.summarize(chunk)
            all_issues.extend(issues)
            all_logic_summaries.append(logic_summary)
            print(f"  - 局部问题: {len(issues)} 条")
            print("  - 局部逻辑摘要完成")

        chapter_to_summaries: Dict[str, List[LocalLogicSummary]] = defaultdict(list)
        chapter_to_issues: Dict[str, List[ReviewIssue]] = defaultdict(list)
        for summary in all_logic_summaries:
            chapter_to_summaries[summary.chapter].append(summary)
        for issue in all_issues:
            chapter_to_issues[issue.location.chapter].append(issue)

        chapter_summaries: List[ChapterSummary] = []
        for chapter, summaries in chapter_to_summaries.items():
            print(f"\n[章节综合] {chapter}")
            chapter_summary = self.chapter_synthesizer.synthesize(
                chapter=chapter,
                logic_summaries=summaries,
                issues=chapter_to_issues.get(chapter, []),
            )
            chapter_summaries.append(chapter_summary)

        print("\n[全局评审] 只直读摘要与结论，其他只用结构化摘要与局部问题")
        global_review = self.global_reviewer.review(document_index, chapter_summaries, all_issues)

        global_top_issue_objs: List[ReviewIssue] = []
        for idx, raw in enumerate(global_review.get("top_issues", []), start=1):
            raw.setdefault("source_agent", "global_reviewer")
            raw.setdefault("issue_id", f"global_{idx:02d}")
            global_top_issue_objs.append(normalize_issue(raw))

        candidates = all_issues + global_top_issue_objs
        severity_rank = {"high": 0, "medium": 1, "low": 2}
        candidates_sorted = sorted(
            candidates,
            key=lambda x: (severity_rank.get(x.severity, 9), x.issue_id),
        )[:verify_top_n]

        verified_issues: List[ReviewIssue] = []
        print(f"\n[证据核查] 计划核查 {len(candidates_sorted)} 条问题")
        for issue in candidates_sorted:
            checked = self.evidence_verifier.verify_issue(document_index, issue)
            if checked.verified is False and checked.source_agent != "global_reviewer":
                continue
            if checked.verified is False and checked.source_agent == "global_reviewer":
                continue
            verified_issues.append(checked)

        return self._assemble_final_report(
            document_index=document_index,
            chapter_summaries=chapter_summaries,
            all_issues=all_issues,
            verified_issues=verified_issues,
            global_review=global_review,
        )

    def _assemble_final_report(
        self,
        document_index: DocumentIndex,
        chapter_summaries: List[ChapterSummary],
        all_issues: List[ReviewIssue],
        verified_issues: List[ReviewIssue],
        global_review: Dict[str, Any],
    ) -> Dict[str, Any]:
        verified_map = {x.issue_id: x for x in verified_issues}
        final_top_issues = []
        for raw in global_review.get("top_issues", []):
            issue_id = raw.get("issue_id")
            if issue_id in verified_map:
                final_top_issues.append(issue_to_dict(verified_map[issue_id]))
            else:
                raw["verified"] = None
                raw["verifier_comment"] = "未进入本轮证据核查列表"
                final_top_issues.append(raw)

        verified_local_issues = [issue_to_dict(x) for x in verified_issues if x.source_agent != "global_reviewer"]

        return {
            "paper_title": document_index.title,
            "source_path": document_index.source_path,
            "settings": {
                "global_raw_text_inputs": ["abstract", "conclusion"],
                "other_global_inputs": ["chapter_summaries", "priority_issues", "chapter_anchor_map"],
            },
            "chunk_schema": {
                "required_fields": [
                    "chunk_id",
                    "chapter_title",
                    "section_title",
                    "page_start",
                    "page_end",
                    "paragraph_start",
                    "paragraph_end",
                    "text",
                    "is_abstract",
                    "is_conclusion",
                ]
            },
            "issue_schema": {
                "required_fields": [
                    "issue_id",
                    "issue_type",
                    "severity",
                    "title",
                    "location",
                    "evidence_quote",
                    "problem",
                    "impact",
                    "suggestion",
                    "source_agent",
                    "verified",
                    "verifier_comment",
                ]
            },
            "reflect_lookup_mechanism": {
                "strategy": "按 issue.location.chunk_id 回查原 chunk，并附带前后相邻 chunk 上下文，再核对 evidence_quote / problem / suggestion 是否被原文支持",
                "radius": 1,
                "verification_scope": "默认核查高优先级问题与全局 top issues",
            },
            "chapter_summaries": [asdict(x) for x in chapter_summaries],
            "verified_local_issues": verified_local_issues,
            "global_review": {
                **global_review,
                "top_issues": final_top_issues,
            },
            "stats": {
                "chunk_count": len(document_index.chunks),
                "local_issue_count": len(all_issues),
                "verified_issue_count": len(verified_issues),
            },
        }


# =========================
# Rendering / Saving
# =========================


def render_markdown_report(report: Dict[str, Any]) -> str:
    g = report["global_review"]
    lines = []
    lines.append(f"# {report['paper_title']} - 盲审评审报告")
    lines.append("")
    lines.append("## 总体评价")
    lines.append(g.get("overall_assessment", ""))
    lines.append("")
    lines.append("## 主要优点")
    for item in g.get("strengths", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 致命弱点")
    for item in g.get("fatal_weaknesses", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 逐项深度审查")
    for k, v in g.get("dimension_reviews", {}).items():
        lines.append(f"### {k}")
        lines.append(v)
        lines.append("")
    lines.append("## 核心问题清单")
    for idx, issue in enumerate(g.get("top_issues", []), start=1):
        loc = issue.get("location", {})
        lines.append(f"### {idx}. {issue.get('title', '')} [{issue.get('severity', '')}]")
        lines.append(
            f"- 位置：{loc.get('chapter', '')} / {loc.get('section', '')} / chunk {loc.get('chunk_id', '')} / 页 {loc.get('page_start', '')}-{loc.get('page_end', '')} / 段 {loc.get('paragraph_start', '')}-{loc.get('paragraph_end', '')}"
        )
        lines.append(f"- 证据：{issue.get('evidence_quote', '')}")
        lines.append(f"- 问题：{issue.get('problem', '')}")
        lines.append(f"- 影响：{issue.get('impact', '')}")
        lines.append(f"- 建议：{issue.get('suggestion', '')}")
        lines.append(f"- 核查：{issue.get('verifier_comment', '未核查')}")
        lines.append("")
    lines.append("## 量化评分")
    for k, v in g.get("scores", {}).items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append(f"**盲审结论：{g.get('blind_review_conclusion', '')}**")
    lines.append("")
    lines.append("## 评阅书意见")
    lines.append(g.get("review_letter", ""))
    lines.append("")
    return "\n".join(lines)



def save_outputs(base_path: str, report: Dict[str, Any]) -> None:
    path = Path(base_path)
    out_json = path.with_name(f"{path.stem}_review_report.json")
    out_md = path.with_name(f"{path.stem}_review_report.md")
    out_json.write_text(compact_json(report), encoding="utf-8")
    out_md.write_text(render_markdown_report(report), encoding="utf-8")
    print(f"\nJSON 报告已保存: {out_json}")
    print(f"Markdown 报告已保存: {out_md}")


# =========================
# CLI
# =========================


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="硕士论文多 agent 盲审系统（全局只直读摘要和结论）")
    parser.add_argument("--paper_path", type=str, required=True, help="论文路径，支持 .pdf / .docx / .txt")
    parser.add_argument("--chunk_size", type=int, default=3500, help="每个文本块的最大字符数")
    parser.add_argument("--overlap", type=int, default=400, help="相邻文本块的重叠字符数")
    parser.add_argument("--verify_top_n", type=int, default=20, help="证据核查的问题条数上限")
    args = parser.parse_args()

    llm_client = HelloAgentsLLM()
    pipeline = ThesisReviewPipeline(
        llm_client=llm_client,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    report = pipeline.run(args.paper_path, verify_top_n=args.verify_top_n)
    save_outputs(args.paper_path, report)


if __name__ == "__main__":
    main()
