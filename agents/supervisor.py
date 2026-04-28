import json
import os
from dataclasses import dataclass, field
from typing import Any, Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from RAG_multiagent.agents.prompts import CRITIC_SYSTEM, PLANNER_SYSTEM, WRITER_SYSTEM
from RAG_multiagent.config import Settings, get_settings
from RAG_multiagent.llm import get_chat_model
from RAG_multiagent.models import (
    Claim,
    Critique,
    Evidence,
    QueryPlan,
    ResearchReport,
    ResearchRequest,
)
from RAG_multiagent.retrieval.evidence import (
    format_evidence_for_prompt,
    grade_and_filter,
)
from RAG_multiagent.retrieval.local_retriever import retrieve_local
from RAG_multiagent.retrieval.web_search import tavily_search
from RAG_multiagent.utils.json import extract_json_object

load_dotenv()

SupervisorAction = Literal[
    "plan",
    "local_research",
    "web_research",
    "grade_evidence",
    "write_report",
    "critique_report",
    "finish",
]

DecisionSource = Literal["tool_call", "json_fallback", "rule_fallback"]


class SupervisorDecision(BaseModel):
    action: SupervisorAction
    rationale: str = Field(default="", max_length=1200)
    queries: list[str] = Field(default_factory=list, max_length=8)
    revision_instructions: str = Field(default="", max_length=2000)
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    source: DecisionSource = "tool_call"


@dataclass
class AgentResearchState:
    request: ResearchRequest
    plan: QueryPlan | None = None
    evidence: list[Evidence] = field(default_factory=list)
    graded_evidence: list[dict[str, Any]] = field(default_factory=list)
    report: ResearchReport | None = None
    critique: Critique | None = None
    errors: list[str] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)


class PlanResearchInput(BaseModel):
    """规划工具参数。"""

    rationale: str = Field(default="", description="为什么现在需要生成或重做研究计划。")
    confidence: float = Field(
        default=0.7, ge=0.0, le=1.0, description="主管对该决策的信心。"
    )


class ResearchSearchInput(BaseModel):
    """本地或网页检索工具参数。"""

    queries: list[str] = Field(
        default_factory=list, description="具体、可检索的查询词列表。"
    )
    rationale: str = Field(default="", description="为什么需要这轮检索。")
    confidence: float = Field(
        default=0.7, ge=0.0, le=1.0, description="主管对该决策的信心。"
    )


class CurateEvidenceInput(BaseModel):
    """证据策展工具参数。"""

    rationale: str = Field(
        default="", description="为什么现在需要去重、评分和筛选证据。"
    )
    confidence: float = Field(
        default=0.7, ge=0.0, le=1.0, description="主管对该决策的信心。"
    )


class WriteReportInput(BaseModel):
    """写作或重写工具参数。"""

    revision_instructions: str = Field(
        default="", description="首次写作或重写时的具体要求。"
    )
    rationale: str = Field(default="", description="为什么现在需要写作或重写报告。")
    confidence: float = Field(
        default=0.7, ge=0.0, le=1.0, description="主管对该决策的信心。"
    )


class CritiqueReportInput(BaseModel):
    """质检工具参数。"""

    rationale: str = Field(default="", description="为什么现在需要检查报告质量。")
    confidence: float = Field(
        default=0.7, ge=0.0, le=1.0, description="主管对该决策的信心。"
    )


class FinishResearchInput(BaseModel):
    """结束工具参数。"""

    rationale: str = Field(default="", description="为什么当前报告已经可以交付。")
    confidence: float = Field(
        default=0.7, ge=0.0, le=1.0, description="主管对结束决策的信心。"
    )


TOOL_TO_ACTION: dict[str, SupervisorAction] = {
    "plan_research": "plan",
    "search_local_knowledge": "local_research",
    "search_web": "web_research",
    "curate_evidence": "grade_evidence",
    "write_or_revise_report": "write_report",
    "critique_report": "critique_report",
    "finish_research": "finish",
}
TOOL_INPUT_SCHEMA: dict[str, type[BaseModel]] = {
    "plan_research": PlanResearchInput,
    "search_local_knowledge": ResearchSearchInput,
    "search_web": ResearchSearchInput,
    "curate_evidence": CurateEvidenceInput,
    "write_or_revise_report": WriteReportInput,
    "critique_report": CritiqueReportInput,
    "finish_research": FinishResearchInput,
}

SUPERVISOR_SYSTEM = """你是一个研究型多智能体系统的主管智能体。
你必须通过工具调用来调度专家能力；每一轮只调用一个最合适的工具。

可用工具：
- plan_research：让规划专家生成或重做研究计划。
- search_local_knowledge：让本地 RAG 专家检索私有知识库。
- search_web：让网页研究专家检索公开互联网资料；只有用户允许网页搜索时使用。
- curate_evidence：让证据策展专家去重、评分、筛选证据。
- write_or_revise_report：让写作专家基于证据撰写或重写报告。
- critique_report：让质检专家检查报告证据覆盖、缺口和 unsupported claims。
- finish_research：当报告已经足够好时结束。

调度原则：
1. 不要机械执行固定流水线；你要根据当前状态选择下一步工具。
2. 没有计划时通常调用 plan_research。
3. 证据不足时调用 search_local_knowledge；如果用户允许且主题需要时效性，再调用 search_web。
4. 有新证据后通常调用 curate_evidence，再考虑写作。
5. 有报告后应调用 critique_report；质检发现缺口时继续检索或重写。
6. 只有当报告存在、证据充分、质检通过或缺口可接受时才调用 finish_research。
7. 工具参数要具体，特别是检索 queries 和重写 revision_instructions。
"""


class AgenticResearchOrchestrator:
    def __init__(self, settings: Settings | None = None):
        self.supervisor_tools = self._build_supervisor_tools()
        self.settings = settings or get_settings()

    def run(self, request: ResearchRequest) -> ResearchReport:
        state = AgentResearchState(request=request)

        for step in range(1, 10):
            decision = self._choose_next_tool(state, step)
            decision = self._apply_safety_guardrails(state, decision)
            self._record_decision(state, step, decision)

            if decision.action == "finish":
                break
            try:
                self._execute_decision(state, decision, step)

            except Exception as e:
                state.errors.append(f"{decision.action}: {e}")
                self._record_observation(state, step, decision.action, f"执行失败：{e}")
        if state.report is None:
            self._force_minimum_report(state, 10)

        state.report.agent_trace = state.trace
        return state.report

    def _build_supervisor_tools(self) -> list[StructuredTool]:
        """构建主管模型可调用的专家工具 schema。"""

        def plan_research(**_: Any) -> str:
            """生成或重做研究计划。"""
            return "规划专家会生成结构化 QueryPlan。"

        def search_local_knowledge(**_: Any) -> str:
            """检索本地知识库。"""
            return "本地 RAG 专家会返回 Evidence。"

        def search_web(**_: Any) -> str:
            """检索公开互联网资料。"""
            return "网页研究专家会返回 Web Evidence。"

        def curate_evidence(**_: Any) -> str:
            """对证据去重、评分、筛选。"""
            return "证据策展专家会更新 evidence 和 graded_evidence。"

        def write_or_revise_report(**_: Any) -> str:
            """基于证据写作或重写报告。"""
            return "写作专家会生成 ResearchReport。"

        def critique_report(**_: Any) -> str:
            """检查报告质量并提出下一步建议。"""
            return "质检专家会生成 Critique。"

        def finish_research(**_: Any) -> str:
            """结束研究并返回当前报告。"""
            return "主管确认研究任务可以结束。"

        return [
            StructuredTool.from_function(
                name="plan_research",
                description="当缺少计划、计划过粗或需要重做研究拆解时调用。",
                func=plan_research,
                args_schema=PlanResearchInput,
            ),
            StructuredTool.from_function(
                name="search_local_knowledge",
                description="检索本地知识库，适合查私有文档、样例资料和已入库知识。",
                func=search_local_knowledge,
                args_schema=ResearchSearchInput,
            ),
            StructuredTool.from_function(
                name="search_web",
                description="检索公开互联网资料，适合查最新事实、行业报告和公开资料。",
                func=search_web,
                args_schema=ResearchSearchInput,
            ),
            StructuredTool.from_function(
                name="curate_evidence",
                description="在已有证据后去重、评分、筛选，准备交给写作专家。",
                func=curate_evidence,
                args_schema=CurateEvidenceInput,
            ),
            StructuredTool.from_function(
                name="write_or_revise_report",
                description="基于当前证据首次写作，或根据质检反馈重写报告。",
                func=write_or_revise_report,
                args_schema=WriteReportInput,
            ),
            StructuredTool.from_function(
                name="critique_report",
                description="检查报告是否证据充分、引用是否覆盖关键结论、是否需要继续。",
                func=critique_report,
                args_schema=CritiqueReportInput,
            ),
            StructuredTool.from_function(
                name="finish_research",
                description="只有报告已经存在且质量可交付时调用。",
                func=finish_research,
                args_schema=FinishResearchInput,
            ),
        ]

    def _choose_next_tool(
        self, state: AgentResearchState, step: int
    ) -> SupervisorDecision:
        """优先让主管模型通过工具调用选择下一步；不支持工具时回退到 JSON 决策。"""
        llm = get_chat_model("dashscope")
        prompt = f"""请观察当前状态，并调用一个最合适的工具推进研究任务。
用户请求：
{state.request.model_dump_json(indent=2)}

当前步数：{step}/{10}

状态摘要：
{self._state_summary(state)}

如果当前模型无法使用工具调用，请仅返回 JSON：
{{
  "action": "plan | local_research | web_research | grade_evidence | write_report | critique_report | finish",
  "rationale": "为什么选择这一步",
  "queries": ["如需检索，给出具体查询词"],
  "revision_instructions": "如需重写报告，给出修改要求",
  "confidence": 0.0
}}
"""
        try:
            model_with_tools = self._bind_tools(llm)
            response = model_with_tools.invoke(
                [SystemMessage(content=SUPERVISOR_SYSTEM), HumanMessage(content=prompt)]
            )
            tool_calls = getattr(response, "tool_calls", None) or []
            if tool_calls:
                return self._decision_from_tool_call(tool_calls[0])
            data = extract_json_object(str(response.content))
            if data:
                return SupervisorDecision.model_validate(data).model_copy(
                    update={"source": "json_fallback"}
                )
        except Exception as exc:  # noqa: BLE001 - 工具调用不被模型支持时使用规则兜底
            state.errors.append(f"supervisor_tool_fallback: {exc}")
        return self._fallback_decision(state)

    def _bind_tools(self, llm: Any) -> Any:
        """兼容不同模型封装的工具绑定签名。"""
        try:
            return llm.bind_tools(self.supervisor_tools, tool_choice="auto")
        except TypeError:
            return llm.bind_tools(self.supervisor_tools)

    def _decision_from_tool_call(self, tool_call: dict[str, Any]) -> SupervisorDecision:
        """把模型产生的工具调用转换成内部调度决策。"""
        tool_name = str(tool_call.get("name") or "")
        if tool_name not in TOOL_TO_ACTION:
            raise ValueError(f"未知主管工具：{tool_name}")

        args = tool_call.get("args") or {}
        parsed = TOOL_INPUT_SCHEMA[tool_name].model_validate(args)
        return SupervisorDecision(
            action=TOOL_TO_ACTION[tool_name],
            rationale=str(getattr(parsed, "rationale", "")),
            queries=list(getattr(parsed, "queries", []) or []),
            revision_instructions=str(getattr(parsed, "revision_instructions", "")),
            confidence=float(getattr(parsed, "confidence", 0.6)),
            source="tool_call",
        )

    def _apply_safety_guardrails(
        self,
        state: AgentResearchState,
        decision: SupervisorDecision,
    ) -> SupervisorDecision:
        """应用最低安全护栏，防止主管在前置条件不足时调用错误工具。"""
        if decision.action == "finish" and state.report is None:
            return decision.model_copy(
                update={
                    "action": "write_report" if state.plan else "plan",
                    "rationale": "主管请求结束，但当前还没有报告；安全护栏要求先产出报告。",
                }
            )
        if decision.action == "web_research" and not state.request.require_web:
            return decision.model_copy(
                update={
                    "action": "local_research",
                    "queries": decision.queries
                    or self._default_queries_for_research(state, "local_research"),
                    "rationale": "用户关闭网页搜索，改为检索本地知识库。",
                }
            )
        if (
            decision.action in {"local_research", "web_research"}
            and not decision.queries
        ):
            return decision.model_copy(
                update={
                    "queries": self._default_queries_for_research(
                        state, decision.action
                    )
                }
            )
        if decision.action in {"grade_evidence", "write_report"} and not state.evidence:
            return decision.model_copy(
                update={
                    "action": "local_research",
                    "queries": self._default_queries_for_research(
                        state, "local_research"
                    ),
                    "rationale": "当前没有证据，先检索本地知识库。",
                }
            )
        if decision.action == "critique_report" and state.report is None:
            return decision.model_copy(
                update={
                    "action": "write_report",
                    "rationale": "当前没有报告，先让写作专家生成报告。",
                }
            )
        return decision

    def _execute_decision(
        self,
        state: AgentResearchState,
        decision: SupervisorDecision,
        step: int,
    ) -> None:
        """根据主管工具调用结果执行对应专家能力。"""
        if decision.action == "plan":
            self._run_planner_agent(state, step)
        elif decision.action == "local_research":
            self._run_local_research_agent(state, decision.queries, step)
        elif decision.action == "web_research":
            self._run_web_research_agent(state, decision.queries, step)
        elif decision.action == "grade_evidence":
            self._run_evidence_curator_agent(state, step)
        elif decision.action == "write_report":
            self._run_writer_agent(state, step, decision.revision_instructions)
        elif decision.action == "critique_report":
            self._run_critic_agent(state, step)

    def _run_planner_agent(self, state: AgentResearchState, step: int) -> None:
        """调用规划专家，把主题拆解成研究问题、检索词和报告章节。"""
        request = state.request
        llm = get_chat_model("dashscope")
        prompt = f"""请为下面研究任务生成 JSON 研究计划。
主题：{request.topic}
深度：{request.depth.value}
报告格式：{request.report_format.value}
语言：{request.language}

JSON schema:
{{
  "title": "string",
  "intent": "string",
  "research_questions": ["string"],
  "local_queries": ["string"],
  "web_queries": ["string"],
  "expected_sections": ["string"],
  "risk_notes": ["string"]
}}
"""
        try:
            response = llm.invoke(
                [SystemMessage(content=PLANNER_SYSTEM), HumanMessage(content=prompt)]
            )
            data = extract_json_object(str(response.content))
            state.plan = QueryPlan.model_validate(data)
            self._record_observation(
                state, step, "planner", f"生成计划：{state.plan.title}"
            )
        except Exception as exc:  # noqa: BLE001 - 规划失败时使用兜底计划保证任务可继续
            state.errors.append(f"planner_fallback: {exc}")
            state.plan = self._fallback_plan(request)
            self._record_observation(state, step, "planner", "使用兜底研究计划。")

    def _run_local_research_agent(
        self,
        state: AgentResearchState,
        queries: list[str],
        step: int,
    ) -> None:
        """调用本地 RAG 专家，从向量库召回证据。"""
        before = len(state.evidence)
        for query in queries:
            try:
                state.evidence.extend(
                    retrieve_local(query, settings=self.settings, k=os.getenv("RA_RETRIEVAL_K"))
                )
            except Exception as exc:  # noqa: BLE001 - 单个查询失败不阻塞其他查询
                state.errors.append(f"local_research:{query}: {exc}")
        self._record_observation(
            state,
            step,
            "local_researcher",
            f"本地检索新增 {len(state.evidence) - before} 条证据。",
        )

    def _run_web_research_agent(
        self,
        state: AgentResearchState,
        queries: list[str],
        step: int,
    ) -> None:
        """调用网页研究专家，补充公开互联网证据。"""
        before = len(state.evidence)
        for query in queries[:6]:
            try:
                state.evidence.extend(tavily_search(query, max_results=5))
            except Exception as exc:  # noqa: BLE001 - 单个网页查询失败不阻塞其他查询
                state.errors.append(f"web_research:{query}: {exc}")
        self._record_observation(
            state,
            step,
            "web_researcher",
            f"网页检索新增 {len(state.evidence) - before} 条证据。",
        )

    def _run_evidence_curator_agent(self, state: AgentResearchState, step: int) -> None:
        """调用证据策展专家，对证据去重、评分和截断。"""
        graded = grade_and_filter(
            state.evidence,
            min_score=self.settings.min_evidence_score,
            limit=self.settings.max_evidence_items,
        )
        state.graded_evidence = [item.model_dump() for item in graded]
        state.evidence = [
            item.evidence.model_copy(update={"score": item.final_score})
            for item in graded
        ]
        self._record_observation(
            state,
            step,
            "evidence_curator",
            f"筛选后保留 {len(state.evidence)} 条证据。",
        )

    def _run_writer_agent(
        self,
        state: AgentResearchState,
        step: int,
        revision_instructions: str = "",
    ) -> None:
        """调用写作专家，基于当前证据生成或重写研究报告。"""
        if state.plan is None:
            state.plan = self._fallback_plan(state.request)

        llm = get_chat_model("dashscope")
        evidence_block = format_evidence_for_prompt(state.evidence)
        critique_block = (
            state.critique.model_dump_json(indent=2)
            if state.critique
            else "暂无质检反馈"
        )
        prompt = f"""研究任务：{state.request.topic}
研究意图：{state.plan.intent}
研究问题：
{chr(10).join(f"- {question}" for question in state.plan.research_questions)}

期望章节：{", ".join(state.plan.expected_sections)}
报告格式：{state.request.report_format.value}
语言：{state.request.language}

可用证据：
{evidence_block if evidence_block else "暂无可用证据"}

上一轮质检反馈：
{critique_block}

主管给出的写作或修改要求：
{revision_instructions or "首次成稿，严格基于证据写作。"}

请写出专业、可执行、证据驱动的 Markdown 报告。每个实质性事实、数字、趋势、因果判断都必须带证据编号。
"""
        response = llm.invoke(
            [SystemMessage(content=WRITER_SYSTEM), HumanMessage(content=prompt)]
        )
        report_md = str(response.content)
        claims = self._extract_claims(report_md, state.evidence)
        state.report = ResearchReport(
            title=state.plan.title,
            executive_summary=self._first_paragraph(report_md),
            report_markdown=report_md,
            claims=claims,
            evidence=state.evidence,
            critique=state.critique,
            agent_trace=state.trace,
        )
        self._record_observation(
            state, step, "writer", f"生成报告，长度 {len(report_md)} 字符。"
        )

    def _run_critic_agent(self, state: AgentResearchState, step: int) -> None:
        """调用质检专家，判断报告是否需要继续检索或重写。"""
        if state.report is None:
            return
        if state.plan is None:
            state.plan = self._fallback_plan(state.request)

        evidence_ids = {item.id for item in state.report.evidence}
        cited_ids = {eid for claim in state.report.claims for eid in claim.evidence_ids}
        heuristic_score = self._heuristic_quality_score(
            state.report, evidence_ids, cited_ids
        )
        missing = [
            question
            for question in state.plan.research_questions
            if question[:10] not in state.report.report_markdown
        ][:5]
        fallback = Critique(
            passed=heuristic_score >= self.settings.quality_threshold
            and not missing[:2],
            quality_score=heuristic_score,
            missing_evidence=missing,
            unsupported_claims=[]
            if state.report.claims
            else ["报告缺少可机器识别的证据引用声明"],
            next_queries=missing[:3],
            notes="启发式兜底质检；主管智能体会基于该结果决定是否继续。",
        )

        prompt = f"""请检查报告质量，并返回 JSON。
研究问题：{json.dumps(state.plan.research_questions, ensure_ascii=False)}
证据 ID：{sorted(evidence_ids)}
已识别引用 ID：{sorted(cited_ids)}
启发式参考分：{heuristic_score}

报告：
{state.report.report_markdown[:6000]}

返回 JSON：
{{
  "passed": true,
  "quality_score": 0.0,
  "missing_evidence": ["还缺什么证据"],
  "unsupported_claims": ["哪些结论证据不足"],
  "next_queries": ["如果需要继续检索，给出查询词"],
  "notes": "简要质检说明"
}}
"""
        try:
            llm = get_chat_model("dashscope")
            response = llm.invoke(
                [SystemMessage(content=CRITIC_SYSTEM), HumanMessage(content=prompt)]
            )
            data = extract_json_object(str(response.content))
            state.critique = Critique.model_validate(data) if data else fallback
        except Exception as exc:  # noqa: BLE001 - 质检模型失败时保留启发式结果
            state.errors.append(f"critic_fallback: {exc}")
            state.critique = fallback

        state.report.critique = state.critique
        self._record_observation(
            state,
            step,
            "critic",
            f"质检分 {state.critique.quality_score:.2f}，passed={state.critique.passed}。",
        )

    def _fallback_decision(self, state: AgentResearchState) -> SupervisorDecision:
        """主管模型无法有效工具调用时，根据状态给出最小可行兜底动作。"""
        if state.plan is None:
            return SupervisorDecision(
                action="plan",
                rationale="缺少研究计划，先规划。",
                source="rule_fallback",
            )
        if not state.evidence:
            return SupervisorDecision(
                action="local_research",
                rationale="缺少证据，先检索本地知识库。",
                queries=self._default_queries_for_research(state, "local_research"),
                source="rule_fallback",
            )
        if not state.graded_evidence:
            return SupervisorDecision(
                action="grade_evidence",
                rationale="已有证据但尚未筛选。",
                source="rule_fallback",
            )
        if state.report is None:
            return SupervisorDecision(
                action="write_report",
                rationale="已有证据，开始写报告。",
                source="rule_fallback",
            )
        if state.critique is None:
            return SupervisorDecision(
                action="critique_report",
                rationale="已有报告，进入质检。",
                source="rule_fallback",
            )
        if state.critique.passed:
            return SupervisorDecision(
                action="finish",
                rationale="质检通过，结束任务。",
                source="rule_fallback",
            )
        return SupervisorDecision(
            action="local_research",
            rationale="质检未通过，继续补充证据。",
            queries=state.critique.next_queries
            or self._default_queries_for_research(state, "local_research"),
            source="rule_fallback",
        )

    def _fallback_plan(self, request: ResearchRequest) -> QueryPlan:
        """在规划专家不可用时生成基础研究计划。"""
        topic = request.topic
        return QueryPlan(
            title=f"{topic}研究报告",
            intent=f"系统分析：{topic}",
            research_questions=[
                f"{topic}的定义、背景和应用边界是什么？",
                f"{topic}当前有哪些主要方法、案例或证据？",
                f"{topic}的风险、限制和未来趋势是什么？",
            ],
            local_queries=[topic, f"{topic} 方法 案例", f"{topic} 风险 限制"],
            web_queries=[topic, f"{topic} latest research", f"{topic} industry report"],
            expected_sections=[
                "摘要",
                "背景",
                "关键发现",
                "风险与限制",
                "建议",
                "参考证据",
            ],
            risk_notes=["缺少足够证据时不得强行下结论"],
        )

    def _default_queries_for_research(
        self,
        state: AgentResearchState,
        action: SupervisorAction,
    ) -> list[str]:
        """在主管没有给出查询词时，从质检、计划或主题中补齐查询。"""
        if state.critique and state.critique.next_queries:
            return state.critique.next_queries[:6]
        if state.plan:
            queries = (
                state.plan.web_queries
                if action == "web_research"
                else state.plan.local_queries
            )
            if queries:
                return queries[:6]
        return [state.request.topic]

    def _state_summary(self, state: AgentResearchState) -> str:
        """压缩当前状态，避免把完整证据和报告全部塞给主管模型。"""
        plan_summary = state.plan.model_dump() if state.plan else None
        evidence_preview = [
            {
                "id": item.id,
                "type": item.source_type.value,
                "title": item.title[:120],
                "score": item.score,
            }
            for item in state.evidence[:10]
        ]
        report_summary = None
        if state.report:
            report_summary = {
                "title": state.report.title,
                "chars": len(state.report.report_markdown),
                "claims": len(state.report.claims),
                "has_critique": state.report.critique is not None,
            }
        payload = {
            "has_plan": state.plan is not None,
            "plan": plan_summary,
            "evidence_count": len(state.evidence),
            "graded_evidence_count": len(state.graded_evidence),
            "evidence_preview": evidence_preview,
            "report": report_summary,
            "critique": state.critique.model_dump() if state.critique else None,
            "errors": state.errors[-5:],
            "recent_trace": state.trace[-6:],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2, default=str)

    def _record_decision(
        self,
        state: AgentResearchState,
        step: int,
        decision: SupervisorDecision,
    ) -> None:
        """记录主管每一轮工具选择，便于展示、调试和评估。"""
        state.trace.append(
            {
                "step": step,
                "agent": "supervisor",
                "action": decision.action,
                "source": decision.source,
                "rationale": decision.rationale,
                "queries": decision.queries,
                "revision_instructions": decision.revision_instructions,
                "confidence": decision.confidence,
            }
        )

    def _record_observation(
        self,
        state: AgentResearchState,
        step: int,
        agent: str,
        summary: str,
    ) -> None:
        """记录专家执行后的观测结果。"""
        state.trace.append({"step": step, "agent": agent, "observation": summary})

    def _force_minimum_report(self, state: AgentResearchState, step: int) -> None:
        """达到安全步数上限仍无报告时，强制产出一份最低可用报告。"""
        if not state.evidence:
            state.evidence = []
        if not state.graded_evidence and state.evidence:
            self._run_evidence_curator_agent(state, step)
        self._run_writer_agent(
            state, step, "达到最大智能体步数，生成当前证据下的最好版本。"
        )
        if state.report and state.critique:
            state.report.critique = state.critique

    def _heuristic_quality_score(
        self,
        report: ResearchReport,
        evidence_ids: set[str],
        cited_ids: set[str],
    ) -> float:
        """给质检模型提供确定性参考分，降低完全主观评分波动。"""
        score = 0.4
        if report.evidence:
            score += min(0.25, len(report.evidence) / 40)
        if report.claims:
            score += min(
                0.25, len(cited_ids & evidence_ids) / max(1, len(evidence_ids))
            )
        if len(report.report_markdown) > 1200:
            score += 0.1
        return min(1.0, score)

    def _first_paragraph(self, markdown: str) -> str:
        """从 Markdown 中提取一个适合作为执行摘要的段落。"""
        for part in markdown.split("\n\n"):
            clean = part.strip(" #\n")
            if len(clean) > 40:
                return clean[:800]
        return markdown[:800]

    def _extract_claims(self, markdown: str, evidence: list[Evidence]) -> list[Claim]:
        """把包含证据 ID 的报告行抽取成后续质检可读取的声明。"""
        ids = {item.id for item in evidence}
        claims: list[Claim] = []
        for line in markdown.splitlines():
            line = line.strip("- *")
            if len(line) < 35:
                continue
            used = [eid for eid in ids if f"[{eid}]" in line]
            if used:
                claims.append(
                    Claim(text=line[:500], evidence_ids=used, confidence=0.75)
                )
            if len(claims) >= 12:
                break
        return claims
