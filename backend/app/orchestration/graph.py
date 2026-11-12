import logging
from datetime import datetime
from typing import Dict, Any, List, Literal
from langgraph.graph import StateGraph, END
from backend.app.core.datetime_utils import get_ist_iso

from backend.app.orchestration.state import AgentState
from backend.app.agents.task_understanding import task_understanding_agent
from backend.app.agents.planner import planner_agent
from backend.app.agents.research import research_agent
from backend.app.agents.execution import execution_agent
from backend.app.agents.memory_agent import memory_agent
from backend.app.agents.evaluator import evaluator_agent
from backend.app.agents.response import response_agent

logger = logging.getLogger(__name__)

def _add_event(state: AgentState, event_type: str, agent_name: str, message: str, data: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    events = list(state.get("events", []))
    events.append({
        "timestamp": get_ist_iso(),
        "event_type": event_type,
        "agent_name": agent_name,
        "message": message,
        "data": data or {}
    })
    return events

# --- Node 1: Retrieve Memory Context ---
async def node_memory_retrieve(state: AgentState) -> Dict[str, Any]:
    user_id = state.get("user_id", "default_user")
    session_id = state.get("session_id")
    prompt = state.get("user_prompt", "")
    current_msg_id = state.get("current_message_id")

    # 1. Retrieve user long-term preferences/facts
    ctx = await memory_agent.retrieve_context(user_id)

    # 2. Retrieve session conversation history (window=10)
    conv_ctx = ""
    chat_history = []
    if session_id:
        chat_history = await memory_agent.get_session_messages(session_id)
        conv_ctx = await memory_agent.retrieve_conversation_context(
            session_id=session_id,
            window=10,
            exclude_id=current_msg_id
        )

    # 3. Check for deterministic memory/preference/history queries
    deterministic_result = await memory_agent.resolve_deterministic_memory_query(
        session_id=session_id or "",
        prompt=prompt,
        current_message_id=current_msg_id,
        user_id=user_id
    )

    events = list(state.get("events", []))
    events.append({
        "timestamp": get_ist_iso(),
        "event_type": "agent_start",
        "agent_name": "MemoryAgent",
        "message": "Loading conversation history and user memory...",
        "data": {"session_id": session_id}
    })

    if deterministic_result and deterministic_result.get("found"):
        target_str = str(deterministic_result.get("target_content"))
        preview = (target_str[:80] + "...") if len(target_str) > 80 else target_str
        events.append({
            "timestamp": get_ist_iso(),
            "event_type": "agent_thought",
            "agent_name": "MemoryAgent",
            "message": f"Retrieved historical session record: '{preview}'",
            "data": deterministic_result
        })
    elif session_id and conv_ctx:
        events.append({
            "timestamp": get_ist_iso(),
            "event_type": "agent_thought",
            "agent_name": "MemoryAgent",
            "message": f"Loaded active session history ({len(chat_history)} messages in session '{session_id}').",
            "data": {"session_id": session_id, "messages_count": len(chat_history)}
        })
    else:
        events.append({
            "timestamp": get_ist_iso(),
            "event_type": "agent_thought",
            "agent_name": "MemoryAgent",
            "message": f"Retrieved historical context and preferences for user '{user_id}'.",
            "data": {}
        })

    return {
        "memory_context": ctx,
        "conversation_context": conv_ctx,
        "chat_history": chat_history,
        "deterministic_memory_result": deterministic_result,
        "events": events
    }

# --- Node 2: Task Understanding ---
async def node_task_understanding(state: AgentState) -> Dict[str, Any]:
    events = _add_event(state, "agent_start", "TaskUnderstandingAgent", "Analyzing natural language instruction and decomposing intent...")
    res = await task_understanding_agent.process(dict(state))
    task_spec = res.get("task_spec", {})
    events = _add_event(state, "agent_thought", "TaskUnderstandingAgent", f"Classified task as '{task_spec.get('task_type')}' with objective: {task_spec.get('objective')}", task_spec)
    return {
        "task_spec": task_spec,
        "events": events
    }

# --- Node 3: Planner Agent ---
async def node_planner(state: AgentState) -> Dict[str, Any]:
    retry_instructions = state.get("retry_instructions")
    if retry_instructions:
        msg = f"Self-Correction Replanning triggered (Retry #{state.get('retry_count', 0) + 1}): {retry_instructions}"
        events = _add_event(state, "agent_thought", "PlannerAgent", msg)
    else:
        events = _add_event(state, "agent_start", "PlannerAgent", "Constructing structured multi-step execution plan...")

    res = await planner_agent.process(dict(state))
    plan = res.get("execution_plan", {})
    steps_count = len(plan.get("steps", []))
    events = _add_event(state, "agent_thought", "PlannerAgent", f"Formulated {steps_count}-step execution sequence.", plan)
    
    # Reset step results on replanning if necessary
    step_results = state.get("step_results", {}) if not retry_instructions else {}
    return {
        "execution_plan": plan,
        "step_results": step_results,
        "events": events
    }

# --- Node 4: Step Execution ---
async def node_execute_step(state: AgentState) -> Dict[str, Any]:
    plan = state.get("execution_plan", {})
    steps = plan.get("steps", [])
    step_results = dict(state.get("step_results", {}))
    artifacts = list(state.get("artifacts", []))
    events = list(state.get("events", []))

    completed_indices = {int(k) for k in step_results.keys()}
    next_step = None
    for s in steps:
        s_idx = s.get("step_index")
        if s_idx not in completed_indices:
            deps = s.get("depends_on", [])
            if all(d in completed_indices for d in deps):
                next_step = s
                break

    if not next_step:
        return {"status": "EXECUTION_COMPLETE"}

    step_idx = next_step.get("step_index")
    agent_name = next_step.get("agent_name", "Execution")
    tool_name = next_step.get("tool_name")
    desc = next_step.get("description", "")

    events.append({
        "timestamp": get_ist_iso(),
        "event_type": "tool_call",
        "agent_name": agent_name,
        "message": f"Step {step_idx}: Executing '{desc}' via tool: {tool_name or 'LLM Direct'}",
        "data": {"step": next_step}
    })

    if tool_name in ["calculator", "data_analysis", "file_gen", "email", "search"]:
        tpl = dict(next_step.get("tool_input_template") or {})
        # Propagate attached dataset only to tools that accept file_path (e.g. data_analysis)
        if tool_name == "data_analysis" and state.get("attached_files") and not tpl.get("file_path"):
            tpl["file_path"] = state.get("attached_files")[0]
            next_step["tool_input_template"] = tpl

        step_output = await execution_agent.execute_step(
            next_step,
            step_results,
            user_prompt=state.get("user_prompt"),
            user_id=state.get("user_id"),
            memory_context=state.get("memory_context")
        )
    elif tool_name == "rag" or (agent_name == "Research" and not tool_name):
        input_tpl = next_step.get("tool_input_template", {}) or {}
        query_val = input_tpl.get("query") or state.get("user_prompt") or desc
        events.append({
            "timestamp": get_ist_iso(),
            "event_type": "agent_thought",
            "agent_name": "ResearchAgent",
            "message": f"Querying ChromaDB vector store for semantic matches to: '{query_val}'...",
            "data": {"query": query_val}
        })
        research_out = await research_agent.process({
            "current_step_query": query_val,
            "user_prompt": state.get("user_prompt"),
            "user_id": state.get("user_id"),
            "memory_context": state.get("memory_context"),
            "conversation_context": state.get("conversation_context", ""),
            "attached_files": state.get("attached_files", []),
            "task_spec": state.get("task_spec", {})
        })
        cits = research_out.get("citations", [])
        chunks_count = len(research_out.get("context_chunks", []))
        events.append({
            "timestamp": get_ist_iso(),
            "event_type": "tool_output",
            "agent_name": "ResearchAgent",
            "message": f"Retrieved {len(cits)} grounded citations across {chunks_count} knowledge passages.",
            "data": {"citations": cits}
        })
        step_output = {
            "step_index": step_idx,
            "tool_name": "rag",
            "success": True,
            "output": research_out.get("research_findings"),
            "citations": cits,
            "context_chunks": research_out.get("context_chunks", []),
            "target_source": research_out.get("target_source"),
            "artifacts": []
        }
    elif tool_name == "memory" or agent_name == "Memory":
        det_res = state.get("deterministic_memory_result")
        if det_res:
            out_str = det_res.get("formatted_answer", "Found record in session memory.")
        else:
            out_str = state.get("conversation_context") or "No prior conversation records found."

        events.append({
            "timestamp": get_ist_iso(),
            "event_type": "tool_output",
            "agent_name": "MemoryAgent",
            "message": "Extracted conversation record from session memory.",
            "data": {"output_preview": str(out_str)[:150]}
        })
        step_output = {
            "step_index": step_idx,
            "tool_name": "memory",
            "success": True,
            "output": out_str,
            "citations": [f"Session Memory (Session ID: {state.get('session_id', 'active')})"],
            "context_chunks": [],
            "artifacts": []
        }
    else:
        tpl = dict(next_step.get("tool_input_template") or {})
        if tool_name == "data_analysis" and state.get("attached_files") and not tpl.get("file_path"):
            tpl["file_path"] = state.get("attached_files")[0]
            next_step["tool_input_template"] = tpl

        step_output = await execution_agent.execute_step(
            next_step,
            step_results,
            user_prompt=state.get("user_prompt"),
            user_id=state.get("user_id"),
            memory_context=state.get("memory_context")
        )

    step_results[str(step_idx)] = step_output
    if step_output.get("artifacts"):
        artifacts.extend(step_output["artifacts"])

    events.append({
        "timestamp": get_ist_iso(),
        "event_type": "step_complete",
        "agent_name": agent_name,
        "message": f"Step {step_idx} completed successfully.",
        "data": {"output_preview": str(step_output.get("output"))[:150]}
    })

    return {
        "step_results": step_results,
        "artifacts": list(set(artifacts)),
        "events": events
    }

def route_execution(state: AgentState) -> Literal["execute_step", "evaluator"]:
    plan = state.get("execution_plan", {})
    steps = plan.get("steps", [])
    step_results = state.get("step_results", {})
    completed_indices = {int(k) for k in step_results.keys()}

    # Check if there are remaining steps ready to run
    has_remaining = any(s.get("step_index") not in completed_indices for s in steps)
    if has_remaining:
        return "execute_step"
    return "evaluator"

# --- Node 5: Evaluator Agent ---
async def node_evaluator(state: AgentState) -> Dict[str, Any]:
    events = _add_event(state, "agent_start", "EvaluatorAgent", "Evaluating consistency, completeness, and correctness of outputs...")
    res = await evaluator_agent.process(dict(state))
    evaluation = res.get("evaluation", {})
    needs_retry = res.get("needs_retry", False)
    retry_instructions = res.get("retry_instructions")

    score = evaluation.get("score", 1.0)
    events = _add_event(
        state,
        "evaluation",
        "EvaluatorAgent",
        f"Evaluation score: {score*100:.1f}%. Satisfactory: {evaluation.get('is_satisfactory')}. Needs Retry: {needs_retry}",
        evaluation
    )

    updates: Dict[str, Any] = {
        "evaluation": evaluation,
        "needs_retry": needs_retry,
        "retry_instructions": retry_instructions,
        "events": events
    }

    if needs_retry:
        updates["retry_count"] = state.get("retry_count", 0) + 1

    return updates

def route_evaluation(state: AgentState) -> Literal["planner", "response"]:
    if state.get("needs_retry") and state.get("retry_count", 0) <= state.get("max_retries", 2):
        return "planner"
    return "response"

# --- Node 6: Response Agent ---
async def node_response(state: AgentState) -> Dict[str, Any]:
    events = _add_event(state, "agent_start", "ResponseAgent", "Synthesizing executive markdown response and compiling artifacts...")
    res = await response_agent.process(dict(state))
    final_text = res.get("final_response", "")
    artifacts = res.get("final_artifacts", [])

    events = _add_event(state, "task_complete", "ResponseAgent", "Final response formulated successfully.")
    return {
        "final_response": final_text,
        "artifacts": artifacts,
        "status": "COMPLETED",
        "events": events
    }

# --- Node 7: Persist Memory ---
async def node_memory_persist(state: AgentState) -> Dict[str, Any]:
    user_id = state.get("user_id", "default_user")
    session_id = state.get("session_id")
    prompt = state.get("user_prompt", "")
    final_resp = state.get("final_response", "")
    await memory_agent.persist_execution_memory(user_id, prompt, final_resp, session_id=session_id)
    events = _add_event(state, "agent_thought", "MemoryAgent", "Saved task outcome into user execution history and persistent session memory.")
    return {"events": events}

# Build StateGraph
def build_multi_agent_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("memory_retrieve", node_memory_retrieve)
    builder.add_node("task_understanding", node_task_understanding)
    builder.add_node("planner", node_planner)
    builder.add_node("execute_step", node_execute_step)
    builder.add_node("evaluator", node_evaluator)
    builder.add_node("response", node_response)
    builder.add_node("memory_persist", node_memory_persist)

    # Define edges
    builder.set_entry_point("memory_retrieve")
    builder.add_edge("memory_retrieve", "task_understanding")
    builder.add_edge("task_understanding", "planner")
    builder.add_edge("planner", "execute_step")

    builder.add_conditional_edges(
        "execute_step",
        route_execution,
        {
            "execute_step": "execute_step",
            "evaluator": "evaluator"
        }
    )

    builder.add_conditional_edges(
        "evaluator",
        route_evaluation,
        {
            "planner": "planner", # Self-correction feedback loop!
            "response": "response"
        }
    )

    builder.add_edge("response", "memory_persist")
    builder.add_edge("memory_persist", END)

    return builder.compile()

multi_agent_graph = build_multi_agent_graph()
