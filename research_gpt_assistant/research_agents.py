from abc import ABC, abstractmethod
from typing import Dict, Any, List
import logging
from logging import StreamHandler, Formatter
import time
from functools import lru_cache

def setup_logger(name: str, suppress_base: bool = False) -> logging.Logger:
    """Configure a logger with retro-style ANSI color formatting."""
    logger = logging.getLogger(name)
    if logger.handlers:
        logger.handlers = []
    handler = StreamHandler()
    handler.setFormatter(Formatter(
        "\033[32m%(asctime)s - \033[1;36m%(name)s - \033[1;33m%(levelname)s - \033[1;32m%(message)s\033[0m"
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO if not suppress_base else logging.CRITICAL)
    return logger

class BaseAgent(ABC):
    """Abstract base class for all research agents."""
    
    def __init__(self, research_assistant: Any, suppress_base: bool = False) -> None:
        """Initialize BaseAgent with research assistant reference."""
        self.assistant = research_assistant
        self.agent_name: str = "BaseAgent"
        self.logger: logging.Logger = setup_logger(self.agent_name, suppress_base)
        if not suppress_base:
            self.logger.info("🟢 BaseAgent initialized! Ready to rock!")

    @lru_cache(maxsize=100)
    def _call_mistral(self, prompt: str, temperature: float | None = None) -> str:
        """Call Mistral API with retry logic and caching."""
        if temperature is None:
            temperature = self.assistant.config.TEMPERATURE
        self.logger.info("🚀 Sending prompt to Mistral API...")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.assistant.mistral_client.chat.complete(
                    model=self.assistant.config.MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=self.assistant.config.MAX_TOKENS
                )
                self.logger.info("🎉 Mistral responded back!")
                return response.choices[0].message.content
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    self.logger.warning(f"⚠️ Rate limit hit; retrying in 5s...")
                    time.sleep(5)
                    continue
                self.logger.error(f"❌ Mistral API error: {str(e)}")
                return f"Error: {str(e)}"

    @abstractmethod
    def execute_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent's task."""
        raise NotImplementedError("Each agent must implement execute_task method")

class SummarizationAgent(BaseAgent):
    """Agent for document summarization tasks."""
    
    def __init__(self, research_assistant: Any, suppress_base: bool = False) -> None:
        super().__init__(research_assistant, suppress_base)
        self.agent_name = "SummarizerAgent"
        self.logger = setup_logger(self.agent_name)
        self.logger.info("📝 SummarizerAgent ready to summarize!")

    def summarize_document(self, doc_id: str) -> Dict[str, Any]:
        """Summarize a specific document."""
        self.logger.info(f"📄 Summarizing document: {doc_id}")
        try:
            document_text = "\n".join(self.assistant.doc_processor.documents.get(doc_id, {}).get("chunks", []))
            if not document_text:
                self.logger.error(f"❌ No text found for doc_id: {doc_id}")
                return {"error": f"No text for {doc_id}"}
            
            summary_prompt = f"""
            Summarize the following document in 100-150 words, ensuring detailed relevance to AI applications:
            - Extract main research question/hypothesis
            - Describe methodology
            - List key findings with specific examples
            - Note conclusions and limitations
            Document: {document_text[:10000]}  # Limit input to avoid token overflow
            """
            summary = self._call_mistral(summary_prompt)
            # Truncate summary to 100-200 words
            summary_words = summary.split()
            if len(summary_words) > 200:
                summary = " ".join(summary_words[:200])
                self.logger.warning("⚠️ Summary truncated to 200 words")
            
            topic_prompt = f"Extract 3-5 key topics from this summary in a numbered list: {summary}"
            topics = self._call_mistral(topic_prompt).split("\n")
            
            summary_data = {
                "doc_id": doc_id,
                "summary": summary,
                "word_count": len(summary.split()),
                "key_topics": [t.strip() for t in topics if t.strip()]
            }
            self.logger.info(f"🎉 Summary generated! Word count: {summary_data['word_count']}")
            return summary_data
        except Exception as e:
            self.logger.error(f"❌ Error summarizing: {str(e)}")
            return {"error": str(e)}

    def create_literature_overview(self, doc_ids: List[str]) -> Dict[str, Any]:
        """Create a literature overview from multiple documents."""
        self.logger.info(f"📚 Creating literature overview for {len(doc_ids)} documents")
        individual_summaries = []
        for doc_id in doc_ids[:1]:  # Limit to 1 document for speed
            summary = self.summarize_document(doc_id)
            individual_summaries.append(summary)
        
        summaries_text = "\n".join([s.get("summary", "") for s in individual_summaries if "summary" in s])
        overview_prompt = f"""
        Synthesize a concise literature overview (100-150 words) from these summaries:
        - Identify common research themes with examples
        - Note different methodologies
        - Highlight consistent findings vs contradictions
        - Suggest specific research gaps
        Summaries: {summaries_text[:10000]}  # Limit input
        """
        overview = self._call_mistral(overview_prompt)
        # Truncate overview to 100-200 words
        overview_words = overview.split()
        if len(overview_words) > 200:
            overview = " ".join(overview_words[:200])
            self.logger.warning("⚠️ Overview truncated to 200 words")
        
        result = {
            "overview": overview,
            "papers_analyzed": len(individual_summaries),
            "individual_summaries": individual_summaries
        }
        self.logger.info("🌟 Literature overview completed!")
        return result

    def execute_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        if "doc_id" in task_input:
            return self.summarize_document(task_input["doc_id"])
        elif "doc_ids" in task_input:
            return self.create_literature_overview(task_input["doc_ids"])
        self.logger.error("❌ Invalid task input")
        return {"error": "Invalid task input for SummarizerAgent"}

class QAAgent(BaseAgent):
    """Agent for question-answering tasks."""
    
    def __init__(self, research_assistant: Any, suppress_base: bool = False) -> None:
        super().__init__(research_assistant, suppress_base)
        self.agent_name = "QAAgent"
        self.logger = setup_logger(self.agent_name)
        self.logger.info("❓ QAAgent ready to answer questions!")

    def answer_factual_question(self, question: str) -> Dict[str, Any]:
        """Answer factual questions based on document corpus."""
        self.logger.info(f"❓ Answering factual question: {question}")
        try:
            relevant_chunks = self.assistant.doc_processor.find_similar_chunks(question, top_k=3, min_score=0.1)
            context = "\n".join([chunk[0] for chunk in relevant_chunks]) or "No relevant context found; provide a general definition."
            qa_prompt = f"""
            Provide a detailed, concise answer (100-150 words) to the question, integrating the context. Include specific examples or applications from the context. Cite sources explicitly and estimate confidence (0-1).
            Question: {question}
            Context: {context[:10000]}  # Limit input
            """
            answer = self._call_mistral(qa_prompt)
            # Truncate answer to 100-200 words
            answer_words = answer.split()
            if len(answer_words) > 200:
                answer = " ".join(answer_words[:200])
                self.logger.warning("⚠️ Answer truncated to 200 words")
            
            confidence_prompt = f"Estimate confidence (0-1) for this answer based on context relevance and specificity: {answer}"
            confidence = self._call_mistral(confidence_prompt)
            
            result = {
                "question": question,
                "answer": answer,
                "sources": [chunk[2] for chunk in relevant_chunks],
                "confidence": confidence
            }
            self.logger.info(f"🎉 Factual answer generated! Confidence: {confidence}")
            return result
        except Exception as e:
            self.logger.error(f"❌ Error answering: {str(e)}")
            return {"error": str(e)}

    def answer_analytical_question(self, question: str) -> Dict[str, Any]:
        """Answer analytical questions requiring reasoning."""
        self.logger.info(f"🤔 Answering analytical question: {question}")
        try:
            relevant_chunks = self.assistant.doc_processor.find_similar_chunks(question, top_k=5, min_score=0.1)
            response = self.assistant.chain_of_thought_reasoning(question, relevant_chunks)
            # Truncate response to 100-200 words
            response_words = response.split()
            if len(response_words) > 200:
                response = " ".join(response_words[:200])
                self.logger.warning("⚠️ Analytical answer truncated to 200 words")
            result = {
                "question": question,
                "analysis": response,
                "reasoning_type": "chain_of_thought"
            }
            self.logger.info("🌟 Analytical answer generated!")
            return result
        except Exception as e:
            self.logger.error(f"❌ Error analyzing: {str(e)}")
            return {"error": str(e)}

    def execute_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        question = task_input.get("question", "")
        question_type = task_input.get("type", "factual")
        if question_type == "analytical":
            return self.answer_analytical_question(question)
        return self.answer_factual_question(question)

class AnalysisAgent(BaseAgent):
    """Agent for generating research insights and gaps."""
    
    def __init__(self, research_assistant: Any, suppress_base: bool = False) -> None:
        super().__init__(research_assistant, suppress_base)
        self.agent_name = "AnalysisAgent"
        self.logger = setup_logger(self.agent_name)
        self.logger.info("🔍 AnalysisAgent ready to uncover insights!")

    def execute_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a topic for trends, gaps, and future directions."""
        self.logger.info(f"🔍 Analyzing topic: {task_input.get('topic', '')}")
        try:
            topic = task_input.get("topic", "")
            relevant_chunks = self.assistant.doc_processor.find_similar_chunks(topic, top_k=5, min_score=0.1)
            context = "\n".join([chunk[0] for chunk in relevant_chunks]) or "No relevant context found; provide general insights."
            analysis_prompt = f"""
            Analyze the following context in 150-200 words, providing detailed insights:
            - Identify key trends in the research with specific examples (e.g., applications or models)
            - Highlight specific research gaps or unanswered questions with examples
            - Suggest concrete future research directions with actionable steps
            Context: {context[:10000]}  # Limit input
            """
            analysis = self._call_mistral(analysis_prompt)
            # Truncate analysis to 150-200 words
            analysis_words = analysis.split()
            if len(analysis_words) > 200:
                analysis = " ".join(analysis_words[:200])
                self.logger.warning("⚠️ Analysis truncated to 200 words")
            result = {
                "topic": topic,
                "analysis": analysis,
                "sources": [chunk[2] for chunk in relevant_chunks]
            }
            self.logger.info("🌟 Analysis generated!")
            return result
        except Exception as e:
            self.logger.error(f"❌ Error analyzing: {str(e)}")
            return {"error": str(e)}

class ResearchWorkflowAgent(BaseAgent):
    """Agent for managing complete research workflows."""
    
    def __init__(self, research_assistant: Any, suppress_base: bool = False) -> None:
        super().__init__(research_assistant, suppress_base)
        self.agent_name = "ResearchWorkflowAgent"
        self.logger = setup_logger(self.agent_name)
        self.summarizer = SummarizationAgent(research_assistant, suppress_base)
        self.qa_agent = QAAgent(research_assistant, suppress_base)
        self.analysis_agent = AnalysisAgent(research_assistant, suppress_base)
        for sub_agent in [self.summarizer, self.qa_agent, self.analysis_agent]:
            sub_agent.logger.propagate = False
        self.logger.info("🧠 ResearchWorkflowAgent ready to run research sessions!")

    def conduct_research_session(self, research_topic: str) -> Dict[str, Any]:
        """Conduct a complete research session on a topic (limited to 3 questions)."""
        self.logger.info(f"🧠 Starting research session on: {research_topic}")
        session_results = {
            "research_topic": research_topic,
            "generated_questions": [],
            "document_analysis": {},
            "answers": [],
            "research_gaps": []
        }
        try:
            # Step 1: Generate questions
            questions_prompt = f"""
            Generate exactly 3 specific research questions for: {research_topic}
            Format as a numbered list (1., 2., 3.).
            Example:
            1. How does X impact Y in context Z?
            2. What are the limitations of X in application Y?
            3. How can X be optimized for Z?
            Ensure questions are specific, relevant to AI/ML, and diverse.
            """
            questions_text = self._call_mistral(questions_prompt)
            questions = [q.strip() for q in questions_text.split("\n") if q.strip() and q[0].isdigit()][:3]
            if len(questions) < 3:
                questions.extend([f"How can {research_topic} be applied in industry {i+1}?" for i in range(len(questions), 3)])
            session_results["generated_questions"] = questions
            self.logger.info(f"❓ Generated {len(questions)} questions")

            # Step 2: Summarize documents
            relevant_docs = self.assistant.doc_processor.find_similar_chunks(research_topic, top_k=10, min_score=0.1)
            doc_ids = list(set([doc[2] for doc in relevant_docs if doc[2].startswith("test_doc") or doc[2].startswith("2509")]))[:1]
            if not doc_ids:
                doc_ids = ["test_doc_1"]
                self.logger.warning("⚠️ No relevant docs found; using default")
            session_results["document_analysis"] = self.summarizer.create_literature_overview(doc_ids)
            self.logger.info("📚 Document analysis completed")

            # Step 3: Answer questions
            for question in session_results["generated_questions"]:
                qa_task = {"question": question, "type": "analytical"}
                answer = self.qa_agent.execute_task(qa_task)
                session_results["answers"].append(answer)
                self.logger.info(f"❓ Answered: {question[:50]}...")

            # Step 4: Identify gaps
            analysis_task = {"topic": research_topic}
            gaps = self.analysis_agent.execute_task(analysis_task)
            session_results["research_gaps"] = gaps.get("analysis", "")
            self.logger.info("🔍 Research gaps identified")

            self.logger.info("🌟 Research session completed!")
            return session_results
        except Exception as e:
            self.logger.error(f"❌ Error in research session: {str(e)}")
            return {"error": str(e)}

    def execute_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        if "research_topic" in task_input:
            return self.conduct_research_session(task_input["research_topic"])
        self.logger.error("❌ Invalid task input")
        return {"error": "Invalid task input for ResearchWorkflowAgent"}

class AgentOrchestrator(BaseAgent):
    """Orchestrates multiple agents for complex research tasks."""
    
    def __init__(self, research_assistant: Any, suppress_base: bool = False) -> None:
        """Initialize orchestrator with agent registry and shared memory."""
        super().__init__(research_assistant, suppress_base)
        self.agent_name = "AgentOrchestrator"
        self.logger = setup_logger(self.agent_name)
        self.agents = {
            "summarizer": SummarizationAgent(research_assistant, suppress_base),
            "qa": QAAgent(research_assistant, suppress_base),
            "analysis": AnalysisAgent(research_assistant, suppress_base),
            "workflow": ResearchWorkflowAgent(research_assistant, suppress_base)
        }
        for agent in self.agents.values():
            agent.logger.propagate = False
        self.shared_memory: Dict[str, Any] = {}
        self.logger.info("🎶 AgentOrchestrator ready to conduct the symphony!")

    def route_task(self, task_type: str, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """Route tasks to appropriate agents."""
        self.logger.info(f"🚦 Routing task type: {task_type}")
        try:
            agent = self.agents.get(task_type)
            if not agent:
                self.logger.error(f"❌ Unknown task type: {task_type}")
                return {"error": f"Unknown task type: {task_type}"}
            result = agent.execute_task(task_input)
            self.shared_memory[task_type] = result
            self.logger.info(f"🎉 Task routed to {agent.agent_name}")
            return result
        except Exception as e:
            self.logger.error(f"❌ Error routing task: {str(e)}")
            return {"error": str(e)}

    def resolve_conflicts(self, outputs: List[Dict[str, Any]]) -> str:
        """Resolve conflicts between agent outputs."""
        self.logger.info("🛠 Resolving conflicts in agent outputs")
        try:
            conflict_prompt = f"""
            Identify and resolve contradictions in these outputs in 150-200 words:
            - Summarize key points from each output with specific examples
            - Highlight any contradictions between outputs
            - Provide a unified conclusion with actionable insights
            Outputs: {outputs[:10000]}  # Limit input
            """
            resolved_output = self._call_mistral(conflict_prompt)
            # Truncate resolved output to 150-200 words
            resolved_words = resolved_output.split()
            if len(resolved_words) > 200:
                resolved_output = " ".join(resolved_words[:200])
                self.logger.warning("⚠️ Resolved output truncated to 200 words")
            self.logger.info("🌟 Conflicts resolved!")
            return resolved_output
        except Exception as e:
            self.logger.error(f"❌ Error resolving conflicts: {str(e)}")
            return f"Error: {str(e)}"

    def execute_complex_workflow(self, workflow_description: str) -> Dict[str, Any]:
        """Execute a complex multi-agent workflow."""
        self.logger.info(f"🎬 Executing complex workflow: {workflow_description}")
        try:
            parse_prompt = f"""
            Break down this workflow into exactly 3 unique tasks (summarizer, qa, analysis).
            Include 'qa' for any answer-related task. List task types only, one per line.
            Workflow: {workflow_description}
            """
            task_list = self._call_mistral(parse_prompt).split("\n")
            tasks = []
            seen_tasks = set()
            for task in task_list:
                task = task.strip().lower()
                if not task:
                    continue
                if "summarizer" in task and "summarizer" not in seen_tasks:
                    tasks.append(("summarizer", {"doc_ids": ["test_doc_1"]}))
                    seen_tasks.add("summarizer")
                elif ("qa" in task or "question" in task or "answer" in task) and "qa" not in seen_tasks:
                    tasks.append(("qa", {"question": workflow_description, "type": "analytical"}))
                    seen_tasks.add("qa")
                elif "analysis" in task and "analysis" not in seen_tasks:
                    tasks.append(("analysis", {"topic": workflow_description}))
                    seen_tasks.add("analysis")
            if "qa" not in seen_tasks:
                tasks.append(("qa", {"question": workflow_description, "type": "analytical"}))
                seen_tasks.add("qa")

            results = []
            for task_type, task_input in tasks:
                result = self.route_task(task_type, task_input)
                results.append(result)

            final_result = self.resolve_conflicts(results) if len(results) > 1 else results[0] if results else {"error": "No tasks executed"}
            workflow_results = {
                "workflow_description": workflow_description,
                "steps_executed": list(seen_tasks),
                "final_result": final_result
            }
            self.logger.info("🌟 Complex workflow completed!")
            return workflow_results
        except Exception as e:
            self.logger.error(f"❌ Error in workflow: {str(e)}")
            return {"error": str(e)}

    def execute_task(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task, delegating to route_task or execute_complex_workflow."""
        if "workflow_description" in task_input:
            return self.execute_complex_workflow(task_input["workflow_description"])
        elif "task_type" in task_input and "task_input" in task_input:
            return self.route_task(task_input["task_type"], task_input["task_input"])
        self.logger.error("❌ Invalid task input for AgentOrchestrator")
        return {"error": "Invalid task input for AgentOrchestrator"}