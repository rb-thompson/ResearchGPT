from abc import ABC, abstractmethod
import logging

class BaseAgent(ABC):
    """Abstract base class for research agents.
    Provides common functionality for all agents."""
    def __init__(self, research_assistant):
        """Initialize BaseAgent with research assistant reference."""
        self.assistant = research_assistant
        self.agent_name = "BaseAgent"
        # Retro-style logging with colors
        self.logger = logging.getLogger("BaseAgent")
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "\033[32m%(asctime)s - \033[1;36m%(name)s - \033[1;33m%(levelname)s - \033[1;32m%(message)s\033[0m"
        ))  # Neon green, cyan, yellow
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.logger.info("🟢 BaseAgent initialized! Ready to rock!")

    def _call_mistral(self, prompt: str, temperature: float = None) -> str:
        """Call Mistral API"""
        if temperature is None:
            temperature = self.assistant.config.TEMPERATURE
        self.logger.info("🚀 Sending prompt to Mistral API...")
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
            self.logger.error(f"❌ Mistral API error: {str(e)}")
            return f"Error: {str(e)}"

    @abstractmethod
    def execute_task(self, task_input):
        """Abstract method for agent tasks."""
        raise NotImplementedError("Each agent must implement execute_task method")
    
class SummarizationAgent(BaseAgent):
    """Agent for document summarization tasks."""
    def __init__(self, research_assistant):
        """Initialize SummarizationAgent with research assistant reference."""
        super().__init__(research_assistant)
        self.agent_name = "SummarizerAgent"
        self.logger = logging.getLogger("SummarizerAgent")
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "\033[32m%(asctime)s - \033[1;36m%(name)s - \033[1;33m%(levelname)s - \033[1;32m%(message)s\033[0m"
        ))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.logger.info("📝 SummarizerAgent ready to summarize!")

    def summarize_document(self, doc_id):
        """Summarize a single document by its ID."""
        self.logger.info(f"📄 Summarizing document: {doc_id}")
        try:
            document_text = "\n".join(self.assistant.doc_processor.documents.get(doc_id, {}).get("chunks", []))
            if not document_text:
                self.logger.error(f"❌ No text found for doc_id: {doc_id}")
                return {"error": f"No text for {doc_id}"}
            
            summary_prompt = f"""
            Summarize the following document in 100-150 words:
            - Extract main research question/hypothesis
            - Describe methodology
            - List key findings
            - Note conclusions and limitations
            Document: {document_text}
            """
            summary = self._call_mistral(summary_prompt)
            
            topic_prompt = f"Extract 3-5 key topics from: {summary}"
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

    def create_literature_overview(self, doc_ids):
        """Create a literature overview from multiple document IDs."""
        self.logger.info(f"📚 Creating literature overview for {len(doc_ids)} documents")
        individual_summaries = []
        for doc_id in doc_ids:
            summary = self.summarize_document(doc_id)
            individual_summaries.append(summary)
        
        summaries_text = "\n".join([s.get("summary", "") for s in individual_summaries if "summary" in s])
        overview_prompt = f"""
        Synthesize a literature overview from these summaries:
        - Identify common research themes
        - Note different methodologies
        - Highlight consistent findings vs contradictions
        - Suggest research gaps
        Summaries: {summaries_text}
        """
        overview = self._call_mistral(overview_prompt)
        
        result = {
            "overview": overview,
            "papers_analyzed": len(doc_ids),
            "individual_summaries": individual_summaries
        }
        self.logger.info("🌟 Literature overview completed!")
        return result

    def execute_task(self, task_input):
        """Execute summarization task based on input type."""
        if "doc_id" in task_input:
            return self.summarize_document(task_input["doc_id"])
        elif "doc_ids" in task_input:
            return self.create_literature_overview(task_input["doc_ids"])
        else:
            self.logger.error("❌ Invalid task input")
            return {"error": "Invalid task input for SummarizerAgent"}
        
class QAAgent(BaseAgent):
    """Agent for QA tasks."""
    def __init__(self, research_assistant):
        """Initialize QAAgent with research assistant reference."""
        super().__init__(research_assistant)
        self.agent_name = "QAAgent"
        self.logger = logging.getLogger("QAAgent")
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "\033[32m%(asctime)s - \033[1;36m%(name)s - \033[1;33m%(levelname)s - \033[1;32m%(message)s\033[0m"
        ))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.logger.info("❓ QAAgent ready to answer questions!")

    def answer_factual_question(self, question):
        """Answer a factual question based on document context."""
        self.logger.info(f"❓ Answering factual question: {question}")
        try:
            relevant_chunks = self.assistant.doc_processor.find_similar_chunks(question, top_k=3)
            context = "\n".join([chunk[0] for chunk in relevant_chunks])
            qa_prompt = f"""
            Answer the question concisely based on the context. Cite sources and estimate confidence (0-1).
            Question: {question}
            Context: {context}
            """
            answer = self._call_mistral(qa_prompt)
            
            confidence_prompt = f"Estimate confidence (0-1) for this answer: {answer}"
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

    def answer_analytical_question(self, question):
        """Answer an analytical question using chain-of-thought reasoning."""
        self.logger.info(f"🤔 Answering analytical question: {question}")
        try:
            relevant_chunks = self.assistant.doc_processor.find_similar_chunks(question, top_k=5)
            response = self.assistant.chain_of_thought_reasoning(question, relevant_chunks)
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

    def execute_task(self, task_input):
        """Execute QA task based on question type."""
        question = task_input.get("question", "")
        question_type = task_input.get("type", "factual")
        if question_type == "analytical":
            return self.answer_analytical_question(question)
        else:
            return self.answer_factual_question(question)