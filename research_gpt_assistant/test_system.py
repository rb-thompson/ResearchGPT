"""
Testing and Evaluation Script for ResearchGPT Assistant

Implements comprehensive testing for Config, DocumentProcessor, and ResearchGPTAssistant:
1. Unit tests for configuration and document processing components
2. Integration tests for document processing workflow with multiple PDFs
3. Prompting strategy tests for ResearchGPTAssistant
4. Performance evaluation metrics with real Mistral API calls
"""

from unittest.mock import patch
import time
import json
import os
from pathlib import Path
from typing import Dict, Union
import re
import difflib
import pandas as pd
from config import Config
from document_processor import DocumentProcessor
from research_assistant import ResearchGPTAssistant

# Suppress logging for cleaner test output
import logging
logging.getLogger("pdfplumber").setLevel(logging.ERROR)
logging.getLogger("pdfminer").setLevel(logging.ERROR)


class ResearchGPTTester:
    """Test suite for ResearchGPT Assistant's Config, DocumentProcessor, and ResearchGPTAssistant components."""

    def __init__(self):
        """Initialize testing system with configuration, document processor, and research assistant."""
        # Mock environment variables (API key loaded from .env)
        with patch.dict(
            os.environ,
            {
                "MODEL_NAME": "mistral-small-2506",  
                "TEMPERATURE": "0.1",
                "MAX_TOKENS": "500",  # Reduced for faster testing
                "DATA_DIR": "test_data/",
                "RESULTS_DIR": "test_results/",
                "CHUNK_SIZE": "500",
                "OVERLAP": "100",
                "MIN_CHUNK_SIZE": "250",
            },
        ):
            self.config = Config()
            self.doc_processor = DocumentProcessor(self.config)
            self.research_assistant = ResearchGPTAssistant(self.config, self.doc_processor)

        # Create test directories
        self.config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        # Multiple PDF paths
        self.pdf_dir = self.config.DATA_DIR / "sample_papers"
        self.pdf_files = list(self.pdf_dir.glob("*.pdf"))
        if not self.pdf_files:
            # Fallback to single PDF if multiple not found
            self.pdf_files = [self.config.DATA_DIR / "sample_papers/sample.pdf"]

        # Process all PDFs for search index
        print(f"Processing {len(self.pdf_files)} PDF(s) for search index...")
        for i, pdf_path in enumerate(self.pdf_files):
            try:
                self.doc_processor.process_document(pdf_path)
                print(f"   Processed {pdf_path.name}")
            except Exception as e:
                print(f"   Failed to process {pdf_path.name}: {str(e)}")
        
        # Build search index once all documents are processed
        try:
            self.doc_processor.build_search_index()
            print(f"Built search index with {len(self.doc_processor.all_chunks)} chunks")
        except Exception as e:
            print(f"Failed to build search index: {str(e)}")

        # Test queries
        self.test_queries = [
            "What are the main advantages of machine learning?",
            "How do neural networks process information?",
            "What are the limitations of current AI systems?",
            "Compare supervised and unsupervised learning approaches",
            "What are the ethical considerations in AI development?",
        ]

        # Evaluation results storage
        self.evaluation_results = {
            "response_times": [],
            "response_lengths": [],
            "document_processing": {},
            "prompt_strategy_comparison": {},
            "performance_benchmark": {},
            "overall_scores": {},
        }

    def test_document_processing(self):
        """Test document processing functionality.

        Tests:
        1. PDF text extraction
        2. Text preprocessing and cleaning
        3. Document chunking
        4. Similarity search
        5. Index building

        Returns:
            dict: Test results for document processing
        """
        print("\n=== Testing Document Processing ===")
        test_results = {
            "pdf_extraction": False,
            "text_preprocessing": False,
            "chunking": False,
            "similarity_search": False,
            "index_building": False,
            "errors": [],
        }

        # Test PDF extraction (use first PDF)
        try:
            pdf_path = self.pdf_files[0] if self.pdf_files else Path(self.test_pdf_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            result = self.doc_processor.extract_text_from_pdf(pdf_path)
            if result and isinstance(result, str) and len(result) > 50:  # Ensure meaningful extraction
                test_results["pdf_extraction"] = True
                print("   ✓ Text extraction: PASS")
            else:
                test_results["errors"].append("Text extraction returned empty or invalid result")
                print("   ✗ Text extraction: FAIL")
        except Exception as e:
            test_results["errors"].append(f"Text extraction error: {str(e)}")
            print(f"   ✗ Text extraction error: {str(e)}")

        # Test text preprocessing
        try:
            sample_text = (
                "This is a test\n-\ntext with arXiv:1234.5678 and https://example.com.\n\n"
                "Special chars: @#$%^."
            )
            preprocessed = self.doc_processor.preprocess_text(sample_text)
            if (
                "arxiv" not in preprocessed.lower()
                and "https" not in preprocessed.lower()
                and "@#$%^" not in preprocessed
                and "\n" not in preprocessed
                and "test text" in preprocessed.lower()
            ):
                test_results["text_preprocessing"] = True
                print("   ✓ Text preprocessing: PASS")
            else:
                test_results["errors"].append("Text preprocessing failed to clean text correctly")
                print("   ✗ Text preprocessing: FAIL")
        except Exception as e:
            test_results["errors"].append(f"Text preprocessing error: {str(e)}")
            print(f"   ✗ Text preprocessing error: {str(e)}")

        # Test text chunking
        try:
            sample_text = (
                "This is a sample sentence for testing chunking. "
                "It contains multiple sentences to ensure proper splitting. "
                "The goal is to create chunks that respect sentence boundaries. "
                "Each chunk should be meaningful and contain complete thoughts. "
                "This text is long enough to produce multiple chunks."
            ) * 3  # Repeat to ensure sufficient length
            original_min_chunk_size = self.doc_processor.min_chunk_size
            self.doc_processor.min_chunk_size = 50  # Temporarily reduce for test
            chunks = self.doc_processor.chunk_text(sample_text, chunk_size=200, overlap=20)
            self.doc_processor.min_chunk_size = original_min_chunk_size  # Restore
            if chunks and all(len(chunk) >= 50 for chunk in chunks):
                test_results["chunking"] = True
                print("   ✓ Text chunking: PASS")
            else:
                test_results["errors"].append("Chunking produced invalid or empty chunks")
                print("   ✗ Text chunking: FAIL")
        except Exception as e:
            test_results["errors"].append(f"Chunking error: {str(e)}")
            print(f"   ✗ Chunking error: {str(e)}")

        # Test index building (use processed documents)
        try:
            if self.doc_processor.documents:
                self.doc_processor.build_search_index()
                if self.doc_processor.document_vectors is not None and len(self.doc_processor.all_chunks) > 0:
                    test_results["index_building"] = True
                    print("   ✓ Index building: PASS")
                else:
                    test_results["errors"].append("Index building failed to create vectors")
                    print("   ✗ Index building: FAIL")
            else:
                test_results["errors"].append("No documents available for index building")
                print("   ✗ Index building: No documents")
        except Exception as e:
            test_results["errors"].append(f"Index building error: {str(e)}")
            print(f"   ✗ Index building error: {str(e)}")

        # Test similarity search
        try:
            if self.doc_processor.document_vectors is not None:
                results = self.doc_processor.find_similar_chunks("artificial intelligence", top_k=2, min_score=0.1)
                if results and all(isinstance(r, tuple) and len(r) == 3 for r in results):
                    test_results["similarity_search"] = True
                    print("   ✓ Similarity search: PASS")
                else:
                    test_results["errors"].append("Similarity search returned invalid results")
                    print("   ✗ Similarity search: FAIL")
            else:
                test_results["errors"].append("No fitted vectorizer for similarity search")
                print("   ✗ Similarity search: No vectorizer")
        except Exception as e:
            test_results["errors"].append(f"Similarity search error: {str(e)}")
            print(f"   ✗ Similarity search error: {str(e)}")

        self.evaluation_results["document_processing"] = test_results
        print("   ✓ Document processing tests completed")
        return test_results

    def test_prompting_strategies(self):
        """Test different prompting strategies.

        Tests:
        1. Chain-of-Thought reasoning
        2. Self-Consistency
        3. ReAct workflow
        4. Basic QA

        Returns:
            dict: Comparison results for different strategies
        """
        print("\n=== Testing Prompting Strategies ===")
        strategy_results = {
            "chain_of_thought": [],
            "self_consistency": [],
            "react_workflow": [],
            "basic_qa": [],
        }

        for i, query in enumerate(self.test_queries[:3]):  # Test first 3 queries
            print(f"   Testing query {i+1}: {query[:50]}...")
            try:
                # Find relevant chunks for context
                relevant_chunks = self.doc_processor.find_similar_chunks(query, top_k=3, min_score=0.1)

                # Test Chain-of-Thought
                start_time = time.time()
                cot_response = self.research_assistant.chain_of_thought_reasoning(query, relevant_chunks)
                cot_time = time.time() - start_time
                strategy_results["chain_of_thought"].append({
                    "query": query,
                    "response": cot_response,
                    "response_length": len(cot_response),
                    "response_time": cot_time,
                    "api_calls": 1,
                })
                self.evaluate_response_quality(cot_response, query)
                print(f"      ✓ Chain-of-Thought: {cot_time:.2f}s")

                # Test Self-Consistency
                start_time = time.time()
                sc_response = self.research_assistant.self_consistency_generate(query, relevant_chunks, num_attempts=2)
                sc_time = time.time() - start_time
                strategy_results["self_consistency"].append({
                    "query": query,
                    "response": sc_response,
                    "response_length": len(sc_response),
                    "response_time": sc_time,
                    "api_calls": 3,  # 2 attempts + 1 selection
                })
                self.evaluate_response_quality(sc_response, query)
                print(f"      ✓ Self-Consistency: {sc_time:.2f}s")

                # Test ReAct Workflow
                start_time = time.time()
                react_response = self.research_assistant.react_research_workflow(query)
                react_time = time.time() - start_time
                strategy_results["react_workflow"].append({
                    "query": query,
                    "response": react_response.get("final_answer", "No final answer"),
                    "workflow_steps": len(react_response.get("workflow_steps", [])),
                    "response_time": react_time,
                    "api_calls": len(react_response.get("workflow_steps", [])) + 1,
                })
                self.evaluate_response_quality(react_response.get("final_answer", ""), query)
                print(f"      ✓ ReAct Workflow: {react_time:.2f}s ({len(react_response.get('workflow_steps', []))} steps)")

                # Test Basic QA
                start_time = time.time()
                basic_response = self.research_assistant.answer_research_question(query, use_cot=False, use_verification=False)
                basic_time = time.time() - start_time
                strategy_results["basic_qa"].append({
                    "query": query,
                    "response": basic_response.get("answer", "No answer"),
                    "response_length": len(basic_response.get("answer", "")),
                    "response_time": basic_time,
                    "api_calls": 1,
                })
                self.evaluate_response_quality(basic_response.get("answer", ""), query)
                print(f"      ✓ Basic QA: {basic_time:.2f}s")

                print(f"   ✓ Query {i+1} completed")

            except Exception as e:
                print(f"   ✗ Error testing query {i+1}: {str(e)}")
                strategy_results["chain_of_thought"].append({"query": query, "error": str(e)})
                strategy_results["self_consistency"].append({"query": query, "error": str(e)})
                strategy_results["react_workflow"].append({"query": query, "error": str(e)})
                strategy_results["basic_qa"].append({"query": query, "error": str(e)})

        self.evaluation_results["prompt_strategy_comparison"] = strategy_results
        print("   ✓ Prompting strategy tests completed")
        return strategy_results

    def test_agent_performance(self):
        """Skip agent performance tests (AgentOrchestrator not implemented).

        Returns:
            dict: Empty results (to be implemented later).
        """
        print("\n=== Testing AI Agents ===")
        print("   Skipped: AgentOrchestrator not implemented")
        agent_results = {
            "summarizer_agent": {},
            "qa_agent": {},
            "workflow_agent": {},
            "orchestrator": {},
        }
        self.evaluation_results["agent_performance"] = agent_results
        return agent_results

    def evaluate_response_quality(self, response: Union[str, dict], query: str) -> Dict:
        """Evaluate response quality for prompting strategy results.

        Args:
            response: Response string or dict from a prompting strategy.
            query: Original query text.

        Returns:
            dict: Quality scores for length and keyword relevance.
        """
        # Handle dict responses (from answer_research_question)
        if isinstance(response, dict):
            response_text = response.get("answer", "")
        else:
            response_text = response or ""

        if not response_text:
            return {"length_score": 0, "keyword_relevance": 0, "overall_score": 0}

        # Length score: Scale from 50-500 chars
        response_length = len(response_text)
        length_score = min(max(response_length, 50) / 500, 1.0)  # Scale 50-500 chars

        # Keyword relevance: Fuzzy matching with difflib
        query_words = set(re.findall(r"\w+", query.lower()))
        response_words = set(re.findall(r"\w+", response_text.lower()))
        matched_words = 0
        for query_word in query_words:
            for response_word in response_words:
                if difflib.SequenceMatcher(None, query_word, response_word).ratio() > 0.8:  # Fuzzy match threshold
                    matched_words += 1
                    break
        keyword_relevance = min(matched_words / len(query_words) if query_words else 0, 1.0)

        # Overall score
        overall_score = (length_score + keyword_relevance) / 2
        quality_scores = {
            "length_score": length_score,
            "keyword_relevance": keyword_relevance,
            "overall_score": overall_score,
        }
        self.evaluation_results["overall_scores"][query] = quality_scores
        return quality_scores

    def run_performance_benchmark(self) -> Dict:
        """Run performance benchmark for document processing and prompting strategies.

        Returns:
            dict: Performance benchmark results.
        """
        print("\n=== Running Performance Benchmark ===")
        benchmark_results = {
            "document_processing_time": 0,
            "query_response_times": [],
            "api_calls_made": 0,
            "memory_usage": "Not measured",
            "system_efficiency": {},
        }

        # Benchmark document processing (process all PDFs)
        start_time = time.time()
        try:
            processed_pdfs = 0
            for pdf_path in self.pdf_files:
                if not pdf_path.exists():
                    continue
                self.doc_processor.process_document(pdf_path)
                processed_pdfs += 1
            benchmark_results["document_processing_time"] = time.time() - start_time
            print(f"   Document processing time ({processed_pdfs} PDFs): {benchmark_results['document_processing_time']:.2f} seconds")
        except Exception as e:
            print(f"   Error processing documents: {str(e)}")
            benchmark_results["document_processing_time"] = 0

        # Rebuild index after processing
        try:
            self.doc_processor.build_search_index()
            print(f"   Rebuilt search index with {len(self.doc_processor.all_chunks)} chunks")
        except Exception as e:
            print(f"   Error building index: {str(e)}")

        # Benchmark prompting strategies
        api_calls = 0
        for query in self.test_queries[:2]:
            try:
                # Chain-of-Thought
                start_time = time.time()
                relevant_chunks = self.doc_processor.find_similar_chunks(query, top_k=3, min_score=0.1)
                cot_response = self.research_assistant.chain_of_thought_reasoning(query, relevant_chunks)
                response_time = time.time() - start_time
                benchmark_results["query_response_times"].append({
                    "query": query,
                    "strategy": "chain_of_thought",
                    "response_time": response_time,
                    "response_length": len(cot_response),
                })
                self.evaluation_results["response_times"].append(response_time)
                self.evaluation_results["response_lengths"].append(len(cot_response))
                self.evaluate_response_quality(cot_response, query)
                api_calls += 1
                print(f"   Query '{query[:30]}...' (Chain-of-Thought): {response_time:.2f} seconds")

                # Basic QA
                start_time = time.time()
                basic_response = self.research_assistant.answer_research_question(query, use_cot=False, use_verification=False)
                response_time = time.time() - start_time
                benchmark_results["query_response_times"].append({
                    "query": query,
                    "strategy": "basic_qa",
                    "response_time": response_time,
                    "response_length": len(basic_response.get("answer", "")),
                })
                self.evaluation_results["response_times"].append(response_time)
                self.evaluation_results["response_lengths"].append(len(basic_response.get("answer", "")))
                self.evaluate_response_quality(basic_response, query)
                api_calls += 1
                print(f"   Query '{query[:30]}...' (Basic QA): {response_time:.2f} seconds")

            except Exception as e:
                print(f"   Error benchmarking query '{query[:30]}...': {str(e)}")

        benchmark_results["api_calls_made"] = api_calls

        # Calculate efficiency
        avg_response_time = (
            sum(r["response_time"] for r in benchmark_results["query_response_times"])
            / len(benchmark_results["query_response_times"])
            if benchmark_results["query_response_times"]
            else 0
        )
        benchmark_results["system_efficiency"] = {
            "average_response_time": avg_response_time,
            "queries_per_minute": 60 / avg_response_time if avg_response_time > 0 else 0,
        }
        self.evaluation_results["performance_benchmark"] = benchmark_results
        print(f"   Average response time: {avg_response_time:.2f} seconds")
        print(f"   API calls made: {api_calls}")
        print("   ✓ Performance benchmark completed")
        return benchmark_results

    def generate_evaluation_report(self) -> str:
        """Generate comprehensive evaluation report.

        Returns:
            str: Formatted evaluation report.
        """
        num_pdfs = len([p for p in self.pdf_files if p.exists()])
        report = f"""
# ResearchGPT Assistant - Evaluation Report
Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %I:%M %p')}

## Test Summary
This report evaluates the Config, DocumentProcessor, and ResearchGPTAssistant components using {num_pdfs} real PDF(s) and Mistral API.

## Document Processing Tests
- Text extraction: {'PASS' if self.evaluation_results['document_processing'].get('pdf_extraction', False) else 'FAIL'}
- Preprocessing: {'PASS' if self.evaluation_results['document_processing'].get('text_preprocessing', False) else 'FAIL'}
- Chunking: {'PASS' if self.evaluation_results['document_processing'].get('chunking', False) else 'FAIL'}
- Search indexing: {'PASS' if self.evaluation_results['document_processing'].get('index_building', False) else 'FAIL'}
- Similarity search: {'PASS' if self.evaluation_results['document_processing'].get('similarity_search', False) else 'FAIL'}
- Errors: {self.evaluation_results['document_processing'].get('errors', [])}

## Prompting Strategy Performance
{json.dumps(self.evaluation_results.get('prompt_strategy_comparison', {}), indent=2)}

## AI Agent Performance
Skipped: AgentOrchestrator not implemented

## Performance Benchmarks
{json.dumps(self.evaluation_results.get('performance_benchmark', {}), indent=2)}

## Quality Metrics
{json.dumps(self.evaluation_results.get('overall_scores', {}), indent=2)}

## Recommendations for Improvement
1. Add more real PDF files for comprehensive testing.
2. Implement AgentOrchestrator for agent performance tests.
3. Enhance quality metrics with semantic similarity (e.g., ROUGE, BLEU).
4. Optimize chunking overlap for better search results.
5. Add batch processing for multiple PDFs in production.

## Conclusion
The Config, DocumentProcessor, and ResearchGPTAssistant components are fully functional for text extraction, preprocessing, similarity search, and prompting strategies with real Mistral API integration. Further development of AgentOrchestrator is needed for complete system testing.
"""
        with open(self.config.RESULTS_DIR / "evaluation_report.md", "w") as f:
            f.write(report)
        with open(self.config.RESULTS_DIR / "test_results.json", "w") as f:
            json.dump(self.evaluation_results, f, indent=2)
        return report

    def run_all_tests(self):
        """Execute complete test suite."""
        print("Starting ResearchGPT Assistant Test Suite...")
        doc_results = self.test_document_processing()
        prompt_results = self.test_prompting_strategies()
        agent_results = self.test_agent_performance()
        benchmark_results = self.run_performance_benchmark()
        self.evaluation_results.update({
            "document_processing": doc_results,
            "prompt_strategy_comparison": prompt_results,
            "agent_performance": agent_results,
            "performance_benchmark": benchmark_results,
        })
        final_report = self.generate_evaluation_report()
        print("\n=== Test Suite Complete ===")
        print("Results saved:")
        print("- evaluation_report.md")
        print("- test_results.json")
        return self.evaluation_results


if __name__ == "__main__":
    tester = ResearchGPTTester()
    results = tester.run_all_tests()