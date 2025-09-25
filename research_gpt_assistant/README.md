# ResearchGPT Assistant

## Project Overview
ResearchGPT Assistant is an intelligent research tool that leverages advanced AI techniques to process academic documents, generate insights, and automate research workflows. This capstone project demonstrates the integration of machine learning (TF-IDF for similarity search), natural language processing (text extraction, chunking), advanced prompting strategies (Chain-of-Thought, Self-Consistency, ReAct), and AI agents (Summarizer, QA, Research Workflow, Orchestrator) using the Mistral API. It processes AI/ML research papers (e.g., neural networks, reinforcement learning), producing high-quality outputs (7.7-8.0/10) for summarization, question answering, and research analysis.

Key features:
- **Document Processing**: Extracts and chunks text from PDFs.
- **Similarity Search**: TF-IDF-based retrieval for queries like "machine learning algorithms."
- **Prompting**: CoT, Self-Consistency, and ReAct for robust reasoning.
- **Agents**: Summarizes papers (100-200 words), answers questions, and conducts research sessions.
- **Reporting**: Generates detailed reports (`demo_report.md`, `test_results.json`).
- **Robustness**: Handles API rate limits (429 errors), missing files, and edge cases.

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd research_gpt_assistant
   ```
2. **Install Dependencies**:
   ```bash
   pip install sklearn pandas pdfplumber python-dotenv nltk
   ```
   Ensure NLTK `punkt` tokenizer is downloaded:
   ```python
   import nltk
   nltk.download('punkt')
   ```
3. **Configure Environment**:
   - Create a `.env` file in the project root:
     ```bash
     MISTRAL_API_KEY=<your-mistral-api-key>
     MODEL_NAME=mistral-small-2506
     TEMPERATURE=0.1
     MAX_TOKENS=500
     CHUNK_SIZE=500
     MIN_CHUNK_SIZE=250
     OVERLAP=100
     MAX_FEATURES=1000
     ```
   - Obtain a Mistral API key from [Mistral AI API](https://mistral.ai).
4. **Prepare PDFs**:
   - Place up to 10 AI/ML research papers (e.g., `2509.09655v1.pdf`, `nn_sample_1.pdf`) in `results/sample_papers/`.
   - Ensure `results/` exists (`mkdir results`).
5. **Verify Files**:
   - Ensure `config.py`, `document_processor.py`, `research_assistant.py`, `research_agents.py`, `unit_tests.py`, and `main.py` are in the project root.

## Usage Instructions
1. **Run Tests**:
   ```bash
   python unit_tests.py
   ```
   - **Expected Output**: 7/7 tests pass, generating `results/evaluation_report.md` and `results/test_results.json`.
   - **Runtime**: ~160-200s.
   - **Quality**: Scores 7.7-8.0/10 for agent outputs.
   - **Files**:
     - `evaluation_report.md`: Test summary, document processing, prompting, and agent performance.
     - `test_results.json`: Detailed metrics (response times, quality scores).

2. **Run Demo**:
   ```bash
   python main.py
   ```
   - **Expected Output**:
     ```
     2025-09-24 18:34:24,972 - MainDemo - INFO - === ResearchGPT Assistant Demo ===
     ...
     2025-09-24 18:36:11,297 - MainDemo - INFO - 📊 Documents processed: {'num_documents': 11, 'total_chunks': 3592, ...}
     ...
     2025-09-24 18:37:35,729 - MainDemo - INFO - ✓ Saved: demo_report.md
     2025-09-24 18:37:35,730 - MainDemo - INFO - === Demo Complete ===
     ```
   - **Runtime**: ~190-200s (86s for 8 PDFs, ~105s for API calls).
   - **Files** (in `results/`):
     - `cot_response.json`: Chain-of-Thought reasoning (~2676 chars).
     - `self_consistency_response.txt`: Self-Consistency prompting (~1154 chars).
     - `react_workflow.json`: ReAct workflow (5 steps).
     - `document_summary.json`: Summary for `2509.09655v1` (200 words).
     - `qa_response.json`: QA response (200 words).
     - `research_workflow.json`: Research session with 3 questions.
     - `verification_result.json`: Verification (confidence ~7.3/10).
     - `demo_report.md`: Demo summary with stats (11 docs, 3592 chunks).

## Sample Outputs
- **Similarity Search** (query: "machine learning algorithms"):
  ```
  [('tation applies also to traditional universality theorems for models such as (cid 37)oolean circuits...', 0.337, 'nn_sample_2'),
   ('ble the biggest breakthrough in machine learning won(cid 10)t be any single conceptual breakthrough...', 0.334, 'nn_sample_2'),
   ('l network does when trained using (cid 24),(cid 19)(cid 19)(cid 19) images (cid 11)(cid 28)(cid 22)...', 0.288, 'nn_sample_2')]
  ```
- **Chain-of-Thought**: Detailed reasoning for “What are the main advantages and limitations of deep learning?” (2676 chars).
- **Research Workflow**: 3 questions answered on “artificial intelligence applications” (e.g., reinforcement learning, federated learning), leveraging 8 AI/ML PDFs.
- **Demo Report** (excerpt from `demo_report.md`):
  ```
  # ResearchGPT Assistant - Demonstration Report
  ## Documents Processed
  - Total Documents: 11
  - Total Chunks: 3592
  - Average Document Length: 139490.73 chars
  - Average Chunk Length: 522.19 chars
  ```

## Project Structure
```
research_gpt_assistant/
├── README.md
├── requirements.txt
├── .env                             # Mistral API key
├── config.py                        # Configuration and API settings
├── document_processor.py            # PDF processing, chunking, TF-IDF
├── research_assistant.py            # Prompting and verification logic
├── research_agents.py               # Agent implementations
├── main.py                          # Demo script
├── test_system.py                   # Testing and evaluation script
├── data/
│   ├── sample_papers/               # AI/ML PDF research papers
│   └── processed/                   # Extracted text files
├── logs/                            # Logging
├── results/
│   ├── summaries/                   # Generated document summaries
│   ├── analyses/                    # Research analyses and insights
│   ├── cot_response.json            # Chain-of-Thought output
│   ├── self_consistency_response.txt  # Self-Consistency output
│   ├── react_workflow.json          # ReAct workflow output
│   ├── document_summary.json        # Summarizer output
│   ├── qa_response.json             # QA output
│   ├── research_workflow.json       # Research Workflow output
│   ├── verification_result.json     # Verification output
│   ├── demo_report.md               # Demo report
│   ├── evaluation_report.md         # Test report
│   └── test_results.json            # Test metrics
└── prompts/
    └── prompt_templates.txt         # All prompt templates used
```

## Acknowledgments
- **Mistral API**: Powers language generation ([Mistral API](https://mistral.ai)).
- **Dependencies**: `sklearn`, `pandas`, `pdfplumber`, `python-dotenv`, `nltk`.
- **Course**: This capstone project integrates ML, NLP, and AI agent concepts from the course.
- **Source**: Project forked from [here](https://github.com/RamaKattunga/ResearchGPT). 