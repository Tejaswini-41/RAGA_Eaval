
```markdown
# RAG-Based Agent for Pull Request Review

This project implements a **RAG-Based Agent** (Retrieval-Augmented Generation) to automate and enhance the process of reviewing pull requests (PRs) in software repositories. The system leverages advanced AI models to generate detailed, actionable feedback for PRs by analyzing code changes, comparing them with similar PRs, and evaluating the quality of the changes using custom metrics.

---

## Features

- **Pull Request Analysis**:
  - Extracts and compares changes in the current PR with similar PRs.
  - Identifies potential risks, conflicts, and areas for improvement.

- **RAG-Based Review Generation**:
  - Uses retrieval-augmented generation to provide context-aware reviews.
  - Supports multiple AI models (e.g., `gemini`, `llama`, `alibaba`, `deepseek`).

- **Enhanced Prompting**:
  - Dynamically generates enhanced prompts for better review quality.
  - Stores and compares baseline and enhanced reviews.

- **Confidence Scoring**:
  - Adds confidence scores to reviews based on similarity and complexity metrics.

- **Chunking Advice**:
  - Provides intelligent chunking strategies for large PRs to optimize processing.

---

## Installation

### 🔽 Clone Repository
```bash
git clone https://github.com/Tejaswini-41/RAGA_Eaval.git
cd RAGA_Eval
```

### 🧪 Set Up Virtual Environment
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file:
```env
GITHUB_ACCESS_TOKEN=your_github_token
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
``` 

---

## Usage

### Running the RAG-Based Review Process

1. Start the review process:
   ```bash
   cd RAGBasedAgent 
   python main.py
   ```

2. Follow the interactive menu to:
   - Perform an initial review.
   - Test stored prompts.
   - Add confidence scores.
   - Generate chunking advice.

### Key Commands

- **Initial Review**:
  - Fetches PR data, generates embeddings, and performs a baseline review.
- **Enhanced Review**:
  - Uses enhanced prompts to generate improved reviews.
- **Confidence Scoring**:
  - Adds confidence metrics to the review output.

---

## Project Structure

```plaintext
RAGBasedAgent/
├── main.py                     # Entry point for the application
├── review_generator.py         # Generates PR reviews
├── review_evaluator.py         # Evaluates review quality using metrics
├── embedding_store.py          # Handles embedding storage in ChromaDB
├── change_analyzer.py          # Compares changes in PRs
├── Confidence_Scorer.py        # Adds confidence scores to reviews
├── prompts/                    # Contains prompt templates and history
├── reviews/                    # Stores generated reviews and PR data
├── recommendations/            # Stores chunking advice and recommendations
└── requirements.txt            # Python dependencies
```

---

## Key Components

### 1. **Review Generator**
- Generates reviews using AI models.
- Supports fallback to rule-based generation if AI fails.

### 2. **Review Evaluator**
- Calculates metrics like relevance, accuracy, groundedness, and completeness.
- Compares baseline and enhanced reviews.

### 3. **Embedding Store**
- Stores PR file embeddings using TF-IDF and ChromaDB.
- Queries similar PRs for context.

### 4. **Confidence Scorer**
- Analyzes PR changes and assigns confidence scores.
- Highlights risks and overlaps with similar PRs.

### 5. **Chunking Advisor**
- Provides chunking strategies for large PRs.
- Ensures optimal processing and context preservation.

---

## Example Workflow

1. **Initial Review**:
   - Fetch PR data and generate a baseline review.
   - Save results for further analysis.

2. **Enhanced Review**:
   - Generate enhanced prompts and reviews.
   - Compare metrics with the baseline.

3. **Confidence Scoring**:
   - Add confidence scores to the review.
   - Highlight risks and overlaps.

4. **Chunking Advice**:
   - Generate chunking strategies for large PRs.

---

## Metrics

The system evaluates PR reviews using the following metrics:
- **Relevance**
- **Accuracy**
- **Groundedness**
- **Completeness**
- **Faithfulness**
- **Contextual Precision**
- **Answer Relevance**
- **BLEU/ROUGE Scores**
- **Overall Quality**

---

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description of your changes.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contact

For questions or support, please contact:
- **Name**: tejaswini, dipali
```
