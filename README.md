
# 🌟 EmpowerHer - Women's Technology Platform  
### 🚀 RAGA_EVAL: Evaluating LLM Responses with Precision

<div align="center">
  <img src="https://img.shields.io/badge/version-1.0.0-blue.svg" />
  <img src="https://img.shields.io/badge/node-18.x-green.svg" />
  <img src="https://img.shields.io/badge/react-19.0.0-blue.svg" />
  <img src="https://img.shields.io/badge/mongodb-atlas-green.svg" />
  <img src="https://img.shields.io/badge/license-MIT-yellow.svg" />
</div>

<p align="center">
  <img width="180" src="https://i.imgur.com/pSOxq3J.png" alt="EmpowerHer Logo" />
</p>

<p align="center"><strong>A platform empowering women in tech through education, mentorship, and community support.</strong></p>

---

## 📋 Project Overview

**RAGA_EVAL** is a cutting-edge evaluation tool that enhances and assesses responses from Large Language Models (LLMs) using smart, iterative prompt optimization. By leveraging multiple evaluation dimensions, RAGA_EVAL provides deep insights into model behavior and output quality.

---

## ✨ Key Features

- 🔄 **Iterative Prompt Optimization** – Improve prompt quality through 3 automatic feedback loops  
- 📊 **Comprehensive Metrics** – Evaluate outputs with Relevance, Accuracy, BLEU, ROUGE, and more  
- 🚦 **Rate Limiting Handling** – Smart retry and backoff to manage API quotas  
- 📈 **Performance Visualization** – Track model improvements across iterations  
- 💡 **Ground Truth Generation** – Reference answers from Gemini for comparison  
- 📁 **Results Export** – Organized outputs in CSV format for easy access  

---

## 🧠 Models & Metrics

### 💬 **Models Evaluated**
- **Ground Truth:** Gemini  
- **Evaluation Models:** Mixtral, Llama, Qwen, DeepSeek (via Groq API)  

### 📏 **Evaluation Metrics**
| Metric         | Description                          |
|----------------|--------------------------------------|
| ✅ Relevance    | Response relevance to the query      |
| 🎯 Accuracy     | Factual correctness                 |
| 📚 Groundedness | Rooted in reference material         |
| 📖 Completeness | Information coverage                 |
| 🧠 Faithfulness | Logical consistency                  |
| 🧾 BLEU/ROUGE   | Text similarity & fluency checks     |

---

## 🗂 Folder Structure

```bash
📁 RAGA_EVAL/
├── interactive_eval.py       # Evaluation CLI interface
├── questions.json            # Evaluation input prompts
├── evaluation/               # Evaluation logic & metrics
│   ├── evaluator.py
│   ├── metrics.py
│   ├── prompt_improver.py
│   └── recommendation.py
├── models/                   # Model wrappers & config
│   ├── gemini_model.py
│   ├── groq_models.py
│   └── model_factory.py
├── RAGA_Eaval/               # RAG agent backend
│   └── RAGBasedAgent/
│       ├── main.py
│       ├── GithubAuth.py
│       ├── review_generator.py
│       └── similarity_query.py
├── responses/                # Raw model outputs
├── improved_responses/       # Responses after enhancement
├── prompts/                  # Updated prompts by iteration
└── results/                  # CSV summary of metrics
```

---

## 🛠️ Installation & Setup

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

### 📦 Install Dependencies
```bash
pip install -r requirements.txt
```

### 🔑 Configure Environment Variables

Create a `.env` file:

```env
GITHUB_ACCESS_TOKEN=your_github_token
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
```

---

## 🚀 How to Use

1. 📝 **Update `questions.json`** with your queries  
2. 🔧 **Run evaluation script**:
   ```bash
   python interactive_eval.py
   ```

3. 📋 **Choose from interactive menu**:
   - Evaluate single question
   - Iterative evaluation (3 rounds)
   - List available questions
   - Exit  

4. 📊 **Review output**:
   - Terminal comparison view
   - CSV exports (`results/`)
   - Enhanced prompts (`prompts/`)

---

## 📊 Sample Output

```
✅ Best model for PR review: llama (Score: 0.625)

📊 MODEL COMPARISON (Reference: GEMINI)
────────────────────────────────────────────────────────────
Model     | Overall | Relev | Accur | Grnd | Comp | BLEU | ROUGE
llama     | 0.625   | 0.74  | 0.85  | 0.80 | 0.20 | 0.18 | 0.20
deepseek  | 0.589   | 0.77  | 0.90  | 0.60 | 0.20 | 0.04 | 0.17
alibaba   | 0.485   | 0.80  | 0.80  | 0.80 | 0.20 | 0.11 | 0.19
```

---

## 🚧 Development Status

| Feature                                      | Status |
|---------------------------------------------|--------|
| Core Evaluation Engine                      | ✅     |
| Multi-Model Support                         | ✅     |
| Iterative Prompt Enhancement                | ✅     |
| Rate Limiting & Retry Logic                 | ✅     |
| CSV Export of Results                       | ✅     |
| RAG Agent Integration                       | 🔜     |

---

## 🔮 Roadmap

- 📦 Integration with RAG-Based Agents  
- 📉 Interactive Web Dashboard  
- 🧠 Support for Semantic Evaluation Metrics  
- ⚙️ Parallel Processing for Large-Scale Runs  

---

## 👥 Contributors

- 👩‍💻 **Tejaswini Durge**  
- 👩‍💻 **Dipali Gangarde**

---

## 📄 License

Licensed under the [MIT License](LICENSE).

🕒 *Last updated: March 16, 2025*
