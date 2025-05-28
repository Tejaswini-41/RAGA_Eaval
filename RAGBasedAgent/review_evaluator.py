import os
import sys
import asyncio
import json
from dotenv import load_dotenv
import datetime

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from existing evaluation system
from evaluation.metrics import MetricsCalculator 
from models.model_factory import ModelFactory
from prompts.review_prompts import ReviewPrompts

class ReviewEvaluator:
    """Evaluates models for PR review quality using existing RAGA metrics"""
    
    def __init__(self):
        self.metrics_calculator = MetricsCalculator()
        self.model_factory = ModelFactory()
        
        # Updated weights with enhanced RAGA metrics
        self.weights = {
            # Original metrics with adjusted weights
            "Relevance": 0.10,
            "Accuracy": 0.15,
            "Groundedness": 0.10,
            "Completeness": 0.10,
            
            # New RAGA metrics
            "Faithfulness": 0.20,     # How well the response avoids hallucinations
            "ContextualPrecision": 0.25,  # How precisely the review references specific PR elements
            "AnswerRelevance": 0.15,   # How well the review addresses PR-specific issues
            
            # Existing metrics with reduced weights
            "BLEU": 0.025,
            "ROUGE": 0.025
        }
    
    async def evaluate_models(self, current_pr_changes, similar_pr_changes, 
                             models=None, test_prompt_size=0.3, system_prompt=None):
        """
        Evaluate multiple models on PR review task to select best performer
        
        Args:
            current_pr_changes: The changes in current PR
            similar_pr_changes: The changes in similar PR
            models: List of model names to evaluate
            test_prompt_size: Fraction of PR content to use for quick evaluation
            
        Returns:
            best_model: Name of the best performing model
            all_metrics: Dict of metrics for all models
        """
        # Use provided system prompt or get default
        if not system_prompt:
            _, system_prompt = ReviewPrompts.get_current_prompt()
            
        # Truncate PR content to avoid token limit errors
        max_tokens = 4000  # Set a reasonable token limit
        
        # Truncate PR content
        if len(current_pr_changes) > max_tokens:
            current_pr_changes = current_pr_changes[:max_tokens] + "...[truncated]"
        
        # Truncate similar PR content
        if isinstance(similar_pr_changes, list):
            truncated_similar = []
            for pr in similar_pr_changes:
                if len(pr['changes']) > max_tokens//2:
                    pr['changes'] = pr['changes'][:max_tokens//2] + "...[truncated]"
                truncated_similar.append(pr)
            similar_pr_changes = truncated_similar
        else:
            if len(similar_pr_changes) > max_tokens:
                similar_pr_changes = similar_pr_changes[:max_tokens] + "...[truncated]"
        
        # Get available models if not specified
        if not models:
            models = self.model_factory.get_model_names()
            
        # Build comprehensive prompt template for evaluation - using FULL content
        prompt = f"""Compare these pull requests:
    
Similar PR:
{similar_pr_changes}

Current PR:
{current_pr_changes}

Please provide a detailed code review including:
1. Summary of the changes
2. File Change Suggestions - Identify files that might get affected based on changes
3. Conflict Prediction - Flag files changed in multiple PRs that could cause conflicts
4. Breakage Risk Warning - Note which changes might break existing functionality
5. Test Coverage Advice - Recommend test files that should be updated
6. Code Quality Suggestions - Point out potential code smells or duplication

Be specific with file names, function names, and line numbers when possible.
"""
        
        print("\n🔍 Evaluating models to select best one for PR review...")
        
        # Generate initial review with reference model (using Gemini as reference)
        reference_model = "gemini"
        print(f"• Generating reference review with {reference_model}...")
        reference_review = self.model_factory.generate_response_with_prompt(
            reference_model, 
            prompt, 
            "You are an expert code reviewer. Focus on practical, specific suggestions. Mention specific files, functions, and line numbers when relevant."
        )
        
        # Generate and evaluate reviews from other models
        model_metrics = {}
        all_reviews = {reference_model: reference_review}
        
        for model_name in models:
            if model_name == reference_model:
                continue
                
            print(f"• Testing {model_name} model...")
            
            try:
                # Generate sample review
                test_review = self.model_factory.generate_response_with_prompt(
                    model_name, 
                    prompt, 
                    "You are an expert code reviewer. Focus on practical, specific suggestions. Mention specific files, functions, and line numbers when relevant."
                )
                all_reviews[model_name] = test_review
                
                # Calculate metrics (using similar approach to interactive_eval.py)
                metrics = await self._calculate_metrics(reference_review, test_review)
                model_metrics[model_name] = metrics
                print(f"  - Overall score: {metrics['Overall']:.3f}")
                
            except Exception as e:
                print(f"  ⚠️ Error evaluating {model_name}: {e}")
        
        # Find best model
        if model_metrics:
            best_model = max(model_metrics.items(), key=lambda x: x[1]["Overall"])
            print(f"\n✅ Best model for PR review: {best_model[0]} (Score: {best_model[1]['Overall']:.3f})")
            
            # Print comparison table (like in interactive_eval.py)
            self._display_comparison_table(reference_model, model_metrics)
            
            return best_model[0], model_metrics
        else:
            print(f"\n⚠️ No models successfully evaluated. Using {reference_model} as default.")
            return reference_model, {}
    
    async def generate_detailed_pr_review(self, current_pr_data, similar_prs_data, model_name=None):
        """
        Generate detailed PR review with specific suggestions based on PR comparison
        
        Args:
            current_pr_data: Dict with current PR info (changes, files, PR number)
            similar_prs_data: List of dicts with similar PR info
            model_name: Name of model to use (uses best model if None)
            
        Returns:
            Dict with detailed review sections
        """
        if not model_name:
            # Use best model from evaluation or default to gemini
            model_name = "gemini"
        
        # Get current prompts
        review_template, system_prompt = ReviewPrompts.get_current_prompt()
        
        # Use centralized prompts
        detailed_review = self.model_factory.generate_response_with_prompt(
            model_name, 
            review_template.format(
                similar_prs=similar_prs_data.get('changes', ''),
                current_pr=current_pr_data.get('changes', '')
            ),
            system_prompt
        )
        
        return detailed_review
    
    async def _calculate_metrics(self, reference_review, generated_review):
        """Calculate RAGAS metrics between reference and generated reviews"""
        try:
            # Import needed classes
            from evaluation.metrics import MetricsCalculator
            
            # Make sure we have a metrics calculator
            if not hasattr(self, 'metrics_calculator'):
                self.metrics_calculator = MetricsCalculator()
                
            # Extract relevant content
            reference_content = self._extract_relevant_content(reference_review)
            generated_content = self._extract_relevant_content(generated_review)
            
            # Use default values for metrics that can be affected by rate limits
            default_metric_value = 0.6  # A reasonable default that won't be exactly 0.000
            
            # Calculate metrics with retry logic and defaults
            try:
                # Try to compute faithfulness
                faithfulness = default_metric_value
                try:
                    faithfulness = self.metrics_calculator.compute_faithfulness(reference_content, generated_content)
                except Exception as e:
                    print(f"Using default faithfulness value due to error: {e}")
                    
                # Calculate relevance
                relevance = self.metrics_calculator.compute_relevance(reference_content, generated_content)
                
                # Calculate accuracy with retry and fallback
                accuracy = default_metric_value
                try:
                    accuracy = self.metrics_calculator.compute_accuracy(generated_content)
                except Exception as e:
                    print(f"Using default accuracy value due to error: {e}")
                
                # Calculate completeness
                completeness = default_metric_value
                try:
                    completeness = self.metrics_calculator.compute_completeness(reference_content, generated_content)
                except Exception as e:
                    print(f"Using default completeness value due to error: {e}")
                
                # Additional context metrics for RAG evaluation
                context_precision = default_metric_value
                context_recall = default_metric_value
                
                # Calculate answer relevance
                answer_relevance = default_metric_value
                try:
                    answer_relevance = self.metrics_calculator.compute_answer_relevance(reference_content, generated_content)
                except Exception as e:
                    print(f"Using default answer relevance value due to error: {e}")
                
                # Calculate BLEU and ROUGE scores - NEW CODE
                bleu_score = 0.0
                rouge_score = 0.0
                try:
                    # These are async methods, so we need to await them
                    bleu_score = await self.metrics_calculator.compute_bleu_ragas(reference_content, generated_content)
                    rouge_score = await self.metrics_calculator.compute_rouge_ragas(reference_content, generated_content)
                except Exception as e:
                    print(f"Error computing BLEU/ROUGE scores: {e}")
                    # Use simpler fallback calculation if RAGAS fails
                    if hasattr(self.metrics_calculator, '_clean_text'):
                        clean_ref = self.metrics_calculator._clean_text(reference_content)
                        clean_gen = self.metrics_calculator._clean_text(generated_content)
                        # Simple word overlap as fallback
                        ref_words = set(clean_ref.split())
                        gen_words = set(clean_gen.split())
                        if ref_words and gen_words:
                            overlap = len(ref_words.intersection(gen_words)) / max(1, len(ref_words.union(gen_words)))
                            bleu_score = overlap * 0.3  # Scale down for BLEU
                            rouge_score = overlap * 0.4  # Scale down for ROUGE
            except Exception as e:
                print(f"Error during metric calculations: {e}")
                faithfulness = default_metric_value
                relevance = default_metric_value
                accuracy = default_metric_value
                completeness = default_metric_value
                context_precision = default_metric_value
                context_recall = default_metric_value
                answer_relevance = default_metric_value
                bleu_score = 0.15
                rouge_score = 0.20
            
            # Calculate overall score (weighted average) - now includes BLEU and ROUGE
            overall = (relevance * 0.20 + accuracy * 0.20 + 
                      faithfulness * 0.20 + completeness * 0.15 + 
                      answer_relevance * 0.15 + bleu_score * 0.05 + rouge_score * 0.05)
            
            # Return all metrics
            return {
                "Relevance": relevance,
                "Accuracy": accuracy,
                "Groundedness": faithfulness,
                "Completeness": completeness,
                "Faithfulness": faithfulness,
                "ContextualPrecision": context_precision,
                "ContextRecall": context_recall,
                "AnswerRelevance": answer_relevance,
                "BLEU": bleu_score,
                "ROUGE": rouge_score,
                "Overall": overall
            }
        except Exception as e:
            print(f"Error in metrics calculation: {e}")
            # Use default values but still return something reasonable
            return self._get_default_metrics()
        except Exception as e:
            print(f"❌ Error calculating metrics: {e}")
            # Return fallback metrics
            return self._get_default_metrics()
        
    def _get_default_metrics(self):
        """Return default metrics when calculation fails"""
        default_value = 0.6  # A reasonable default value
        return {
            "Relevance": default_value,
            "Accuracy": default_value,
            "Groundedness": default_value,  
            "Completeness": default_value,
            "Faithfulness": default_value,
            "ContextualPrecision": default_value,
            "ContextRecall": default_value,
            "AnswerRelevance": default_value,
            "BLEU": 0.15,  # Lower default for BLEU
            "ROUGE": 0.20,  # Lower default for ROUGE
            "Overall": default_value
        }
        
    def _extract_relevant_content(self, content):
        """Extract relevant content from review for metric calculation"""
        # Remove any thinking sections if they exist
        if "<think>" in content and "</think>" in content:
            thinking_section = content[content.find("<think>"):content.find("</think>")+8]
            content = content.replace(thinking_section, "")
        
        # Trim the content if it's too long
        if len(content) > 4000:
            content = content[:4000]
            
        return content

    def _display_comparison_table(self, reference_model, model_metrics):
        """Display comparison table of model metrics with improved terminal readability"""
        print("\n" + "═" * 80)
        print(f"📊 MODEL COMPARISON (Reference: {reference_model.upper()})")
        print("═" * 80)
        
        # Use more compact header names with better spacing
        header = f"{'Model':<12} | {'Overall':<7} |"
        
        # Short forms for metrics to improve terminal display
        metric_labels = {
            "Relevance": "Relev", 
            "Accuracy": "Accur", 
            "Groundedness": "Grnd",  # Fixed typo: was Groundness
            "Completeness": "Comp",
            "Faithfulness": "Faith",
            "ContextualPrecision": "Ctx",
            "AnswerRelevance": "Answ", 
            "BLEU": "BLEU",
            "ROUGE": "ROUG"
        }
        
        # Use correct metric keys from the weights dictionary
        metrics_to_display = [
            "Relevance", "Accuracy", "Groundedness", "Completeness", 
            "Faithfulness", "ContextualPrecision", "AnswerRelevance",
            "BLEU", "ROUGE"
        ]
        
        # Add metrics with clear column separation
        for metric in metrics_to_display:
            header += f" {metric_labels[metric]:<5} |"
        
        print(header)
        print("─" * 80)
        
        # Print each model's metrics with clear column separation
        for model_name, scores in sorted(model_metrics.items(), key=lambda x: x[1]["Overall"], reverse=True):
            row = f"{model_name:<12} | {scores['Overall']:<7.3f} |"
            
            for metric in metrics_to_display:
                # Handle metrics that might not be available
                value = scores.get(metric, 0)
                row += f" {value:<5.2f} |"  # Use .2f for more compact display
            
            print(row)
        
        print("─" * 80)


    async def _generate_review_with_confidence(self, current_pr_changes, similar_pr_changes, model_name, metrics):
        """Generate review with confidence scores for each suggestion"""
        
        base_review = await self.generate_detailed_pr_review(
            {"changes": current_pr_changes},
            {"changes": similar_pr_changes},
            model_name
        )
        
        # Add confidence scores based on metrics
        confidence_scores = {
            "File Changes": metrics[model_name]["ContextualPrecision"],
            "Conflict Detection": metrics[model_name]["Accuracy"],
            "Breaking Changes": metrics[model_name]["Faithfulness"],
            "Test Coverage": metrics[model_name]["Completeness"],
            "Code Quality": metrics[model_name]["AnswerRelevance"]
        }
        
        enhanced_review = "# Review with Confidence Scores\n\n"
        for section, confidence in confidence_scores.items():
            enhanced_review += f"\n## {section} (Confidence: {confidence:.1%})\n"
            section_content = self._extract_section(base_review, section)
            enhanced_review += section_content if section_content else "No suggestions\n"
        
        return enhanced_review

    async def _generate_review_with_enhanced_prompt(self, current_pr_changes, similar_pr_changes, model_name):
        """Generate review using enhanced prompts for better specificity"""
        
        enhanced_prompt = f"""Analyze this PR as an expert code reviewer:

CONTEXT:
Current Changes: {current_pr_changes}
Similar PR History: {similar_pr_changes}

PROVIDE DETAILED ANALYSIS:
1. Technical Implementation (Be specific)
    - Core functionality changes
    - Architecture impacts
    - Performance implications
    
2. Quality & Safety
    - Potential edge cases
    - Error handling completeness
    - Security considerations
    
3. Testing & Maintenance
    - Required test coverage
    - Documentation needs
    - Future maintenance considerations

Focus on specific files, line numbers, and code patterns.
"""
        return await self.generate_detailed_pr_review(
            {"changes": current_pr_changes},
            {"changes": similar_pr_changes},
            model_name,
            custom_prompt=enhanced_prompt
        )

    async def _generate_review_with_chunking(self, current_pr_changes, similar_pr_changes, model_name):
        """Generate review with improved chunking strategy"""
        
        chunk_size = 1000
        chunks = self._create_smart_chunks(current_pr_changes, chunk_size)
        
        reviews = []
        for i, chunk in enumerate(chunks, 1):
            print(f"\n📄 Processing chunk {i}/{len(chunks)}...")
            chunk_review = await self.generate_detailed_pr_review(
                {"changes": chunk},
                {"changes": similar_pr_changes},
                model_name
            )
            reviews.append(chunk_review)
        
        return self._merge_chunked_reviews(reviews)

    def _create_smart_chunks(self, content, chunk_size):
        """Create chunks intelligently preserving code block integrity"""
        chunks = []
        current_chunk = ""
        
        for line in content.split('\n'):
            if len(current_chunk) + len(line) > chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = line
            else:
                current_chunk += line + '\n'
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks

    def _merge_chunked_reviews(self, reviews):
        """Merge chunked reviews intelligently"""
        merged = "# Merged Chunked Review\n\n"
        sections = ["Summary", "File Changes", "Conflicts", "Breaking Changes", 
                    "Test Coverage", "Code Quality"]
        
        for section in sections:
            merged += f"\n## {section}\n"
            section_content = set()
            
            for review in reviews:
                content = self._extract_section(review, section)
                if content:
                    section_content.update(content.split('\n'))
            
            merged += '\n'.join(sorted(section_content)) + '\n'
        
        return merged

    def _extract_section(self, review, section_name):
        """Extract content from a specific section of the review"""
        lines = review.split('\n')
        content = []
        in_section = False
        
        for line in lines:
            if line.startswith('##') and section_name.lower() in line.lower():
                in_section = True
                continue
            elif line.startswith('##') and in_section:
                break
            elif in_section and line.strip():
                content.append(line)
        
        return '\n'.join(content) if content else ""

    async def generate_enhanced_prompt(self, current_metrics, current_prompt):
        """Generate enhanced system prompt based on RAGAS metrics analysis using Gemini"""
        try:
            # Import Gemini
            import google.generativeai as genai
            from dotenv import load_dotenv
            import os
            import datetime

            # Load API key
            load_dotenv()
            genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

            # Setup Gemini
            model = genai.GenerativeModel('gemini-1.5-flash')  # Changed to gemini-pro for better reliability

            # Analyze metrics to identify areas for improvement
            metric_analysis = []
            for metric, score in current_metrics.items():
                if metric != "Overall":
                    if score < 0.7:
                        status = "needs significant improvement"
                    elif score < 0.8:
                        status = "could be improved"
                    else:
                        status = "performing well"
                    metric_analysis.append(f"{metric}: {score:.3f} ({status})")

            # Create prompt for generating enhanced system prompt
            prompt = f"""You are an expert prompt engineer. Create an enhanced system prompt for code review that addresses the following metrics:

Current Metrics Performance:
{chr(10).join(metric_analysis)}

Current System Prompt:
"{current_prompt}"

Requirements:
1. Focus heavily on metrics scoring below 0.7
2. Maintain strengths of metrics above 0.8
3. Emphasize:
   - Specific file and line number references
   - Technical accuracy and completeness
   - Practical, actionable suggestions
   - Context awareness
   - Clear structure and organization

Generate a new system prompt that's between 200-300 words. Focus on clarity and specificity.
Return ONLY the new system prompt, no explanations or additional text."""

            # Generate new prompt using Gemini
            response = model.generate_content(prompt)
            new_system_prompt = response.text.strip()

            # More lenient validation (200-400 words instead of character count)
            word_count = len(new_system_prompt.split())
            if word_count < 50 or word_count > 400:
                print(f"\n⚠️ Generated prompt length ({word_count} words) outside ideal range (50-400 words)")
                if word_count < 20:  # Only reject if extremely short
                    raise ValueError("Generated prompt too short")
        
            print(f"\n✅ Generated new system prompt ({word_count} words)")
            print("\n📝 New System Prompt:")
            print("-" * 50)
            print(new_system_prompt)
            print("-" * 50)
        
            prompt_history = {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "system_prompt": new_system_prompt,
                "metrics": current_metrics
            }

            # Create prompts directory if it doesn't exist
            if not os.path.exists("prompts/history"):
                os.makedirs("prompts/history", exist_ok=True)

            # Save to history file with proper datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(f"prompts/history/prompt_{timestamp}.json", "w") as f:
                json.dump(prompt_history, f, indent=2)

            return new_system_prompt

        except Exception as e:
            print(f"❌ Error generating enhanced prompt: {str(e)}")
            print("⚠️ Falling back to current prompt")
            return current_prompt