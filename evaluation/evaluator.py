import asyncio
from .metrics import MetricsCalculator
from .prompt_improver import PromptImprover

class ResponseEvaluator:
    def __init__(self):
        self.metrics = MetricsCalculator()
        self.recommender = PromptImprover()
        self.weights = {
            "Relevance": 0.20,
            "Accuracy": 0.25,
            "Groundedness": 0.20,
            "Completeness": 0.15,
            "BLEU": 0.10,
            "ROUGE": 0.10
        }
    
    
    async def evaluate_responses(self, reference_model, responses, question, system_prompt):
        """
        Evaluate responses and generate system prompt improvements
        
        Args:
            reference_model: Model to use as reference
            responses: Dict of model responses
            question: The original question
            system_prompt: Current system prompt used
            
        Returns:
            dict: Evaluation results with scores and prompt improvement suggestions
        """
        reference = responses.get(reference_model)
        if not reference:
            raise ValueError(f"Reference model {reference_model} not found in responses")
        
        # Calculate scores for all models (except reference)
        model_scores = {}
        prompt_improvements = {}
        
        for model_name, response in responses.items():
            if model_name == reference_model:
                continue
                
            print(f"Evaluating {model_name}...")
            
            # Calculate all metrics (existing code)
            relevance = self.metrics.compute_relevance(reference, response)
            accuracy = self.metrics.compute_accuracy(response)
            groundedness = self.metrics.compute_groundedness(reference, response)
            completeness = self.metrics.compute_completeness(reference, response)
            bleu = await self.metrics.compute_bleu_ragas(reference, response)
            rouge = await self.metrics.compute_rouge_ragas(reference, response)
            
            scores = {
                "Relevance": round(relevance, 3),
                "Accuracy": round(accuracy, 3),
                "Groundedness": round(groundedness, 3),
                "Completeness": round(completeness, 3),
                "BLEU": round(bleu, 3),
                "ROUGE": round(rouge, 3)
            }
            
            weighted_score = sum(scores[metric] * weight 
                               for metric, weight in self.weights.items())
            scores["Overall"] = round(weighted_score, 3)
            
            model_scores[model_name] = scores
            
            # Generate prompt improvement suggestions
            print(f"Generating prompt improvements for {model_name}...")
            prompt_improvements[model_name] = self.recommender.get_prompt_improvements(
                question, system_prompt, response, reference, scores
            )
        
        # Sort models by overall score
        sorted_models = sorted(model_scores.items(), 
                             key=lambda x: x[1]["Overall"], 
                             reverse=True)
        
        return {
            "reference_model": reference_model,
            "reference_response": reference,
            "model_scores": model_scores,
            "rankings": sorted_models,
            "prompt_improvements": prompt_improvements
        }