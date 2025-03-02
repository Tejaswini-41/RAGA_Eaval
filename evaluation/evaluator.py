import asyncio
from .metrics import MetricsCalculator

class ResponseEvaluator:
    def __init__(self):
        self.metrics = MetricsCalculator()
        self.weights = {
            "Relevance": 0.20,
            "Accuracy": 0.25,
            "Groundedness": 0.20,
            "Completeness": 0.15,
            "BLEU": 0.10,
            "ROUGE": 0.10
        }
    
    async def evaluate_responses(self, reference_model, responses):
        """
        Evaluate responses for a single question
        
        Args:
            reference_model (str): Model name to use as reference (e.g., "gemini")
            responses (dict): Dict of {model_name: response_text}
            
        Returns:
            dict: Evaluation results with scores and rankings
        """
        reference = responses.get(reference_model)
        if not reference:
            raise ValueError(f"Reference model {reference_model} not found in responses")
        
        # Calculate scores for all models (except reference model)
        model_scores = {}
        
        for model_name, response in responses.items():
            if model_name == reference_model:
                continue
                
            print(f"Evaluating {model_name}...")
            
            # Calculate base metrics
            relevance = self.metrics.compute_relevance(reference, response)
            accuracy = self.metrics.compute_accuracy(response)
            groundedness = self.metrics.compute_groundedness(reference, response)
            completeness = self.metrics.compute_completeness(reference, response)
            
            # Calculate RAGAS metrics
            bleu = await self.metrics.compute_bleu_ragas(reference, response)
            rouge = await self.metrics.compute_rouge_ragas(reference, response)
            
            # Store all metrics
            scores = {
                "Relevance": round(relevance, 3),
                "Accuracy": round(accuracy, 3),
                "Groundedness": round(groundedness, 3),
                "Completeness": round(completeness, 3),
                "BLEU": round(bleu, 3),
                "ROUGE": round(rouge, 3)
            }
            
            # Calculate weighted score
            weighted_score = sum(scores[metric] * weight 
                               for metric, weight in self.weights.items())
            scores["Overall"] = round(weighted_score, 3)
            
            model_scores[model_name] = scores
        
        # Sort models by overall score
        sorted_models = sorted(model_scores.items(), 
                              key=lambda x: x[1]["Overall"], 
                              reverse=True)
        
        return {
            "reference_model": reference_model,
            "reference_response": reference,
            "model_scores": model_scores,
            "rankings": sorted_models
        }