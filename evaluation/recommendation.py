class ScoreRecommender:
    """Generates improvement recommendations based on evaluation scores"""
    
    def __init__(self):
        # Define threshold values for each metric
        self.thresholds = {
            "Relevance": 0.85,
            "Accuracy": 0.9,
            "Groundedness": 0.7,
            "Completeness": 0.8,
            "BLEU": 0.1,
            "ROUGE": 0.25
        }
        
        # Recommendations for improving each metric
        self.recommendations = {
            "Relevance": [
                "Focus more directly on the question's core topic",
                "Ensure the response addresses the specific query rather than related topics",
                "Use key terms from the question in the response",
                "Structure the response to match the question format (how-to, list, explanation)"
            ],
            "Accuracy": [
                "Double-check factual statements for correctness",
                "Avoid making claims without sufficient supporting evidence",
                "Use more precise language and avoid generalizations",
                "Consider mainstream consensus on topics rather than fringe perspectives"
            ],
            "Groundedness": [
                "Include more information from the reference material",
                "Connect each part of the response to specific information from the source",
                "Avoid introducing new information not present in the reference",
                "Use similar terminology and phrasing as the reference material"
            ],
            "Completeness": [
                "Cover all key points mentioned in the reference material",
                "Include important details that provide context",
                "Address all aspects of the question comprehensively",
                "Ensure no critical information is omitted from the response"
            ],
            "BLEU": [
                "Use more n-gram phrases from the reference text",
                "Include key terminology present in the reference",
                "Match sentence structure where appropriate",
                "Maintain similar length and detail level as the reference"
            ],
            "ROUGE": [
                "Include more overlapping words with the reference",
                "Use similar sentence structures to the reference",
                "Maintain similar sequence of ideas as the reference",
                "Include important keywords from the reference"
            ]
        }
    
    def get_recommendations(self, model_name, scores):
        """
        Generate recommendations based on evaluation scores
        
        Args:
            model_name: Name of the model
            scores: Dict of metric scores
            
        Returns:
            Dict of metrics that need improvement with recommendations
        """
        recommendations = {}
        
        for metric, score in scores.items():
            if metric == "Overall":
                continue
                
            threshold = self.thresholds.get(metric, 0.7)
            
            # Check if score is below threshold
            if score < threshold:
                # Select 1-2 relevant recommendations for this metric
                import random
                num_recommendations = min(2, len(self.recommendations[metric]))
                selected_recommendations = random.sample(self.recommendations[metric], num_recommendations)
                
                recommendations[metric] = {
                    "score": score,
                    "target": threshold,
                    "gap": round(threshold - score, 2),
                    "suggestions": selected_recommendations
                }
        
        return recommendations