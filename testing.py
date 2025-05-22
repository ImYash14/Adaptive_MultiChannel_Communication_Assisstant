import os
import shutil
import logging
from typing import List, Dict, Union
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class RAGEvaluator:
    def __init__(self, model_name: str = "gpt2"):
        """
        Initialize RAG Evaluator with a language model.

        :param model_name: Hugging Face model name for perplexity calculation
        """
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Setup tokenizer and model for perplexity
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()  # Ensure model is in evaluation mode

    def simple_tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization method to replace NLTK's word_tokenize.

        :param text: Input text to tokenize
        :return: List of tokens
        """
        # Remove punctuation and split on whitespace
        return text.lower().replace(',', '').replace('.', '').split()

    def calculate_bleu_score(self, reference: str, candidate: str) -> float:
        """
        Calculate simplified BLEU-like score between reference and candidate.

        :param reference: Ground truth text
        :param candidate: Generated text
        :return: BLEU-like score
        """
        ref_tokens = self.simple_tokenize(reference)
        cand_tokens = self.simple_tokenize(candidate)

        try:
            # Simple precision calculation
            matching_tokens = set(ref_tokens) & set(cand_tokens)
            bleu_score = len(matching_tokens) / len(set(ref_tokens))
            return bleu_score
        except Exception as e:
            self.logger.warning(f"BLEU score calculation error: {e}")
            return 0.0

    def calculate_perplexity(self, text: str) -> float:
        """
        Calculate perplexity of generated text.

        :param text: Text to calculate perplexity for
        :return: Perplexity score
        """
        try:
            inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)
            with torch.no_grad():
                outputs = self.model(inputs['input_ids'], labels=inputs['input_ids'])
                loss = outputs.loss
            return torch.exp(loss).item()
        except Exception as e:
            self.logger.warning(f"Perplexity calculation error: {e}")
            return float('inf')

    def calculate_context_relevance(self, query: str, context: str) -> Dict[str, float]:
        """
        Calculate context relevance metrics.

        :param query: Original query
        :param context: Retrieved context
        :return: Dictionary of relevance metrics
        """
        query_tokens = set(self.simple_tokenize(query))
        context_tokens = set(self.simple_tokenize(context))

        overlap = query_tokens.intersection(context_tokens)
        precision = len(overlap) / len(query_tokens) if query_tokens else 0
        recall = len(overlap) / len(context_tokens) if context_tokens else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    def evaluate_rag_system(self, queries: List[str], responses: List[str], contexts: List[str]) -> Dict[str, Union[float, List[float]]]:
        """
        Comprehensive evaluation of RAG system.

        :param queries: List of queries
        :param responses: List of generated responses
        :param contexts: List of retrieved contexts
        :return: Evaluation metrics dictionary
        """
        if not (len(queries) == len(responses) == len(contexts)):
            raise ValueError("Number of queries, responses, and contexts must match.")

        bleu_scores = []
        perplexity_scores = []
        context_relevance_scores = []

        for query, response, context in zip(queries, responses, contexts):
            bleu = self.calculate_bleu_score(query, response)
            bleu_scores.append(bleu)

            perplexity = self.calculate_perplexity(response)
            perplexity_scores.append(perplexity)

            context_relevance = self.calculate_context_relevance(query, context)
            context_relevance_scores.append(context_relevance['f1_score'])

        return {
            "average_bleu_score": np.mean(bleu_scores),
            "bleu_scores": bleu_scores,
            "average_perplexity": np.mean(perplexity_scores),
            "perplexity_scores": perplexity_scores,
            "average_context_relevance": np.mean(context_relevance_scores),
            "context_relevance_scores": context_relevance_scores
        }


def main():
    # Define queries and responses
    queries = [
        """ I recently purchased a software license from your company, and while the features are impressive, I encountered several issues. First, the installation process was overly complicated, and I had to spend hours figuring it out despite following the instructions. Second, the customer service team was unresponsive when I reached out for help — I waited for over three days to get a reply, which was very frustrating. On a positive note, the software itself has excellent functionality, and I’m very pleased with how it integrates with other tools I use. However, the user interface could use some improvements as it’s not very intuitive. Additionally, I noticed your FAQ section doesn’t address some common technical issues I faced. Overall, the product has potential, but the experience has been mixed due to these challenges.
""",
        """I recently ordered a smartwatch from your electronics section, and I have mixed feelings about my experience. The delivery was surprisingly fast, and I appreciated the regular updates on the shipment status. However, the packaging was poorly done, and the product box was slightly damaged upon arrival, which was disappointing. The watch itself works fine, but the strap quality feels subpar and uncomfortable to wear for long hours. Moreover, the setup instructions were confusing and lacked clarity, making it hard for me to pair the watch with my phone. I also reached out to your support team for help, and while they eventually resolved my issue, the initial response time was too slow. I’ve generally been happy with your products in the past, but this experience didn’t meet my expectations. Lastly, your FAQ didn’t have enough information on troubleshooting pairing issues, which could have saved me some frustration"""
    ]
    responses = [
        """=== Sentiment Analysis ===
Overall Sentiment: MIXED

Detailed Aspects:
- Purchasing process: neutral (The customer was able to purchase the software license, but no explicit sentiment was expressed about the process.)
- Installation process: negative (The customer found the installation process to be overly complicated and had to spend several hours figuring it out.)
- Customer service: negative (The customer service team was unresponsive, and the customer had to wait for over three days to receive a reply.)
- Software functionality: positive (The customer is pleased with the software's functionality and how it integrates with other tools.)
- User interface: negative (The user interface is not very intuitive, which could be improved.)
- FAQ section: negative (The FAQ section does not address some common technical issues the customer faced.)
- Overall product potential: positive (Despite the challenges, the customer sees potential in the product.)

=== Generated Response ===
Dear Customer,

Thank you for taking the time to share your feedback regarding your recent purchase of a software license from Happy Company. We truly value your input as it helps us identify areas where we can improve and continue to provide excellent service to our customers.

We appreciate your positive comments regarding the software's functionality and its integration with other tools you use. We're glad to hear that you find these features impressive and beneficial. However, we understand your concerns regarding the installation process and user interface.

We apologize for the complications you encountered during the installation process, and we will take your feedback into account to simplify it for future customers. Additionally, we acknowledge the difficulties you faced with the user interface, and we will work on making it more intuitive to enhance the user experience.

Regarding your customer service experience, we sincerely apologize for the delay in response. We strive to provide timely assistance to all our customers, and we will investigate the cause of the delay to prevent it from happening in the future.

We also acknowledge your feedback regarding our FAQ section. We will update it to include answers to common technical issues our customers may face.

We're thrilled to hear that despite the challenges, you see potential in our product. Your feedback is essential to us, and we are committed to addressing your concerns and providing you with the best possible experience.

Please do not hesitate to reach out if you have any further questions or concerns.
""",
    
"""Overall Sentiment: MIXED

Detailed Aspects:
- Product Quality: negative (The strap quality of the smartwatch feels subpar and uncomfortable to wear for long hours.)
- Delivery: positive (The delivery was surprisingly fast, and regular updates on the shipment status were appreciated.)
- Packaging: negative (The packaging was poorly done, and the product box was slightly damaged upon arrival.)
- Setup Instructions: negative (The setup instructions were confusing and lacked clarity, making it hard for the customer to pair the watch with their phone.)
- Customer Support: mixed (While the support team eventually resolved the issue, the initial response time was too slow.)
- FAQ/Troubleshooting: negative (The FAQ didn't have enough information on troubleshooting pairing issues, causing frustration for the customer.)
- Past Experience with Happy Company: positive (The customer has generally been happy with products from Happy Company in the past.)

=== Generated Response ===
Dear Customer,

Thank you for taking the time to share your feedback regarding your recent order with us. We appreciate your honesty and are grateful for the opportunity to address your concerns.

Firstly, we would like to commend our delivery team for the fast and efficient service they provided. We are pleased to hear that the regular updates on the shipment status were appreciated.

However, we are sorry to learn that the packaging was poorly done, and the product box was slightly damaged upon arrival. We understand how disappointing this must have been, and we will ensure that our packaging team takes extra care in handling and packing our products in the future.

Regarding the smartwatch, we apologize for the subpar quality of the strap and the discomfort it has caused. We want to assure you that our quality control team will look into this matter and take appropriate actions to prevent similar issues from occurring in the future.

We are also sorry to hear that the setup instructions were confusing and lacked clarity, causing frustration during the pairing process. We will review our instructions and make necessary improvements to ensure they are clear and easy to understand.

We regret the slow initial response time from our support team, and we appreciate your patience as they worked to resolve your issue. We will take steps to improve our response time and ensure that our customers receive timely assistance.

Finally, we understand that our FAQ/Troubleshooting section did not have enough information on troubleshooting pairing issues. We will update our FAQ section to include more detailed information and resources to help our customers resolve common issues.

At Happy Company, we strive to provide the best possible products and services to our customers. We take your feedback seriously and are committed to making improvements based on your suggestions.

Once again, thank you for your feedback, and we hope to have the opportunity to serve you better in the future.

Sincerely,

[Your Name]
Customer Service Representative
Happy Company

"""
    ]

    # Add meaningful contexts to demonstrate context relevance
    contexts = [
        "Software license customer support involves resolving installation problems, providing clear documentation, and offering timely technical assistance.",
        "Electronic device customer feedback often focuses on product quality, packaging, and initial setup experience for smartwatches and similar gadgets."
    ]

    # Evaluate
    try:
        evaluator = RAGEvaluator()
        results = evaluator.evaluate_rag_system(queries, responses, contexts)

        print("RAG System Evaluation Results:")
        print(f"Average BLEU Score: {results['average_bleu_score']:.4f}")
        print(f"Average Perplexity: {results['average_perplexity']:.4f}")
        print(f"Average Context Relevance: {results['average_context_relevance']:.4f}")
        print("\nDetailed Context Relevance Scores:")
        for i, score in enumerate(results['context_relevance_scores'], 1):
            print(f"Query {i} Context Relevance: {score:.4f}")
    except Exception as e:
        print(f"Error in RAG system evaluation: {e}")


if __name__ == "__main__":
    main()