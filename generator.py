import warnings
warnings.filterwarnings("ignore", message=".*Accessing.*__path__.*")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

class RecipeGenerator:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load in half-precision (bfloat16) to easily fit on a consumer GPU
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, 
            torch_dtype=torch.bfloat16,
            device_map=device
        )

    def generate(self, query: str, retrieved_docs: list[dict]) -> str:
        # 1. Format the retrieved recipes into the prompt context
        context_blocks = []
        for i, doc in enumerate(retrieved_docs, start=1):
            context_blocks.append(f"--- Recipe {i} ---\n{doc['recipe_text']}\n")
        
        context_str = "\n".join(context_blocks)
        
        system_prompt = (
            "You are a helpful culinary assistant. Use ONLY the provided recipes "
            "to answer the user's question. If a recipe contains the answer, show the recipe in a way that both answers user's question and illustrates the cooking."
            "If the recipes do not contain the answer, "
            "say 'I cannot answer this based on the provided recipes.' However, in this case include one recipe in your answer and indicate it is the closest match based on the question. Do not use outside knowledge. "
            "However, if the user asks for substitutions or modifications, provide reasonable suggestions based on the ingredients in the recipes."
        )
        
        user_message = f"Recipes Context:\n{context_str}\n\nQuestion: {query}"

        # 2. Format using Llama-3's official chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        
        encoded_inputs = self.tokenizer.apply_chat_template(
            messages, 
            return_tensors="pt", 
            add_generation_prompt=True,
            return_dict=True
        ).to(self.model.device)

        # 3. Generate the answer
        outputs = self.model.generate(
            **encoded_inputs,
            max_new_tokens=1400,
            temperature=0.3, # Low temp for factual answers
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Slice off the prompt tokens to return only the generated answer
        generated_tokens = outputs[0][encoded_inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

if __name__ == "__main__":
    from retrieval import RecipeRetriever
    print("Running in cli demo mode...")
    print("Loading Retriever...")
    retriever = RecipeRetriever()
    print("Loading Generator...")
    generator = RecipeGenerator()
    
    query = "what can i do with chicken soup campbell, milk, butter and mushroom?"
    print(f"\nQuery: {query}")
    
    docs = retriever.search(query, k=2)
    print("\nSynthesizing answer from retrieved docs...")
    answer = generator.generate(query, docs)
    
    print(f"\n--- LLM Answer ---\n{answer}\n")
