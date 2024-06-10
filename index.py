from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

def classify_query(query):
   
    if "translate" in query.lower():
        return "translation"
    elif "summarize" in query.lower():
        return "summarization"
    elif "write" in query.lower():
        return "creative_writing"
    else:
        return "general"

def suggest_llm(query_type):
 
    if query_type == "translation":
        return "Helsinki-NLP/opus-mt-en-fr"
    elif query_type == "summarization":
        return "facebook/bart-large-cnn"
    elif query_type == "creative_writing":
        return "gpt2"
    else:
        return "gpt2"

def main():
    while True:
        user_query = input("Please enter your query (type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            print("Exiting...")
            break

        query_type = classify_query(user_query)
        suggested_llm = suggest_llm(query_type)

        print(f"Using model: {suggested_llm}")

        model = AutoModelForCausalLM.from_pretrained(suggested_llm)
        tokenizer = AutoTokenizer.from_pretrained(suggested_llm)

        generator = pipeline('text-generation', model=model, tokenizer=tokenizer, framework='pt', pad_token_id=tokenizer.eos_token_id)

        response = generator(user_query, max_length=100, truncation=True)
        
        print("Generated Response:")
        print(response[0]['generated_text'])

if __name__ == "__main__":
    main()
