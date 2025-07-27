from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever  # Ensure retriever is defined and initialized in vector.py

# === Step 1: Set up LLM and Prompt ===
model = OllamaLLM(model="llama3")  # Use the actual name of the model installed in Ollama

template = """
You are an expert in answering questions about the contents of a restaurant-related PDF.

Here are some relevant excerpts from the document:
{reviews}

Here is the question to answer:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# === Step 2: Interactive Q&A Loop ===
while True:
    print("\n-------------------------------")
    question = input("Ask your question (q to quit): ").strip()

    if question.lower() == "q":
        break

    # Step 2.1: Retrieve relevant documents
    try:
        review_docs = retriever.invoke(question)
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        continue

    if not review_docs:
        print("No relevant information found.")
        continue

    review_texts = "\n\n".join([doc.page_content for doc in review_docs])

    # Step 2.2: Generate response
    try:
        result = chain.invoke({"reviews": review_texts, "question": question})
        # Some versions return a BaseMessage, others return plain text
        print("\nAnswer:")
        print(result.content if hasattr(result, "content") else result)
    except Exception as e:
        print(f"Error generating answer: {e}")
