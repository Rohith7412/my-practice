from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

chunks = [
    "ABC Institute offers courses in Artificial Intelligence, Data Science, Python, Machine Learning and Web Development.",

    "The Artificial Intelligence course duration is 6 months.",

    "The Data Science course duration is 8 months.",

    "Python is taught during the first two months of the Artificial Intelligence course.",

    "The Machine Learning module is taught during months three and four.",

    "The AI course includes Python, Machine Learning, Deep Learning, NLP and Generative AI.",

    "Students must complete a final project to receive the course certificate.",

    "Classes are conducted from Monday to Friday.",

    "The institute provides both online and offline classes."
]

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chunk_embeddings = embedding_model.encode(chunks)

llm = pipeline("text-generation", model="google/flan-t5-small")

question = input("Ask a question: ")
question_embedding = embedding_model.encode(question)
scores = cosine_similarity([question_embedding],chunk_embeddings)[0]
best_index = scores.argmax()
relevant_chunk = chunks[best_index]

prompt = f"""
Answer the question using only the context.

Question:
{question}

Answer:
"""

response = llm(prompt,max_new_tokens=100)
print("\nRetrieved Context:")
print(relevant_chunk)
print("\nLLM Answer:")
print(response[0]["generated_text"])