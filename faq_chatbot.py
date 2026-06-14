import nltk
import string
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download tokenizer
nltk.download('punkt')

# FAQ Questions and Answers
faq_data = {
    "What is AI?": "AI stands for Artificial Intelligence, which enables machines to simulate human intelligence.",
    "What is Machine Learning?": "Machine Learning is a subset of AI that allows computers to learn from data.",
    "What is Python?": "Python is a high-level programming language used in AI, web development, and data science.",
    "What is NLP?": "Natural Language Processing helps computers understand and process human language.",
    "What is ChatGPT?": "ChatGPT is an AI chatbot developed by OpenAI.",
    "What is Deep Learning?": "Deep Learning is a branch of Machine Learning based on neural networks.",
    "Why is Python popular?": "Python is easy to learn and has powerful libraries for AI and data science.",
    "What are the applications of AI?": "AI is used in healthcare, finance, education, robotics, and many other fields.",
    "Who developed Python?": "Python was created by Guido van Rossum.",
    "What is Data Science?": "Data Science is the study of extracting useful information from data."
}

# Text preprocessing
def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in string.punctuation]
    return " ".join(tokens)

# Prepare data
questions = list(faq_data.keys())
processed_questions = [preprocess(q) for q in questions]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(processed_questions)

# Chatbot response function
def chatbot_response(user_input):
    user_input = preprocess(user_input)
    user_vector = vectorizer.transform([user_input])

    similarity_scores = cosine_similarity(user_vector, question_vectors)
    best_match_index = similarity_scores.argmax()
    score = similarity_scores[0][best_match_index]

    if score > 0.2:
        return faq_data[questions[best_match_index]]
    else:
        return "Sorry, I don't understand your question."

# Main Chat Loop
print("\n===== FAQ CHATBOT =====")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Bot: Goodbye!")
        break

    response = chatbot_response(user_input)
    print("Bot:", response)
