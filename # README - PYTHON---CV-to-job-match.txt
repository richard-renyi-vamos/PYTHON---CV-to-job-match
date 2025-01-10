CHATGPT 

*******

ChatGPT said:
ChatGPT
Here's a Python script that compares the content of a CV to a job description and generates recommendations. It uses Natural Language Processing (NLP) techniques, such as keyword extraction and semantic similarity, to evaluate the match.

Cv Job Match

Answer in chat instead
This script compares the CV to the job description by extracting keywords and calculating their similarity. It provides specific recommendations by identifying keywords from the job description that are missing in the CV.

How to use:
Install the spacy library if you don't already have it:
bash
Copy code
pip install spacy
python -m spacy download en_core_web_sm
Run the script and paste your CV and job description when prompted.
The program will output a similarity score and a list of recommendations to enhance your CV.
Let me know if you need help with testing or enhancing the code further! 😊

**********

import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_keywords(text, nlp):
    """Extracts keywords from text using spaCy."""
    doc = nlp(text)
    keywords = [token.text for token in doc if token.is_alpha and not token.is_stop]
    return keywords

def calculate_similarity(cv_text, jd_text, nlp):
    """Calculates the similarity between CV and job description."""
    cv_keywords = extract_keywords(cv_text, nlp)
    jd_keywords = extract_keywords(jd_text, nlp)

    vectorizer = CountVectorizer().fit_transform([' '.join(cv_keywords), ' '.join(jd_keywords)])
    vectors = vectorizer.toarray()
    similarity = cosine_similarity(vectors)[0, 1]
    return similarity, cv_keywords, jd_keywords

def recommend_improvements(cv_keywords, jd_keywords):
    """Recommends improvements to the CV based on missing keywords."""
    missing_keywords = set(jd_keywords) - set(cv_keywords)
    return list(missing_keywords)

def main():
    # Load the spaCy language model
    nlp = spacy.load("en_core_web_sm")

    # Input CV text and job description
    cv_text = input("Paste your CV text:\n")
    jd_text = input("Paste the job description:\n")

    # Calculate similarity and extract recommendations
    similarity, cv_keywords, jd_keywords = calculate_similarity(cv_text, jd_text, nlp)
    recommendations = recommend_improvements(cv_keywords, jd_keywords)

    print(f"\nSimilarity Score: {similarity:.2f}")
    if recommendations:
        print("\nRecommendations:")
        for keyword in recommendations:
            print(f"- Add details about '{keyword}'")
    else:
        print("Your CV closely matches the job description!")

if __name__ == "__main__":
    main()
