import sys
import pickle
from sklearn.feature_extraction.text import HashingVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def is_int(input):
    try:
        num = int(input)
    except ValueError:
        return False
    return True


def pre_processor(doc):
    return doc.lower()


def tokenizer(doc):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(doc)
    return [lemmatizer.lemmatize(token) for token in tokens]


def load_state():
    hash_vector = pickle.load(open("data/hash_vector.pkl", "rb"))
    list_questions = pickle.load(open("data/list_question.pkl", "rb"))

    hash_vectorizer = HashingVectorizer(preprocessor = pre_processor, tokenizer = tokenizer)

    return hash_vectorizer, hash_vector, list_questions


def get_most_similar(input_question, return_size = 100):
    hash_vectorizer, hash_vector, list_questions = load_state()

    input_question_vector = hash_vectorizer.transform([input_question])

    # Calculating cosine similarity here. Ignoring denominator in the formula because it will remain that same. This calculation finally leads to an argmax task
    cosine_sim_vector = (hash_vector * input_question_vector.T)
    list_sim_values = cosine_sim_vector.toarray().tolist()
    list_sim_values = [entry[0] for entry in list_sim_values]
    list_sim_values = [(i, list_sim_values[i]) for i in range(len(list_sim_values))]
    list_sim_values = sorted(list_sim_values, key = lambda tup: tup[1], reverse = True)
    
    # A list of top return_size highest questions
    top_list = list_sim_values[:return_size]

    return_list = list()

    for entry in top_list:
        entry_ = list_questions[entry[0]]
        return_list.append(entry_)
    
    return_list = [entry[1] for entry in return_list]
    
    return return_list


if __name__ == "__main__": 
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print ("""
            Usage: python get_most_similar.py [input_question] [type integer: return_size(optional)]
            Gets the most similar questions to your question.\n""")
        sys.exit(1)
    
    return_size = 100
    if len(sys.argv) == 3:
        return_size = sys.argv[2]
        if not is_int(return_size):
            print ("""
                Usage: python get_most_similar.py [input_question] [type integer: return_size(optional)]
                Gets the most similar questions to your question.\n""")
            sys.exit(1)
        return_size = int(return_size)

    return_list = get_most_similar(sys.argv[1], return_size)
    print(return_list)