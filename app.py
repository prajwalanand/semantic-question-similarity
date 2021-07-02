import re
import json
from flask import Flask, request, render_template, redirect, url_for
from get_most_similar import get_most_similar
from predict_score import model
app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('search_window.html')

@app.route('/handle_data', methods=['POST'])
def handle_data():
    with open("black_list_words.json") as inpfile:
        black_list = json.load(inpfile)['block_words']
    predictor = model()
    sentence = request.form['question']
    print("Sentence: ", sentence)
    list_questions = get_most_similar(sentence,200)
    print(list_questions)
    score = []
    for q in list_questions:
        if any([x in q for x in black_list]):
            continue
        s = predictor.get_score(sentence,q)
        link = parse_string(q)
        # score.append((s,link))
        # s=0
        score.append((s,q,link))
        # print("Link: ",link)
    score = sorted(score,reverse=True)[:5]
    print("Top 5 questions: ",score)
    # session['sentences'] = score

    return render_template('search_window.html',results=score)
    # return redirect('/show_results')
    # return redirect(url_for('.parse_string', sentence=sentence))

def parse_string(s):
    s = re.sub(r'[^\w\d\s]','',s)
    s = re.sub(r'[\s]','-',s)
    s = 'https://www.quora.com/'+s
    # print(s)
    return s
    # return redirect('/')

@app.route('/show_results')
def show_results():
    vals = session.get('sentences',None)
    print(vals)
    # return render_template('search_window.html',results=vals)
