# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 16:28:17 2019

@author: prajw
"""

ids = {}
questions = {}
with open("quora_duplicate_questions.tsv", "r+", encoding="utf8") as inpfile:
    next(inpfile)
    lineno = 1
    for line in inpfile:
        lineno += 1
        l = line.strip().split("\t")
        try:
            if int(l[1]) not in questions:
                questions[int(l[1])] = l[3]
            if int(l[2]) not in questions:
                questions[int(l[2])] = l[4]
            if (int(l[1]), int(l[2])) not in ids and (int(l[2]), int(l[1])) not in ids:
                ids[(int(l[1]), int(l[2]))] = l[-1]
        except:
            print(lineno)

outfile = open("unique_questions.txt", "w+", encoding="utf8")
for i in sorted(questions.keys()):
    print(i, questions[i], file=outfile)
outfile.close()

outfile = open("question_pairs.txt", "w+", encoding="utf8")
for i in sorted(ids):
    print(i[0], i[1], ids[i], file=outfile)
outfile.close()