 # -*- coding:utf-8 -*- 


import json

def gen_answer_file(answers):

    documents = []
    questions = []
    qids = []
    f = open('../data/squad/my/dev.txt', 'r', encoding='utf-8')
    while True:
        line = f.readline()
        if not line:
            break
        question = line.strip()
        qid = f.readline().strip()
        document = f.readline().strip()

        questions.append(question)
        qids.append(qid)
        documents.append(document)

        f.readline()
    f.close()
    dic = {}
    for answer in answers:
        idx = qids.index(answer[0]) 
        seq = documents[idx].split(' ')
        ans = seq[answer[1]:answer[2]+1]
        if len(ans) > 1 :
            dic[answer[0]] = ' '.join(ans)
        elif len(ans) == 1 :
            dic[answer[0]] = ans[0]
        else :
            dic[answer[0]] = ' '
    f = open('../data/squad/my/predictionv1.txt', 'w')
    json.dump(dic, f)
    f.close()
    print('Have generated prediction file.')