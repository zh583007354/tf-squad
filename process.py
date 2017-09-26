
import sys
import nltk
from nltk.tokenize import word_tokenize


# format:

#sequence1(Doument) \t sequence2(Question) \t sequence of the positions where the answer appear in Document (e.g. 3 4 5 6) \n
def prepSQuAD():
	reload(sys)
	sys.setdefaultencoding('utf-8')
	import json
	
	count = 0
	filenames = ['dev', 'train']
	for filename in filenames:
		fpr = open("data/squad/"+filename+"-v1.1.json", 'r')
		line = fpr.readline()
		js = json.loads(line)
		fpw = open("data/squad/my/"+filename+".txt", 'w')
		for c in js["data"]:
			for p in c["paragraphs"]:
				context = p["context"].split(' ')
				for qa in p["qas"]:

					question = word_tokenize(qa["question"])
					q_id = qa["id"]

					if filename == 'train':
						for a in qa['answers']:
							answer = a['text'].strip()
							answer_start = int(a['answer_start'])

						#add '.' here, just because NLTK is not good enough in some cases
						answer_words = word_tokenize(answer+'.')
						if answer_words[-1] == '.':
							answer_words = answer_words[:-1]
						else:
							answer_words = word_tokenize(answer)

						prev_context_words = word_tokenize( p["context"][0:answer_start ] )
						left_context_words = word_tokenize( p["context"][answer_start:] )
						answer_reproduce = []
						for i in range(len(answer_words)):
							if i < len(left_context_words):
								w = left_context_words[i]
								answer_reproduce.append(w)
						join_a = ' '.join(answer_words)
						join_ar = ' '.join(answer_reproduce)

						#if not ((join_ar in join_a) or (join_a in join_ar)):
						if join_a != join_ar:
							#print join_ar
							#print join_a
							#print 'answer:'+answer
							count += 1
						fpw.write(' '.join(question)+'\n')
						pos_list = []
						for i in range(len(answer_words)):
							if i < len(left_context_words):
								pos_list.append(str(len(prev_context_words)+i+1))
						if len(pos_list) == 0:
							print join_ar
							print join_a
							print 'answer:'+answer
						assert(len(pos_list) > 0)
						fpw.write(' '.join(pos_list)+'\n')
						fpw.write(' '.join(prev_context_words+left_context_words)+'\n')
						fpw.write('\n')
					else:
						fpw.write(' '.join(question)+'\n')
						fpw.write(q_id+'\n')
						fpw.write(' '.join(word_tokenize( p["context"]) )+'\n')
						fpw.write('\n')

		fpw.close()
	print ('SQuAD preprossing finished!')


if __name__ == "__main__":
	prepSQuAD()