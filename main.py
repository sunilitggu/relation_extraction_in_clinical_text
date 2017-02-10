from sklearn.cross_validation import KFold
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import numpy as np
import sklearn as sk
import csv
import re
from geniatagger import GeniaTagger
tagger = GeniaTagger('/home/sunilnew/python_packages/geniatagger-3.0.2/geniatagger')
from nltk.tokenize import WordPunctTokenizer
tokenizer = WordPunctTokenizer()
import pickle
from cnn_train import *


def preProcess(sent):
	sent = sent.lower()
	sent = tokenizer.tokenize(sent)
	sent = ' '.join(sent)
	sent = re.sub('\d', 'dg',sent)
	sent_list,_,_,_,_ = zip(*tagger.parse(sent)) 
	sent = ' '.join(sent_list)
	return sent

def find_sub_list(sl,l):
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            return ind,ind+sll-1


def makePaddedList(sent_contents, pad_symbol= '<pad>'):
	maxl = max([len(sent) for sent in sent_contents])
	T = []
 	for sent in sent_contents:
		t = []
		lenth = len(sent)
		for i in range(lenth):
			t.append(sent[i])
		for i in range(lenth,maxl):
			t.append(pad_symbol)
		T.append(t)	

	return T, maxl

def makeWordList(sent_list):
	wf = {}
	for sent in sent_list:
		for w in sent:
 			if w in wf:
				wf[w] += 1
			else:
				wf[w] = 0
	wl = {}
	i = 0
#	wl['unkown'] = 0	
	for w,f in wf.iteritems():		
		wl[w] = i
		i += 1
	return wl

def mapWordToId(sent_contents, word_dict):
	T = []
	for sent in sent_contents:
		t = []
		for w in sent:
			t.append(word_dict[w])
		T.append(t)
	return T

def mapLabelToId(sent_lables, label_dict):
#	print"sent_lables", sent_lables
#	print"label_dict", label_dict
	return [label_dict[label] for label in sent_lables]

def dataRead(fname):
	print "Input File Reading"
	fp = open(ftrain, 'r')
	samples = fp.read().strip().split('\n\n\n')
  	sent_names     = []		#1-d array 
	sent_lengths   = []		#1-d array
  	sent_contents  = []		#2-d array [[w1,w2,....] ...]
  	sent_lables    = []		#1-d array
  	entity1_list   = []		#2-d array [[e1,e1_s,e1_e,e1_t] [e1,e1_s,e1_e,e1_t]...]
  	entity2_list   = []		#2-d array [[e1,e1_s,e1_e,e1_t] [e1,e1_s,e1_e,e1_t]...]
  	for sample in samples:
		name, sent, entities, relation = sample.strip().split('\n')

		ma = re.match(r"\[['\"](.*)['\"], '(.*)', ['\"](.*)['\"]\]", relation.strip())
		if(ma):
			lable = ma.group(2)		
		elif relation == '[0]':
			lable = 'other'
		else:
			print "Error in reading", relation
			exit(0)

		if (lable == 'TrWP' or lable == 'TrNAP' or lable == 'TrIP'):
			continue
		sent_lables.append(lable)
#		sent = preProcess(sent)
#		sent_lengths.append(len(sent.split()))
		sent_names.append(name)
		sent_contents.append(sent)

		m = re.match(r"\(\[['\"](.*)['\"], (\d*), (\d*), '(.*)'\], \[['\"](.*)['\"], (\d*), (\d*), '(.*)'\]\)", entities.strip())
		if m :
			e1   = m.group(1)
#			e1 = preProcess(e1)

			e1_s = int(m.group(2))
			e1_e = int(m.group(3))
			e1_t = m.group(4)

			e2   = m.group(5) 
#			e2 = preProcess(e2)
			e2_s = int(m.group(6))
			e2_e = int(m.group(7))
			e2_t = m.group(8)
			if(e1_s < e2_s):
				entity1_list.append([e1,e1_s,e1_e,e1_t])
				entity2_list.append([e2,e2_s,e2_e,e2_t])
			else:
				entity1_list.append([e2,e2_s,e2_e,e2_t])
				entity2_list.append([e1,e1_s,e1_e,e1_t])
#			print e1,e2
		else:
			print "Error in readign", entities.strip()
#			exit(0)	
  	return sent_contents, entity1_list, entity2_list, sent_lables 

def makeFeatures(sent_list, entity1_list, entity2_list):
	print 'Making Features'
	word_list = []
	d1_list = []
	d2_list = []
	type_list = []
	pos_list = []
	chunk_list = []
	for sent, ent1, ent2 in zip(sent_list, entity1_list, entity2_list):
		sent = preProcess(sent)
		sent_list1, _, pos_list1, chunk_list1, _ = zip(*tagger.parse(sent))
		
		entity1 = preProcess(ent1[0]).split()
		entity2 = preProcess(ent2[0]).split()
#		print entity1
#		print sent_list1
		s1,e1 = find_sub_list(entity1, list(sent_list1))
		s2,e2 = find_sub_list(entity2, list(sent_list1))
		# distance1 feature	
		d1 = []
		for i in range(len(sent_list1)):
		    if i < s1 :
			d1.append(str(i - s1))
		    elif i > e1 :
			d1.append(str(i - e1 ))
		    else:
			d1.append('0')
		#distance2 feature		
		d2 = []
		for i in range(len(sent_list1)):
		    if i < s2:
			d2.append(str(i - s2))
		    elif i > e2:
			d2.append(str(i - s2))
		    else:
			d2.append('0')
		#type feature
		t = []
		for i in range(len(sent_list1)):
			t.append('Out')
		for i in range(s1, e1+1):
			t[i] = ent1[3]		
		for i in range(s2, e2+1):
			t[i] = ent2[3]

		word_list.append(sent_list1)
		pos_list.append(pos_list1)
		chunk_list.append(chunk_list1)
		d1_list.append(d1)
		d2_list.append(d2)
 		type_list.append(t) 

    	return word_list, pos_list, chunk_list, d1_list, d2_list, type_list

def readWordEmb(word_dict, fname, embSize=50):
	print "Reading word vectors"
	wv = []
	wl = []
	with open(fname, 'r') as f:
		for line in f :			
			vs = line.split()
			if len(vs) < 50 :
				continue
			vect = map(float, vs[1:])
			wv.append(vect)
			wl.append(vs[0])
	wordemb = []
	count = 0
	for word, id in word_dict.iteritems():
		if word in wl:
			wordemb.append(wv[wl.index(word)])
		else:
			count += 1
			wordemb.append(np.random.rand(embSize))
	wordemb = np.asarray(wordemb, dtype='float32')
	print "number of unknown word in word embedding", count
	return wordemb


ftrain = "data/combine.train"
#ftrain = "../i2b2_data/temp.train"
#ftrain = '../i2b2_data/beth.train'
#ftrain = '../i2b2_data/test.train'
emb = True
#wefile = '/home/sunilnew/python_prog/relation_extraction/word2vec/scripts/i2b2_corpus_word2vec.txt'
wefile = '/home/sunilnew/python_prog/relation_extraction/word2vec/scripts/50dim_drug_disease_cbow_w9_hs.txt'

sent_contents, entity1_list, entity2_list, sent_lables = dataRead(ftrain)
word_list, pos_list, chunk_list, d1_list, d2_list, type_list = makeFeatures(sent_contents, entity1_list, entity2_list)

print "Length of word list", len(word_list)

#padding
word_list, seq_len = makePaddedList(word_list)
pos_list,_ = makePaddedList(pos_list)
chunk_list,_ = makePaddedList(chunk_list) 
d1_list,_ = makePaddedList(d1_list)
d2_list,_ = makePaddedList(d2_list)
type_list,_ = makePaddedList(type_list)

# Wordlist
#label_dict = {'other':0, 'TrWP': 1, 'TeCP': 2, 'TrCP': 3, 'TrNAP': 4, 'TrAP': 5, 'PIP': 6, 'TrIP': 7, 'TeRP': 8}
label_dict = {'other':0, 'TeCP': 1, 'TrCP': 2, 'TrAP': 3, 'PIP': 4, 'TeRP': 5}
 
word_dict = makeWordList(word_list)
pos_dict = makeWordList(pos_list)
chunk_dict = makeWordList(chunk_list)
d1_dict = makeWordList(d1_list)
d2_dict = makeWordList(d2_list)
type_dict = makeWordList(type_list)

print "word dictonary length", len(word_dict)
#Word Embedding
wv = readWordEmb(word_dict, wefile)		


# Mapping
W_train =  np.array(mapWordToId(word_list, word_dict))
P_train = np.array(mapWordToId(pos_list, pos_dict))
C_train = np.array(mapWordToId(chunk_list, chunk_dict))
d1_train = np.array(mapWordToId(d1_list, d1_dict))
d2_train = np.array(mapWordToId(d2_list, d2_dict))
T_train = np.array(mapWordToId(type_list,type_dict))

Y_t = mapLabelToId(sent_lables, label_dict)

Y_train = np.zeros((len(Y_t), len(label_dict)))
for i in range(len(Y_t)):
	Y_train[i][Y_t[i]] = 1.0


 
with open('filename.pickle', 'wb') as handle:
	pickle.dump(W_train, handle)
	pickle.dump(P_train, handle)
	pickle.dump(C_train, handle)
	pickle.dump(d1_train, handle)
	pickle.dump(d2_train, handle)
	pickle.dump(T_train, handle)
	pickle.dump(Y_train, handle)
	pickle.dump(wv, handle)
	pickle.dump(word_dict, handle)
	pickle.dump(pos_dict, handle)
	pickle.dump(chunk_dict, handle)
	pickle.dump(d1_dict, handle)
	pickle.dump(d2_dict, handle)
	pickle.dump(type_dict, handle)	 
	pickle.dump(label_dict, handle) 
"""

with open('filename.pickle', 'rb') as handle:
	W_train = pickle.load(handle)
	P_train = pickle.load(handle)
	C_train = pickle.load(handle)
	d1_train = pickle.load(handle)
	d2_train = pickle.load(handle)
	T_train = pickle.load(handle)
	Y_train = pickle.load(handle)
	wv = pickle.load(handle)
	word_dict= pickle.load(handle)
	pos_dict = pickle.load(handle)
	chunk_dict = pickle.load(handle)
	d1_dict = pickle.load(handle)
	d2_dict = pickle.load(handle)
	type_dict = pickle.load(handle)
	label_dict = pickle.load(handle)
"""

num_sample = len(Y_train)
seq_len = len(W_train[0])
fp = open("check_rv_t_p1_p2_pos_chunk_f46.txt",'w')


#vocabulary size
word_dict_size = len(word_dict)
pos_dict_size = len(pos_dict)
chunk_dict_size = len(chunk_dict)
d1_dict_size = len(d1_dict)
d2_dict_size = len(d2_dict)
type_dict_size = len(type_dict)
label_dict_size = len(label_dict)

shuffle_indices = np.random.permutation(np.arange(num_sample))
W_train =  W_train[shuffle_indices]
P_train =  P_train[shuffle_indices]
C_train =  C_train[shuffle_indices]
d1_train=  d1_train[shuffle_indices]
d2_train=  d2_train[shuffle_indices]
T_train =  T_train[shuffle_indices]
Y_train =  Y_train[shuffle_indices]

def test_step(W_te, P_te, C_te, d1_te, d2_te, T_te, Y_te):
	n = len(W_te)	 
	num = int(n/batch_size) + 1
	sample = []
	for batch_num in range(num):	
		start_index = batch_num*batch_size
		end_index = min((batch_num + 1) * batch_size, n)
		sample.append(range(start_index, end_index))
	acc = [] 
	pred = []
	for i in sample:
		a,p = cnn.test_step(W_te[i], P_te[i], C_te[i], d1_te[i], d2_te[i], T_te[i], Y_te[i])
#		acc.extend(a)
		pred.extend(p)
	return pred

kf = KFold(num_sample, n_folds=5)

for train, test in kf:
	W_tr, W_te = W_train[train], W_train[test]
	P_tr, P_te = P_train[train], P_train[test]
	C_tr, C_te = C_train[train], C_train[test]
	d1_tr, d1_te = d1_train[train], d1_train[test]
	d2_tr, d2_te = d2_train[train], d2_train[test]
	T_tr, T_te = T_train[train], T_train[test]
	Y_tr, Y_te = Y_train[train], Y_train[test]

 	cnn = CNN_Relation(label_dict_size, seq_len, word_dict_size, pos_dict_size, chunk_dict_size, d1_dict_size, d2_dict_size, type_dict_size, wv)	

 	num_train = len(W_tr)
	y_true_list = []
	y_pred_list = []
	num_epochs = 80
	N = 20
	batch_size=128
	num_batches_per_epoch = int(num_train/batch_size) + 1
	
	for j in range(num_epochs):		
		#Shuffling
		shuffle_indices = np.random.permutation(np.arange(num_train))
		W_tr =  W_tr[shuffle_indices]
		P_tr =  P_tr[shuffle_indices]
		C_tr = 	C_tr[shuffle_indices]
		d1_tr = d1_tr[shuffle_indices]
		d2_tr = d2_tr[shuffle_indices]
		T_tr = T_tr[shuffle_indices]
		Y_tr = Y_tr[shuffle_indices]
		sam=[]
		for batch_num in range(num_batches_per_epoch):	
			start_index = batch_num*batch_size
			end_index = min((batch_num + 1) * batch_size, num_train)
			sam.append(range(start_index, end_index))

		for rang in sam:
#			print "sunil"
			cnn.train_step(W_tr[rang], P_tr[rang], C_tr[rang], d1_tr[rang], d2_tr[rang], T_tr[rang], Y_tr[rang])

		if (j%N) == 0:
			pred = test_step(W_te, P_te, C_te, d1_te, d2_te, T_te, Y_te)			 
			print "test data size ", len(pred)
			y_true = np.argmax(Y_te, 1)
			y_pred = pred
			y_true_list.append(y_true)
			y_pred_list.append(y_pred)
		
	for y_true, y_pred in zip(y_true_list, y_pred_list):
 		fp.write(str(precision_score(y_true, y_pred,[1,2,3,4,5], average='weighted' )))
		fp.write('\t')
		fp.write(str(recall_score(y_true, y_pred, [1,2,3,4,5], average='weighted' )))
		fp.write('\t')
		fp.write(str(f1_score(y_true, y_pred, [1,2,3,4,5], average='weighted' )))
		fp.write('\t')
		fp.write('\n')
		 
	fp.write('\n')
	fp.write('\n')	
		
