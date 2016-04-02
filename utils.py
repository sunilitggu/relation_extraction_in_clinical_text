import re
import numpy as np
import nltk

#from helper import *
#from practnlptools.tools import Annotator
#ann=Annotator()
#from geniatagger import GeniaTagger
#tagger = GeniaTagger('/home/sunilitggu/python_pack/geniatagger-3.0.2/geniatagger')


def readData(ftrain):
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
	sent_lengths.append(len(sent.split()))
	sent_names.append(name)
	sent_contents.append(sent.split())
	
	m = re.match(r"\(\[['\"](.*)['\"], (\d*), (\d*), '(.*)'\], \[['\"](.*)['\"], (\d*), (\d*), '(.*)'\]\)", entities.strip())
	if m :
		e1   = m.group(1)
		e1_s = int(m.group(2))
		e1_e = int(m.group(3))
		e1_t = m.group(4)

		e2   = m.group(5)
		e2_s = int(m.group(6))
		e2_e = int(m.group(7))
		e2_t = m.group(8)
		if(e1_s < e2_s):
			entity1_list.append([e1,e1_s,e1_e,e1_t])
			entity2_list.append([e2,e2_s,e2_e,e2_t])
		else:
			entity1_list.append([e2,e2_s,e2_e,e2_t])
			entity2_list.append([e1,e1_s,e1_e,e1_t])
#		print e1,e2
	else:
		print "Error in readign", entities.strip()
#		exit(0)
	
	ma = re.match(r"\[['\"](.*)['\"], '(.*)', ['\"](.*)['\"]\]", relation.strip())
	if(ma):
		lable = ma.group(2)		
	elif relation == '[0]':
		lable = 'other'
	else:
		print "Error in reading", relation
		exit(0)
#	print lable
	sent_lables.append(lable)
  return sent_contents,entity1_list, entity2_list, sent_lables 

def makePosFeatures(sent_contents):
	pos_tag_list = []
	for sent in sent_contents:
#		tags = tagger.parse(sent)
#		sent_t, sent_o, sent_pos, sent_chunk, sent_bio = zip(*tags)
		pos_tag = nltk.pos_tag(sent)
		pos_tag = zip(*pos_tag)[1]
#		print pos_tag
		pos_tag_list.append(pos_tag)		
	return pos_tag_list 

def mapWordToId(sent_contents, word_dict):
	T = []
	for sent in sent_contents:
		t = []
		for w in sent:
			t.append(word_dict[w])
		T.append(t)
	return T

def makeDistanceFeatures(sent_contents, entity1_list, entity2_list):
	d1_list = []
	d2_list = []
	type_list = []
	for sent, e1_part, e2_part in zip(sent_contents, entity1_list, entity2_list):
		entity1, s1, e1, t1 = e1_part
		entity1, s2, e2, t2 = e2_part
		maxl = len(sent)

		d1 = []		
		for i in range(maxl):
			if i < s1 :
				d1.append(str(i - s1))
			elif i > e1 :
				d1.append(str(i - e1 ))
			else:
				d1.append('0')
		d1_list.append(d1)

		d2 = []
		for i in range(maxl):
			if i < s2 :
				d2.append(str(i - s2))
			elif i > s2 :
				d2.append(str(i - s2))
			else:
				d2.append('0')		
		d2_list.append(d2)
		
		t = []
		for i in range(maxl):
			t.append('Out')
		for i in range(s1,e1+1):
			if(t1 == 'problem'):
				t[i] = 'Prob'
			elif(t1 == 'treatment'):
				t[i] = 'Treat'
			elif(t1 == 'test'):
				t[i] = 'Test'

		for i in range(s2, e2+1):
			if(t2 == 'problem'):
				t[i] = 'Prob'
			elif(t2 == 'treatment'):
				t[i] = 'Treat'
			elif(t2 == 'test'):
				t[i] = 'Test'
		type_list.append(t)
	return d1_list, d2_list, type_list

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
	wl['unkown'] = 0	
	for w,f in wf.iteritems():
		i += 1
		wl[w] = i
	return wl

def makeRelList(rel_list):
	rel_dict = {}
 	for rel in rel_list:
		if rel in rel_dict:
			rel_dict[rel] += 1
		else:
			rel_dict[rel] = 0
	wl = {}
	i = 0
 	for w,f in rel_dict.iteritems():
		wl[w] = i
		i += 1
	return wl 

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

def mapLabelToId(sent_lables, label_dict):
	return [label_dict[label] for label in sent_lables]

