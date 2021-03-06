#!/usr/bin/env python
import optparse
import sys
from collections import defaultdict
from IBM1 import *
from nltk.stem import SnowballStemmer
reload(sys)
sys.setdefaultencoding("latin-1")

optparser = optparse.OptionParser()
optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="Threshold for aligning with Dice's coefficient (default=0.5)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (opts.train, opts.french)
e_data = "%s.%s" % (opts.train, opts.english)

sys.stderr.write("Training with Dice's coefficient...")
bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]
eng_stemmer = SnowballStemmer("english")
frn_stemmer = SnowballStemmer("french")
stem_bitext = []

for (n,(f,e)) in enumerate(bitext):
  eng_stem = [frn_stemmer.stem(word) for word in e]
  frn_stem = [eng_stemmer.stem(word.decode("utf-8")) for word in f]
  stem_bitext.append([frn_stem, eng_stem])

bitext = stem_bitext

#now import ibm model 1
ibm_model1 = IBM1(bitext)

k = 5
theta = defaultdict(float)
ef_count = defaultdict(float)
f_count = defaultdict(float)
align_prob = defaultdict(float)
align_count = defaultdict(float)
align_count_given_i = defaultdict(float)
trans_prob = ibm_model1.get_theta()

#initialize theta_0
for(n,(f,e)) in enumerate(bitext):
  for (j,e_j) in enumerate(e):
    for (i,f_i) in enumerate(f):
      align_prob[(j,i,len(e),len(f))] = 1.0 / len(f)

for i in range(0,k):
  ef_count = defaultdict(float)
  f_count = defaultdict(float)
  align_count = defaultdict(float)
  align_count_given_i = defaultdict(float)
  Z = defaultdict(float) # again the normalization, but this time is a array for different e
  
  for (n,(f,e)) in enumerate(bitext):
    l = len(e)
    m = len(f)

    #E step: first step, count the normalization
    for (j,e_j) in enumerate(e):
      Z[e_j] = 0.0
      for (i,f_i) in enumerate(f):
        Z[e_j] += trans_prob[(e_j,f_i)] * align_prob[(j,i,l,m)]

    #E step: second step, get real count
    for (j,e_j) in enumerate(e):
      for (i,f_i) in enumerate(f):
        this_count = trans_prob[(e_j,f_i)] * align_prob[(j,i,l,m)]
        this_count_after_normalized = this_count / Z[e_j]
        ef_count[(e_j,f_i)] += this_count_after_normalized
        f_count[f_i] += this_count_after_normalized
        align_count[(j,i,l,m)] += this_count_after_normalized
        align_count_given_i[(j,l,m)] += this_count_after_normalized

  #M step: first step, update the translate probability
  for (k,(e_j,f_i)) in enumerate(trans_prob.keys()):
    trans_prob[(e_j,f_i)] = ef_count[(e_j,f_i)] / f_count[f_i]

  #M step: second step, update the alignment probability
  for(n,(f,e)) in enumerate(bitext):
    l = len(e)
    m = len(f)
    for (j,e_j) in enumerate(e):
      for (i,f_i) in enumerate(f):
        align_prob[(j,i,l,m)] = align_count[(j,i,l,m)] / align_count_given_i[(j,l,m)]

for (f, e) in bitext:
  l = len(e)
  m = len(f)
  for (j, e_j) in enumerate(e):
    best_p = 0
    best_i = 0
    for (i, f_i) in enumerate(f):
      curr_prob = trans_prob[(e_j,f_i)] * align_prob[(j,i,l,m)]
      if curr_prob >= best_p:
        best_p = curr_prob
        best_i = i
    sys.stdout.write("%i-%i " % (best_i,j))
  sys.stdout.write("\n")