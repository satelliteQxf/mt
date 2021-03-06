#!/usr/bin/env python
import optparse
import sys
from collections import defaultdict
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


k = 5
theta = defaultdict(float)
ef_count = defaultdict(float)
f_count = defaultdict(float)

#initialize theta_0
ef_count = defaultdict(float)
f_count = defaultdict(float)
for(n,(f,e)) in enumerate(bitext):
  for e_j in set(e):
    for f_i in set(f):
      ef_count[(e_j,f_i)] += 1
      f_count[f_i] += 1
for (e_j,f_i) in ef_count:
  theta[(e_j,f_i)] = ef_count[(e_j,f_i)] / f_count[f_i]
print theta
exit()

for i in range(0,k):
  #E step
  ef_count = defaultdict(float)
  f_count = defaultdict(float)
  for (n,(f,e)) in enumerate(bitext):
    for e_j in set(e):
      Z = 0
      for f_i in set(f):
        Z += theta[(e_j,f_i)]
      for f_i in set(f):
        c = 1.0 * theta[(e_j,f_i)] / Z
        ef_count[(e_j,f_i)] += c
        f_count[f_i] += c

  #M step
  for (k,(e_j,f_i)) in enumerate(ef_count.keys()):
    theta[(e_j,f_i)] = ef_count[(e_j,f_i)] / f_count[f_i]


for (f, e) in bitext:
  for (j, e_j) in enumerate(e):
    best_p = 0
    best_i = 0
    for (i, f_i) in enumerate(f):
      if theta[(e_j,f_i)] >= best_p:
        best_p = theta[(e_j,f_i)]
        best_i = i
    sys.stdout.write("%i-%i " % (best_i,j))
  sys.stdout.write("\n")