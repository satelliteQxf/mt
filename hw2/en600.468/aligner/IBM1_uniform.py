#!/usr/bin/env python
import optparse
import sys
from collections import defaultdict
from nltk.stem import SnowballStemmer
reload(sys)
sys.setdefaultencoding("latin-1")


class IBM1:

  def __init__(self, bitext):
    self.bitext = bitext
    self.theta = self.train()


  def train(self): # return theta
    k = 10
    theta = defaultdict(float)


    #initialize theta_0
    ef_count = defaultdict(float)
    f_count = defaultdict(float)
    for(n,(f,e)) in enumerate(self.bitext):
      for e_j in set(e):
        for f_i in set(f):
          theta[(e_j,f_i)] = 1 / len(f)

    for i in range(0,k):
      #E step
      ef_count = defaultdict(float)
      f_count = defaultdict(float)
      for (n,(f,e)) in enumerate(self.bitext):
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

    return theta

  def get_theta(self):
    return self.theta



  def print_result(self):
    for (f, e) in self.bitext:
      for (j, e_j) in enumerate(e):
        best_p = 0
        best_i = 0
        for (i, f_i) in enumerate(f):
          if self.theta[(e_j,f_i)] >= best_p:
            best_p = self.theta[(e_j,f_i)]
            best_i = i
        sys.stdout.write("%i-%i " % (best_i,j))
      sys.stdout.write("\n")
