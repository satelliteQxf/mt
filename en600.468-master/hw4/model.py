import torch
import torch.nn as nn
from torch.autograd import Variable
import math

# TODO: Your implementation goes here
class RNNLM(nn.Module):
  def __init__(self, vocab_size):
    super(RNNLM, self).__init__()
    self.lookup = nn.Parameter(torch.Tensor(vocab_size, 32).uniform_())
    self.weight_x = nn.Parameter(torch.Tensor(32, 16).uniform_(-1.0/math.sqrt(32), 1.0/math.sqrt(32)))
    self.weight_h = nn.Parameter(torch.Tensor(16, 16).uniform_(-1.0/math.sqrt(16), 1.0/math.sqrt(16)))
    self.weight_o = nn.Parameter(torch.Tensor(16, vocab_size).uniform_(-1.0/math.sqrt(16), 1.0/math.sqrt(16)))

  def forward(self, input_batch):
    #X = self.lookup[input_batch.data,:]
    
    sequence_length = input_batch.size()[0]
    batch_length = input_batch.size()[1]
    X = torch.index_select(self.lookup, 0, input_batch.view(-1)).view(sequence_length, batch_length, 32)

    #print self.lookup.size()[0]
    output = Variable(torch.Tensor(sequence_length, batch_length, self.lookup.size()[0]), requires_grad = False)
    
    H = nn.Parameter(torch.Tensor(16).uniform_()).expand(batch_length, 16)
    # H_table = Variable(torch.zeros(batch_length, 16))
    H_table = Variable(torch.Tensor(sequence_length, batch_length, 16))
    # H_table = []


    bias_two = nn.Parameter(torch.randn(16), requires_grad = True)
    bias_three = nn.Parameter(torch.rand(16), requires_grad = True)
    
    for i in range(len(X)):

      input_x = X[i,:,:]
      H_tmp1 = input_x.mm(self.weight_x)# + bias_two
      H_tmp2 = H.mm(self.weight_h)# + bias_three
      H_sum = H_tmp1 + H_tmp2
      sigmoid = nn.Sigmoid()
      H_cur = sigmoid(H_sum)
      # H = H_cur
      # H_table.append(H_cur)
      H_table[i] = H_cur

    # outlayer_in = H_table.mm(self.weight_o)
    for i in range(sequence_length):
      outlayer_in = H_table[i].mm(self.weight_o) #+ bias_three
      # m = nn.LogSoftmax()
      output[i,:,:] = torch.log(self.softmax(outlayer_in, input_batch))

    return output

  def softmax(self, vector):
    vector = torch.exp(vector)
    vector_sum = vector.sum(dim=1)
    expaned_vector = vector_sum.view((vector_sum.size()[0], 1))
    return vector / expaned_vector
 
# TODO: Your implementation goes here
class BiRNNLM(nn.Module):
  def __init__(self, vocab_size):
    super(BiRNNLM, self).__init__()
    #self.lookup = nn.Parameter(torch.randn(vocab_size, 32), requires_grad = True)
    self.lookup   = nn.Parameter(torch.Tensor(vocab_size, 32).uniform_(-1.0/math.sqrt(32), 1.0/math.sqrt(32)))
    self.weight_xf = nn.Parameter(torch.Tensor(32, 8).uniform_(-1.0/math.sqrt(8), 1.0/math.sqrt(8)))
    self.weight_hf = nn.Parameter(torch.Tensor(8, 8).uniform_(-1.0/math.sqrt(8), 1.0/math.sqrt(8)))

    self.weight_xb = nn.Parameter(torch.Tensor(32, 8).uniform_(-1.0/math.sqrt(8), 1.0/math.sqrt(8)))
    self.weight_hb = nn.Parameter(torch.Tensor(8, 8).uniform_(-1.0/math.sqrt(8), 1.0/math.sqrt(8)))
    self.weight_o = nn.Parameter(torch.Tensor(16, vocab_size).uniform_(-1.0/math.sqrt(vocab_size), 1.0/math.sqrt(vocab_size)))
    self.H_f = nn.Parameter(torch.Tensor(8).uniform_(-1.0/math.sqrt(8), 1.0/math.sqrt(8)))
    self.H_b = nn.Parameter(torch.Tensor(8).uniform_(-1.0/math.sqrt(8), 1.0/math.sqrt(8)))
    # self.lookup = torch.rand(vocab_size, 32)
 
  def forward(self, input_batch):
    sequence_length = input_batch.size()[0]
    batch_length = input_batch.size()[1]
    X = torch.index_select(self.lookup, 0, input_batch.view(-1)).view(sequence_length, batch_length, 32)
    Hf_pre = self.H_f.expand(batch_length, 8)
    Hb_pre = self.H_b.expand(batch_length, 8)

    output = Variable(torch.Tensor(sequence_length, batch_length, self.lookup.size()[0]), requires_grad = False)
    
    H_f_table = Variable(torch.Tensor(sequence_length, batch_length, 8))
    H_b_table = Variable(torch.Tensor(sequence_length, batch_length, 8))

    bias_two_f1 = nn.Parameter(torch.Tensor(8).uniform_(-1.0/math.sqrt(8), 1.0/math.sqrt(8)), requires_grad = True)
    bias_two_f2 = nn.Parameter(torch.Tensor(8).uniform_(-1.0/math.sqrt(8), 1.0/math.sqrt(8)), requires_grad = True)
    bias_two_b1 = nn.Parameter(torch.Tensor(8).uniform_(-1.0/math.sqrt(8), 1.0/math.sqrt(8)), requires_grad = True)
    bias_two_b2 = nn.Parameter(torch.Tensor(8).uniform_(-1.0/math.sqrt(8), 1.0/math.sqrt(8)), requires_grad = True)

    bias_three = nn.Parameter(torch.Tensor(self.lookup.size()[0]).uniform_(-1.0/math.sqrt(self.lookup.size()[0]), 1.0/math.sqrt(self.lookup.size()[0])), requires_grad = True)
    for i in range(sequence_length):

      input_x = X[i,:,:]
      H_f_table[i] = Hf_pre
      #forward
      H_tmp1 = input_x.mm(self.weight_xf) + bias_two_f1
      H_tmp2 = Hf_pre.mm(self.weight_hf) + bias_two_f2
      H_sum = H_tmp1 + H_tmp2 #+ bias_two_f
      H_f_cur = torch.tanh(H_sum)
      Hf_pre = H_f_cur




    for i in range(sequence_length-1,-1, -1):
      input_x = X[i,:,:]
      H_b_table[i] = Hb_pre
      #backward
      H_tmp1 = input_x.mm(self.weight_xb) + bias_two_b1
      H_tmp2 = Hb_pre.mm(self.weight_hb) + bias_two_b2
      H_sum = H_tmp1 + H_tmp2 #+ bias_two_b
      H_b_cur = torch.tanh(H_sum)
      Hb_pre = H_b_cur

    for i in range(len(H_f_table)):
      H_i = torch.cat((H_f_table[i],H_b_table[i]),1)
      outlayer_in = H_i.mm(self.weight_o) + bias_three
      m = nn.LogSoftmax()
      output[i,:,:] = m(outlayer_in)

    return output