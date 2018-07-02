from fastai.text import *

class TextReader():
    """ Returns a language model iterator that iterates through batches that are of length N(bptt,5)
    The first batch returned is always bptt+25; the max possible width.  This is done because of they way that pytorch
    allocates cuda memory in order to prevent multiple buffers from being created as the batch width grows.
    """
    def __init__(self, nums, bptt, backwards=False):
        self.bptt,self.backwards = bptt,backwards
        self.data = self.batchify(nums)
        self.i,self.iter = 0,0
        self.n = len(self.data)

    def __iter__(self):
        self.i,self.iter = 0,0
        while self.i < self.n-1 and self.iter<len(self):
            res = self.get_batch(self.i, self.bptt)
            self.i += self.bptt
            self.iter += 1
            yield res

    def __len__(self): return self.n // self.bptt 

    def batchify(self, data):
        data = np.array(data)[:,None]
        if self.backwards: data=data[::-1]
        return T(data)

    def get_batch(self, i, seq_len):
        source = self.data
        seq_len = min(seq_len, len(source) - 1 - i)
        return source[i:i+seq_len], source[i+1:i+1+seq_len].view(-1)

def my_validate(model, source, bptt=2000):
    """
    Return the validation loss and perplexity of a model

    model: model to test
    source: data on which to evaluate the mdoe
    bptt: bptt for this evaluation (doesn't change the result, only the speed)
    """
    data_source = TextReader(source, bptt)
    model.eval()
    model.reset()
    total_loss = 0.
    for inputs, targets in tqdm(data_source):
        outputs, raws, outs = model(V(inputs))
        p_vocab = F.softmax(outputs,1)
        for i, pv in enumerate(p_vocab):
            targ_pred = pv[targets[i]]
            total_loss -= torch.log(targ_pred.detach())
    mean = total_loss / (bptt * len(data_source))
    return mean, np.exp(mean)

def one_hot1(vec, size):
    a = torch.zeros(len(vec), size)
    for i,v in enumerate(vec):
        a[i,v] = 1.
    return V(a)

def my_cache_pointer(model, source, vocab_size, scale=1, theta = 0.662, lambd = 0.1279, window=3785, bptt=2000):
    data_source = TextReader(source, bptt)
    model.eval()
    model.reset()
    total_loss = 0.
    targ_history = None
    hid_history = None
    for inputs, targets in tqdm(data_source):
        outputs, raws, outs = model(V(inputs))
        p_vocab = F.softmax(outputs * scale,1)
        start = 0 if targ_history is None else targ_history.size(0)
        targ_history = one_hot1(targets, vocab_size) if targ_history is None else torch.cat([targ_history, one_hot1(targets, vocab_size)])
        hiddens = raws[-1].squeeze() #results of the last layer + remove the batch size.
        hid_history = hiddens * scale if hid_history is None else torch.cat([hid_history, hiddens * scale])
        for i, pv in enumerate(p_vocab):
            #Get the cached values
            p = pv
            if start + i > 0:
                targ_cache = targ_history[:start+i] if start + i <= window else targ_history[start+i-window:start+i]
                hid_cache = hid_history[:start+i] if start + i <= window else hid_history[start+i-window:start+i]
                all_dot_prods = torch.mv(theta * hid_cache, hiddens[i])
                exp_dot_prods = F.softmax(all_dot_prods).unsqueeze(1)
                p_cache = (exp_dot_prods.expand_as(targ_cache) * targ_cache).sum(0).squeeze()
                p = (1-lambd) * pv + lambd * p_cache
            targ_pred = p[targets[i]]
            total_loss -= torch.log(targ_pred.detach())
        targ_history = targ_history[-window:]
        hid_history = hid_history[-window:]
    mean = total_loss / (bptt * len(data_source))
    return mean, np.exp(mean)