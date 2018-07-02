from fastai.text import *
from utils import get_opt_fn, get_phases, log_msg
from callbacks import *
from lm_val_fns import *
import fire

EOS = '<eos>'
PATH = Path('data/wikitext/')

def read_file(filename):
    """
    Reads the file in filemane and prepares the tokens.
    """
    tokens = []
    with open(PATH/filename) as f:
        for line in f: 
            tokens.append(line.split() + [EOS])
    return np.array(tokens)

def main_train(lr, moms, wd, wd_loss, opt_fn, bs, bptt, drops, beta2, amsgrad, div, nbs, lin_end, clip, alpha, beta, qrnn, bias, fname):
    """
    Trains a Language Model

    lr (float): maximum learning rate
    moms (float/tuple): value of the momentum/beta1. If tuple, cyclical momentums will be used
    wd (float): weight decay to be used
    wd_loss (bool): weight decay computed inside the loss if True (l2 reg) else outside (true wd)
    opt_fn (optimizer): name of the optim function to use (should be SGD, RMSProp or Adam)
    bs (int): batch size
    bptt (int): bptt parameter for the training
    drops (np.array of float): dropouts to use
    beta2 (float): beta2 parameter of Adam or alpha parameter of RMSProp
    amsgrad (bool): for Adam, sues amsgrad or not
    div (float): value to divide the maximum learning rate by
    nbs (list): number of epochs for each phase (ascending, constant if len==4, descending, annealing)
    lin_end (bool): if True, the annealing phase goes from the minimum lr to 1/100th of it linearly
                    if False, uses a cosine annealing to 0
    clip (float): value of gradient clipping to use
    alpha (float): alpha parameter for the AR regularization function
    beta (float): beta parameter for the AR regularization function
    qrnn (bool): if True, will use QRNNs instead of LSTMs
    bias (bool): if True, the decoder in the LM has bias
    """
    trn_tok = read_file('wiki.train.tokens')
    val_tok = read_file('wiki.valid.tokens')
    tst_tok = read_file('wiki.test.tokens')
    cnt = Counter(word for sent in trn_tok for word in sent)
    itos = [o for o,c in cnt.most_common()]
    itos.insert(0,'_pad_')
    vocab_size = len(itos)
    if qrnn: em_sz, nh, nl = 400, 1550, 4
    else: em_sz, nh, nl = 400, 1150, 3
    stoi = collections.defaultdict(lambda : 5, {w:i for i,w in enumerate(itos)})
    trn_ids = np.array([([stoi[w] for w in s]) for s in trn_tok])
    val_ids = np.array([([stoi[w] for w in s]) for s in val_tok])
    tst_ids = np.array([([stoi[w] for w in s]) for s in tst_tok])
    trn_dl = LanguageModelLoader(np.concatenate(trn_ids), bs, bptt)
    val_dl = LanguageModelLoader(np.concatenate(val_ids), bs, bptt)
    md = LanguageModelData(PATH, 0, vocab_size, trn_dl, val_dl, bs=bs, bptt=bptt)
    defaut_drops = np.array([0.6,0.4,0.5,0.1,0.2]) if not qrnn else np.array([0.4,0.4,0.1,0.1,0.2])
    drops = defaut_drops if drops is None else np.array(list(drops))
    mom = moms[0] if isinstance(moms, Iterable) else moms
    opt_fn = get_opt_fn(opt_fn, mom, beta2, amsgrad)
    learner= md.get_model(opt_fn, em_sz, nh, nl, dropouti=drops[0], dropout=drops[1], wdrop=drops[2], 
                          dropoute=drops[3], dropouth=drops[4], qrnn=qrnn, bias=bias)
    learner.metrics = [accuracy]
    learner.clip = clip
    learner.reg_fn = partial(seq2seq_reg, alpha=alpha, beta=beta)
    learner.unfreeze()
    phases = get_phases(lr, moms, opt_fn, div, list(nbs), wd, lin_end, wd_loss)
    learner.fit_opt_sched(phases, callbacks=[LogResults(learner, fname)])
    val_los, val_pp = my_validate(learner.model, np.concatenate(val_ids))
    log_msg(open(fname, 'a'), f'Validation loss: {val_los}, Validation perplexity: {val_pp}')
    tst_los, tst_pp = my_validate(learner.model, np.concatenate(tst_ids))
    log_msg(open(fname, 'a'), f'Test loss: {tst_los}, Test perplexity: {tst_pp}')
    cache_vlos, cache_vpp = my_cache_pointer(learner.model, np.concatenate(val_ids), vocab_size)
    log_msg(open(fname, 'a'), f'Cache validation loss: {cache_vlos}, Cache validation perplexity: {cache_vpp}')
    cache_tlos, cache_tpp = my_cache_pointer(learner.model, np.concatenate(tst_ids), vocab_size)
    log_msg(open(fname, 'a'), f'Cache test loss: {cache_tlos}, Cache test perplexity: {cache_tpp}')

def train_lm(lr, moms=(0.8,0.7), wd=1.2e-6, wd_loss=True, opt_fn='Adam', bs=100, bptt=70, drops=None, beta2=0.99, amsgrad=False,
             div=10, nbs=(7.5,37.5,37.5,7.5), lin_end=False, clip=0.12, alpha=2, beta=1, qrnn=False, bias=True, 
             name='', cuda_id=0, nb_exp=1):
    """
    Launches the trainings.

    See main_train for the description of all the arguments.
    name (string): name to be added to the log file
    cuda_id (int): index of the GPU to use
    nb_exp (int): number of experiments to run in a row
    """
    torch.cuda.set_device(cuda_id)
    init_text = f'{name}_{cuda_id}' + '\n'
    init_text += f'lr: {lr}; moms: {moms}; wd: {wd}; wd_loss: {wd_loss}; opt_fn: {opt_fn}; bs: {bs}; bptt: {bptt}; drops: {drops};'
    init_text += f'beta2: {beta2}; amsgrad: {amsgrad}; div: {div}; nbs: {nbs}; lin_end: {lin_end}; clip: {clip}; alpha: {alpha}; beta: {beta}; '
    init_text += f'qrnn: {qrnn}; bias: {bias}'
    fname = f'logs_{name}_{cuda_id}.txt'
    log_msg(open(fname, 'w'), init_text)
    for i in range(nb_exp):
        log_msg(open(fname, 'a'), '\n' + f'Experiment {i+1}')
        main_train(lr, moms, wd, wd_loss, opt_fn, bs, bptt, drops, beta2, amsgrad, div, nbs, lin_end, clip, alpha, beta, qrnn, bias, fname)

if __name__ == '__main__': fire.Fire(train_lm)