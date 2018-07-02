from fastai.conv_learner import *
from utils import get_opt_fn, get_phases, log_msg
from callbacks import *
import fire

def main_train(lr, moms, wd, wd_loss, opt_fn, bs, cyc_len, beta2, amsgrad, div, pct, lin_end, tta, div_lr, fname):
    """
    Trains a Language Model

    lr (float): maximum learning rate
    moms (float/tuple): value of the momentum/beta1. If tuple, cyclical momentums will be used
    wd (float): weight decay to be used
    wd_loss (bool): weight decay computed inside the loss if True (l2 reg) else outside (true wd)
    opt_fn (optimizer): name of the optim function to use (should be SGD, RMSProp or Adam)
    bs (int): batch size
    cyc_len (int): length of the cycle
    beta2 (float): beta2 parameter of Adam or alpha parameter of RMSProp
    amsgrad (bool): for Adam, sues amsgrad or not
    div (float): value to divide the maximum learning rate by
    pct (float): percentage to leave for the annealing at the end
    lin_end (bool): if True, the annealing phase goes from the minimum lr to 1/100th of it linearly
                    if False, uses a cosine annealing to 0
    tta (bool): if True, uses Test Time Augmentation to evaluate the model
    div_lr (float): number to divide the diffential lrs
    """
    arch, sz = resnet50, 224
    PATH = Path("data/cars/")
    val_idxs = list(range(8144, 16185))
    tfms =tfms_from_model(arch, sz, aug_tfms = transforms_side_on, max_zoom=1.05)
    data = ImageClassifierData.from_csv(PATH, '', PATH/'annots.csv', bs, tfms, val_idxs=val_idxs)
    learn = ConvLearner.pretrained(arch, data, metrics=[accuracy])
    mom = moms[0] if isinstance(moms, Iterable) else moms
    opt_fn = get_opt_fn(opt_fn, mom, beta2, amsgrad)
    learn.opt_fn = opt_fn
    nbs = [cyc_len[0] * (1-pct) / 2, cyc_len[0] * (1-pct) / 2, cyc_len[0] * pct]
    phases = get_phases(lr[0], moms, opt_fn, div, list(nbs), wd, lin_end, wd_loss)
    learn.fit_opt_sched(phases, callbacks=[LogResults(learn, fname)])
    learn.unfreeze()
    lrs = np.array([lr[1]/(div_lr**2), lr[1]/div_lr, lr[1]])
    nbs = [cyc_len[1] * (1-pct) / 2, cyc_len[1] * (1-pct) / 2, cyc_len[1] * pct]
    phases = get_phases(lrs, moms, opt_fn, div, list(nbs), wd, lin_end, wd_loss)
    learn.fit_opt_sched(phases, callbacks=[LogResults(learn, fname)])
    if tta:
        preds, targs = learn.TTA()
        probs = np.exp(preds)
        probs = np.mean(probs,0)
        acc = learn.metrics[0](V(probs), V(targs))
        loss = learn.crit(V(np.log(probs)), V(targs)).item()
        log_msg(open(fname, 'a'), f'Final loss: {loss}, Final accuracy: {acc}')

def train_lm(lr, moms=(0.95,0.85), wd=1.2e-6, wd_loss=True, opt_fn='Adam', bs=128, cyc_len=(20,40), beta2=0.99, amsgrad=False,
             div=10, pct=0.1, lin_end=True, tta=False, div_lr=3, name='', cuda_id=0, nb_exp=1):
    """
    Launches the trainings.

    See main_train for the description of all the arguments.
    name (string): name to be added to the log file
    cuda_id (int): index of the GPU to use
    nb_exp (int): number of experiments to run in a row
    """
    torch.cuda.set_device(cuda_id)
    init_text = f'{name}_{cuda_id}' + '\n'
    init_text += f'lr: {lr}; moms: {moms}; wd: {wd}; wd_loss: {wd_loss}; opt_fn: {opt_fn}; bs: {bs}; cyc_len: {cyc_len};'
    init_text += f'beta2: {beta2}; amsgrad: {amsgrad}; div: {div}; pct: {pct}; lin_end: {lin_end}; tta: {tta}; div_lr: {div_lr}'
    fname = f'logs_{name}_{cuda_id}.txt'
    log_msg(open(fname, 'w'), init_text)
    for i in range(nb_exp):
        log_msg(open(fname, 'a'), '\n' + f'Experiment {i+1}')
        main_train(lr, moms, wd, wd_loss, opt_fn, bs, cyc_len, beta2, amsgrad, div, pct, lin_end, tta, div_lr, fname)

if __name__ == '__main__': fire.Fire(train_lm)