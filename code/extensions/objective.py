"""Local manuscript mask estimator and algebraically stable released IRL estimator."""
import math
import torch

def perturbation_chunks(states, masks, noise, chunk_size):
    """Yield individual coordinate perturbations; never mutate the input tensor.

    states/noise: (B,T,D), masks: (B,D). Sums over T,D; batch mean is applied below.
    Also supports explicitly declared soft masks for a later independent variant.
    """
    b,t,d=states.shape
    assert masks.shape==(b,d) and noise.shape==states.shape and chunk_size>0
    assert torch.all((masks>=0)&(masks<=1))
    bi,dj=torch.where(masks<1)
    bi=bi.repeat_interleave(t);dj=dj.repeat_interleave(t);ti=torch.arange(t,device=states.device).repeat(len(bi)//t)
    for start in range(0,len(bi),chunk_size):
        ib,j,it=bi[start:start+chunk_size],dj[start:start+chunk_size],ti[start:start+chunk_size]
        original=states[ib,it];perturbed=original.clone()
        perturbed[torch.arange(len(ib),device=states.device),j]+=noise[ib,it,j]
        yield original,perturbed,ib,(1-masks[ib,j])

def local_loss_chunks(cost_fn,states,context,masks,noise,chunk_size=4096):
    for original,perturbed,ib,w in perturbation_chunks(states,masks,noise,chunk_size):
        yield ((cost_fn(perturbed,context[ib])-cost_fn(original,context[ib])).reshape(-1).abs()*w).sum()/len(states)

def released_cross_sample(demo_cost,train_cost):
    """Exactly the released cross-sample ratio MEAN, in stable log space.

    mean_ij exp(-c_i)/p_j = mean_i exp(-c_i)*mean_j 1/p_j.
    The detached softmax weights and cross-sample gradient semantics are preserved.
    """
    def logterm(c):
        c=c.reshape(-1);lp=torch.log_softmax(-c.detach(),dim=0)
        return torch.logsumexp(-c,0)-math.log(len(c))+torch.logsumexp(-lp,0)-math.log(len(c))
    return demo_cost.mean()+torch.logaddexp(logterm(train_cost),logterm(demo_cost))
