"""Demonstration-derived soft irrelevance weights; no oracle correction or test data."""
import numpy as np

def empirical_relevance(reference_means,demo_means):
    reference=np.asarray(reference_means,dtype=float);demo=np.asarray(demo_means,dtype=float)
    assert reference.ndim==demo.ndim==2 and reference.shape[1]==demo.shape[1]==19
    assert len(reference)>0 and len(demo)==10 and np.isfinite(reference).all() and np.isfinite(demo).all()
    scores=[]
    for j in range(19):
        r=np.sort(reference[:,j]);d=np.sort(demo[:,j]);grid=np.unique(np.concatenate([r,d]))
        distance=np.max(np.abs(np.searchsorted(r,grid,side='right')/len(r)-np.searchsorted(d,grid,side='right')/len(d)))
        scores.append(float(distance))
    return np.asarray(scores)

def soften(binary_mask,relevance):
    b=np.asarray(binary_mask,dtype=float);d=np.asarray(relevance,dtype=float)
    assert b.shape==d.shape==(19,) and set(b)<={0.,1.} and np.all((d>=0)&(d<=1))
    return b+(1-b)*d
