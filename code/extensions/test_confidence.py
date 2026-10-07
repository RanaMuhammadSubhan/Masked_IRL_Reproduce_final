import unittest,numpy as np,torch
from confidence import empirical_relevance,soften
from objective import local_loss_chunks

class ConfidenceTests(unittest.TestCase):
    def test_analytic_cdf(self):
        reference=np.tile(np.arange(4)[:,None],(1,19));demo=np.zeros((10,19))
        np.testing.assert_allclose(empirical_relevance(reference,demo),.75,atol=0,rtol=0)
    def test_constant_and_equal_distributions(self):
        x=np.ones((10,19));np.testing.assert_array_equal(empirical_relevance(x,x),np.zeros(19))
        x=np.tile(np.arange(10)[:,None],(1,19));np.testing.assert_array_equal(empirical_relevance(x,x),np.zeros(19))
    def test_bounds_nonmutation_and_relevant_bits(self):
        rng=np.random.default_rng(12);ref=rng.normal(size=(300,19));demo=ref[:10].copy();before=ref.copy()
        d=empirical_relevance(ref,demo);b=np.array([0,1]*9+[0],dtype=float);old=b.copy();m=soften(b,d)
        self.assertTrue(np.all((d>=0)&(d<=1)) and np.all((m>=0)&(m<=1)))
        np.testing.assert_array_equal(m[b==1],np.ones(int(b.sum())))
        np.testing.assert_array_equal(ref,before);np.testing.assert_array_equal(b,old)
        np.testing.assert_allclose(1-m,(1-b)*(1-d),atol=1e-15)
    def test_not_an_oracle_clean_reuse(self):
        b=np.zeros(19);m=soften(b,np.ones(19)*.5)
        self.assertFalse(np.array_equal(b,m))
    def test_coordinate_scale_invariance(self):
        rng=np.random.default_rng(10);ref=rng.normal(size=(100,19));demo=ref[:10]
        np.testing.assert_allclose(empirical_relevance(ref,demo),empirical_relevance(ref*7+11,demo*7+11),atol=1e-15)
    def test_soft_analytic_scale(self):
        x=torch.zeros(2,3,4,dtype=torch.float64);n=torch.ones_like(x)*.25;m=torch.tensor([[0,.25,.5,1],[.75,1,0,.5]],dtype=torch.float64)
        v=sum(local_loss_chunks(lambda x,c:x.sum(-1),x,torch.zeros(2,1),m,n,2))
        torch.testing.assert_close(v,((1-m).sum()*3*.25/2))
    def test_soft_chunk_gradients(self):
        torch.manual_seed(9);x=torch.randn(3,4,5,dtype=torch.float64);n=torch.rand_like(x);m=torch.rand(3,5,dtype=torch.float64);c=torch.randn(3,2,dtype=torch.float64)
        net=torch.nn.Sequential(torch.nn.Linear(7,9),torch.nn.Tanh(),torch.nn.Linear(9,1)).double();reference=None
        for chunk in [1,7,10000]:
            net.zero_grad();value=0
            for loss in local_loss_chunks(lambda x,c:net(torch.cat([x,c],1)),x,c,m,n,chunk):value+=float(loss.detach());loss.backward()
            grad=torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in net.parameters()])
            if reference is None:reference=(value,grad.clone())
            else:self.assertAlmostEqual(value,reference[0],places=12);torch.testing.assert_close(grad,reference[1],rtol=1e-10,atol=1e-12)
if __name__=='__main__':unittest.main(verbosity=2)
