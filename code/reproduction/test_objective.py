import unittest,torch
from objective import perturbation_chunks,local_loss_chunks,released_cross_sample

class ObjectiveTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(77)
        self.x=torch.randn(3,4,5,dtype=torch.float64)
        self.m=torch.tensor([[1,0,1,1,1],[0,1,0,1,1],[1,1,1,1,0]],dtype=torch.float64)
        self.z=torch.rand_like(self.x);self.context=torch.randn(3,2,dtype=torch.float64)
    def test_coordinate_and_bounds(self):
        total=0
        for orig,pert,ib,w in perturbation_chunks(self.x,self.m,self.z,3):
            delta=pert-orig
            self.assertTrue(torch.all((delta!=0).sum(1)==1))
            self.assertTrue(torch.all(delta*self.m[ib]==0))
            self.assertTrue(torch.all((delta>=0)&(delta<=1)))
            total+=len(orig)
        self.assertEqual(total,16)
    def test_no_mutation(self):
        old=self.x.clone();list(perturbation_chunks(self.x,self.m,self.z,2));self.assertTrue(torch.equal(old,self.x))
    def test_analytic_sum_scaling(self):
        fn=lambda x,c:x.sum(-1)
        value=sum(local_loss_chunks(fn,self.x,self.context,self.m,self.z,2))
        expected=(self.z*(1-self.m[:,None,:])).sum()/3
        torch.testing.assert_close(value,expected)
    def test_chunked_values_and_gradients(self):
        net=torch.nn.Sequential(torch.nn.Linear(7,6),torch.nn.Tanh(),torch.nn.Linear(6,1)).double()
        ref=None
        for chunk in [1,3,10000]:
            net.zero_grad();value=0.
            for loss in local_loss_chunks(lambda x,c:net(torch.cat([x,c],1)),self.x,self.context,self.m,self.z,chunk):
                value+=float(loss.detach());loss.backward()
            grads=torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in net.parameters()])
            if ref is None:ref=(value,grads.clone())
            else:
                self.assertAlmostEqual(value,ref[0],places=12);torch.testing.assert_close(grads,ref[1],rtol=1e-10,atol=1e-12)
    def test_no_time_cancellation(self):
        x=torch.tensor([[[1.],[-1.]]],dtype=torch.float64);z=torch.ones_like(x)*.5
        local=sum(local_loss_chunks(lambda x,c:x.square().sum(-1),x,torch.zeros(1,1),torch.zeros(1,1),z,100))
        global_difference=((x+z).square().sum()-x.square().sum()).abs()
        self.assertAlmostEqual(float(local),2.);self.assertAlmostEqual(float(global_difference),.5)
    def test_released_estimator_value_gradient(self):
        a=torch.tensor([.2,.7,-.3],dtype=torch.float64,requires_grad=True);b=torch.tensor([.1,1.],dtype=torch.float64,requires_grad=True)
        orig=a.mean()+torch.log((torch.exp(-a[:,None])/torch.softmax(-a.detach(),0)).mean()+(torch.exp(-b[:,None])/torch.softmax(-b.detach(),0)).mean())
        stable=released_cross_sample(a,b)
        torch.testing.assert_close(orig,stable)
        g1=torch.autograd.grad(orig,(a,b),retain_graph=True);g2=torch.autograd.grad(stable,(a,b))
        for x,y in zip(g1,g2):torch.testing.assert_close(x,y)
    def test_large_cost_stability(self):
        a=torch.tensor([1000.,-1000.],requires_grad=True);v=released_cross_sample(a,a);v.backward()
        self.assertTrue(torch.isfinite(v) and torch.isfinite(a.grad).all())
    def test_corruption_contract(self):
        import numpy as np
        from train_v3 import corrupt_mask
        mask=np.array([0,1]*9+[0],dtype=float);before=mask.copy();theta=[1,0,-1,0,1]
        x,meta=corrupt_mask(mask,.2,12345,theta);y,other=corrupt_mask(mask,.2,12345,theta)
        self.assertTrue(np.array_equal(x,y) and meta==other and np.array_equal(mask,before))
        self.assertEqual(x.shape,(19,));self.assertTrue(set(x)<={0.,1.})
        self.assertTrue(np.array_equal(corrupt_mask(mask,0,12345,theta)[0],mask))
        self.assertTrue(np.array_equal(corrupt_mask(mask,1,12345,theta)[0],1-mask))
        self.assertAlmostEqual(meta['realized_fraction'],float(np.mean(x!=mask)))

if __name__=='__main__':unittest.main(verbosity=2)
