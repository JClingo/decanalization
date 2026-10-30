"""PPO for the option policy (policy_agent.py). Run with the torch environment:
  python ppo_train.py HRL|NF [iterations]
Training world: reliable-opportunity degrading world, no intervention, seeds 10000+.
Actors are numpy copies of the network (multiprocessing); the learner is torch.
"""
import json,sys,time
from multiprocessing import Pool
import numpy as np,torch,torch.nn as nn
from mapagent import MapParams
from world import phase_dose
from policy_agent import OptionTrial,OBS,OPTIONS,DECISION_TICKS,GAMMA_TICK
from settings import WORLD

GAMMA=GAMMA_TICK**DECISION_TICKS;LAM=0.95;TRIALS=16;EPOCHS=4;MINI=2048;CLIP=0.2;LR=3e-4;ENT=0.01

class Net(nn.Module):
    def __init__(self):
        super().__init__();self.l1=nn.Linear(OBS,64);self.l2=nn.Linear(64,64);self.out=nn.Linear(64,OPTIONS+1)
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
    def forward(self,x):h=torch.tanh(self.l2(torch.tanh(self.l1(x))));o=self.out(h);return o[...,:OPTIONS],o[...,OPTIONS]
    def numpy(self):return [t.detach().numpy().T.copy() if t.dim()==2 else t.detach().numpy().copy() for t in
                            (self.l1.weight,self.l1.bias,self.l2.weight,self.l2.bias,self.out.weight,self.out.bias)]

def rollout(args):
    family,w,seed,world=args;p=MapParams(**world)
    t=OptionTrial(p,seed,phase_dose(p,'CCC'),family,w,record=True).run()
    S=t.log['samples'];return t.trans,float(np.mean([abs(s[3]) for s in S])),float(np.mean([s[3] for s in S]))

def gae(tr):
    adv=np.zeros(len(tr));last=0.
    for i in reversed(range(len(tr))):
        nv=tr[i+1]['v'] if i+1<len(tr) else 0.;delta=tr[i]['reward']+GAMMA*nv-tr[i]['v'];last=delta+GAMMA*LAM*last;adv[i]=last
    return adv,adv+np.array([x['v'] for x in tr])

if __name__=='__main__':
    family=sys.argv[1];iters=int(sys.argv[2]) if len(sys.argv)>2 else 40;seed=int(sys.argv[5]) if len(sys.argv)>5 else 0;torch.manual_seed(1234+seed)
    tag=sys.argv[3] if len(sys.argv)>3 else '';world=dict(WORLD,**(json.loads(sys.argv[4]) if len(sys.argv)>4 else {}))
    net=Net();opt=torch.optim.Adam(net.parameters(),lr=LR);curve=[]
    with Pool() as pool:
        for it in range(iters):
            t0=time.time();w=net.numpy()
            res=pool.map(rollout,[(family,w,10000+100000*seed+it*TRIALS+j,world) for j in range(TRIALS)])
            obs,act,logp,adv,ret=[],[],[],[],[]
            for tr,_,_ in res:
                a,r=gae(tr);obs+= [x['obs'] for x in tr];act+=[x['a'] for x in tr];logp+=[x['logp'] for x in tr];adv+=list(a);ret+=list(r)
            O=torch.tensor(np.array(obs),dtype=torch.float32);A=torch.tensor(act);LP=torch.tensor(logp,dtype=torch.float32)
            AD=torch.tensor(adv,dtype=torch.float32);AD=(AD-AD.mean())/(AD.std()+1e-8);R=torch.tensor(ret,dtype=torch.float32)
            for _ in range(EPOCHS):
                for idx in torch.randperm(len(O)).split(MINI):
                    logits,v=net(O[idx]);dist=torch.distributions.Categorical(logits=logits);ratio=torch.exp(dist.log_prob(A[idx])-LP[idx])
                    pl=-torch.min(ratio*AD[idx],torch.clamp(ratio,1-CLIP,1+CLIP)*AD[idx]).mean();vl=((v-R[idx])**2).mean()
                    loss=pl+0.5*vl-ENT*dist.entropy().mean();opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),0.5);opt.step()
            with torch.no_grad():
                pr=torch.softmax(net(O)[0],-1).mean(0).numpy()
            row=dict(it=it,mae=float(np.mean([m for _,m,_ in res])),mean_h=float(np.mean([m for _,_,m in res])),
                     ret=float(np.mean([sum(x['reward'] for x in tr) for tr,_,_ in res])),option_probs=pr.round(3).tolist(),secs=round(time.time()-t0,1))
            curve.append(row);print(json.dumps(row),flush=True)
            import os;os.makedirs(f'runs/ckpt_{family}{tag}',exist_ok=True);np.savez(f'runs/ckpt_{family}{tag}/it{it:03d}.npz',*net.numpy())
    json.dump(curve,open(f'runs/train_{family}{tag}.json','w'),indent=1)
