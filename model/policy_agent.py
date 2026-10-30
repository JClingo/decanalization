"""Learned option policy over the cognitive map (HRL vs NF).

The map (values, precision, regrowth expectation, frontier) is as in mapagent.py. Every
DECISION_TICKS ticks a policy picks one option, executed with the map agent's steering rules:
  0 SEEK     head for the remembered cell with the best worth / (1 + distance) (frontier if none)
  1 EXPLORE  head for the nearest never-seen cell >= 3 tiles away
  2 CRUISE   follow the cheapest ray (worn path)
  3 AVOID    steer away from visible food (cruise if none)
Rewards per tick (summed over the option): kappa * (D_prev - D_cur), kappa = dt / (1 - gamma_tick)
  HRL drive-ratio:      D = |h| / 25
  NF  nf-deficit-drive: D = max(-h, 0) / 25
Simplification: no finite-endpoint correction; trials are long (1,400 decisions).
Decanalization (dose d): precision reduction as in mapagent.py (prec_lever), plus a policy
sampling temperature 1 + temp_boost * d (the learned-policy analogue of the exploration boost).
"""
import math
import numpy as np
from mapagent import MapParams,MapTrial

DECISION_TICKS=10;OPTIONS=4;OBS=11;GAMMA_TICK=0.99

def mlp_forward(w,x):
    h=np.tanh(x@w[0]+w[1]);h=np.tanh(h@w[2]+w[3]);out=h@w[4]+w[5];return out[...,:OPTIONS],out[...,OPTIONS]

def drive(family,h):return abs(h)/25. if family=='HRL' else max(-h,0.)/25.

class OptionTrial(MapTrial):
    def __init__(self,p,seed,dose,family,weights,temp_boost=0.,record=False,greedy=False):
        super().__init__(p,seed,dose);self.family,self.w,self.temp_boost,self.record,self.greedy=family,weights,temp_boost,record,greedy
        self.option=2;self.ticks=0;self.trans=[];self.acc=0.;self.D=drive(family,self.h);self.last_eat=0.
        self.pol_rng=np.random.default_rng(seed*31+7)
    # --- geometry helpers ---
    def rel_angle(self,dx,dy):return (math.atan2(dy,dx)-self.heading+math.pi)%(2*math.pi)-math.pi
    def offsets(self):
        g=self.p.grid;gx,gy=np.meshgrid(np.arange(g)+0.5,np.arange(g)+0.5,indexing='ij')
        dx=(gx-self.x+g/2)%g-g/2;dy=(gy-self.y+g/2)%g-g/2;return dx,dy,np.hypot(dx,dy)
    def best_target(self,dx,dy,dist):
        p=self.p;avail=1-np.exp(-(self.t-self.last_harvest)/p.base_delay);worth=self.V*avail
        score=np.where(worth>=p.target_min,worth/(1+dist),0.);i=np.unravel_index(score.argmax(),score.shape)
        return (i,float(worth[i]),float(self.P[i]),float(dist[i])) if score.max()>0 else None
    def frontier(self,dist):
        fr=np.where((~self.seen)&(dist>=3),dist,np.inf);i=np.unravel_index(fr.argmin(),fr.shape)
        return (i,float(fr[i])) if np.isfinite(fr[i]) else None
    def observation(self,dx,dy,dist,cx,cy,sample,cost):
        b=self.best_target(dx,dy,dist);f=self.frontier(dist);vis=sample>0
        if vis.any():
            vd=np.hypot((cx[vis]+0.5-self.x+25)%50-25,(cy[vis]+0.5-self.y+25)%50-25);vmax,vnear=sample[vis].max(),vd.min()
        else:vmax,vnear=0.,10.
        return np.array([np.clip(self.h/25,-4,4),(b[1] if b else 0.)/15,(b[3] if b else 25.)/25,math.log1p(b[2] if b else 0.)/3,
            vmax/15,vnear/10,(f[1] if f else 25.)/25,self.seen.mean(),float(self.cost(self.n_eff(self.cell)))/5,
            min((self.t-self.last_eat)/30,3.),1.])
    def choose(self,obs):
        logits,value=mlp_forward(self.w,obs);tau=1+self.temp_boost*self.dose(self.t)
        z=logits/tau;z=z-z.max();pr=np.exp(z);pr/=pr.sum()
        a=int(pr.argmax()) if self.greedy else int(self.pol_rng.choice(OPTIONS,p=pr))
        return a,float(np.log(pr[a]+1e-12)),float(value)
    # --- one tick ---
    def step(self,action=None):
        self.maybe_add_rich();p=self.p;d=self.dose(self.t);dp=d*p.prec_lever
        if d>0 and not self.flattened:self.flatten(dp);self.flattened=True
        if d==0:self.flattened=False
        self.decay()
        fov=math.radians(p.fov_deg);L=p.range_tiles
        angles=self.heading+np.linspace(-fov/2,fov/2,p.rays);s=np.arange(p.ray_step,L+1e-9,p.ray_step)
        xs=((self.x+np.cos(angles)[:,None]*s[None,:])%p.grid).astype(int).ravel();ys=((self.y+np.sin(angles)[:,None]*s[None,:])%p.grid).astype(int).ravel()
        cells=np.unique(np.stack([xs,ys],1),axis=0);cx,cy=cells[:,0],cells[:,1];self.seen[cx,cy]=True
        k=self.patch_of[cx,cy];sample=np.where((k>=0)&self.present[cx,cy],self.q[np.maximum(k,0)],0.)
        if dp>0:
            saved=self.P[cx,cy].copy();self.P[cx,cy]*=(1-dp);self.observe(cx,cy,sample,p.P_see);self.P[cx,cy]+=saved*dp
        else:self.observe(cx,cy,sample,p.P_see)
        cost=self.cost(self.n_eff_grid(xs.reshape(p.rays,-1),ys.reshape(p.rays,-1))).mean(1);rel=angles-self.heading
        dx,dy,dist=self.offsets()
        if self.ticks%DECISION_TICKS==0:
            obs=self.observation(dx,dy,dist,cx,cy,sample,cost)
            if self.record and self.trans:self.trans[-1]['reward']=self.acc
            a,logp,v=self.choose(obs);self.option=a;self.acc=0.
            if self.record:self.trans.append(dict(obs=obs,a=a,logp=logp,v=v,reward=0.))
            self.log.setdefault('options',[]).append((self.t,a))
        o=self.option
        if o==0:
            b=self.best_target(dx,dy,dist)
            if b:target=self.rel_angle(dx[b[0]],dy[b[0]])
            else:o=1
        if o==1:
            f=self.frontier(dist);target=self.rel_angle(dx[f[0]],dy[f[0]]) if f else 0.
        if o==3:
            vis=sample>0
            if vis.any():
                fx=(cx[vis]+0.5-self.x+p.grid/2)%p.grid-p.grid/2;fy=(cy[vis]+0.5-self.y+p.grid/2)%p.grid-p.grid/2
                target=self.rel_angle(-fx.mean(),-fy.mean())
            else:o=2
        if o==2:target=rel[np.argmin(cost)]
        a_=float(np.clip(target/math.radians(p.max_turn_deg),-1,1)+self.rng.normal(0,p.steer_noise))
        self.rng.normal(size=(2,p.rays))
        self.heading+=math.radians(p.max_turn_deg)*float(np.clip(a_,-1,1))
        self.x=(self.x+math.cos(self.heading)*p.speed*p.dt)%p.grid;self.y=(self.y+math.sin(self.heading)*p.speed*p.dt)%p.grid
        self.t+=p.dt;self.ticks+=1;c=(int(self.x),int(self.y))
        if c!=self.cell:self.cell=c;self.rate=self.enter(c)
        self.h-=(self.rate+p.passive)*p.dt
        for cc in self.cells_due():self.present[cc]=True;self.regrow[cc]=np.inf
        kk=self.patch_of[c]
        if kk>=0 and self.present[c]:
            self.h+=self.q[kk];self.present[c]=False;f=self.fertility(kk)*(1-p.degrade*self.deg_k[kk]);self.f[kk]=f
            self.regrow[c]=self.t+p.base_delay*self.delay_k[kk]/max(f,1e-3)
            self.observe(np.array([c[0]]),np.array([c[1]]),np.array([self.q[kk]]),p.P_reward);self.last_harvest[c]=self.t;self.last_eat=self.t
            self.log['harvests'].append((self.t,int(kk),c[0],c[1],float(self.q[kk]),f))
        else:self.observe(np.array([c[0]]),np.array([c[1]]),np.array([0.]),p.P_touch*p.dt)
        D=drive(self.family,self.h);self.acc+=(p.dt/(1-GAMMA_TICK))*(self.D-D);self.D=D
        if round(self.t/p.dt)%10==0:
            kn=self.near[c];degraded=bool(kn>=0 and self.fertility(kn)<0.5)
            self.log['samples'].append((self.t,self.x,self.y,self.h,self.kappa,math.degrees(fov),L,degraded))
    def run(self):
        super().run()
        if self.record and self.trans:self.trans[-1]['reward']=self.acc
        return self
