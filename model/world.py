"""Canalization-as-perceptual-narrowing gridworld (prototype; see revisionPlan/canalizationModel/SPEC.md).

Units: tiles (1 tile = 1,000 Unreal units), seconds. Toroidal 50x50 grid.
One trial = a sequence of phases; the decanalization dose d(t) relaxes perception only.
"""
from dataclasses import dataclass,field,asdict
import math
import numpy as np

@dataclass
class Params:
    grid:int=50;dt:float=0.05;speed:float=5.;max_turn_deg:float=10.
    passive:float=2.;c0:float=5.;c_min:float=1.;wear_k:float=3.;wear_half_life:float=200.
    patches:int=10;patch_size:int=3;q_lo:float=5.;q_hi:float=35.
    base_delay:float=8.;degrade:float=0.08;recover_tau:float=300.
    n0:float=3.;alpha:float=1/20;beta:float=1/120;familiar_fam:float=0.5;prior_gain:float=1.
    # Costs (0 = off). breadth_cost: h/s at full field and range (metabolic cost of wide perception).
    # perc_noise / dec_noise: log-normal noise SD on perceived food and cost, scaled by breadth / by dose d.
    breadth_cost:float=0.;perc_noise:float=0.;dec_noise:float=0.
    # Reward-driven canal depth (0 = off): kappa grows with fam * min(1, rbar / r_ref), rbar = harvest-rate EMA (tau_r).
    kappa_reward:float=0.;r_ref:float=0.5;tau_r:float=20.
    # Heterogeneous world (0 = off): rich patches of quality [rich_q_lo, rich_q_hi], at least rich_min_dist tiles from the start.
    rich_patches:int=0;rich_q_lo:float=25.;rich_q_hi:float=35.;rich_min_dist:float=20.
    # rich_appear: game time at which the rich patches appear (placed relative to the agent's position then); 0 = at the start.
    rich_appear:float=0.
    # Reliable rich patches: rich_degrade multiplies degradation (0 = never degrade); rich_delay scales their regrowth delay.
    rich_degrade:float=1.;rich_delay:float=1.
    fov_deg:float=90.;fov_narrow:float=0.75;range_tiles:float=10.;range_narrow:float=0.7;rays:int=7;ray_step:float=0.25
    h_seek:float=5.;h_avoid:float=15.;steer_noise:float=0.2
    phases:tuple=(300.,100.,300.)

class Trial:
    def __init__(self,p,seed,dose):
        """dose(t) -> d in [0, 1]: perceived depth is (1 - d) * kappa. A dose with .wide = True never narrows
        and carries no decanalization noise (the WIDE reference)."""
        self.p,self.dose=p,dose;self.wide=getattr(dose,"wide",False);self.rng=np.random.default_rng(seed);g=p.grid
        layout=np.random.default_rng(seed*7919+1)
        self.n=np.zeros((g,g));self.last=np.zeros((g,g))
        self.patch_of=-np.ones((g,g),int);corners=[]
        while len(corners)<p.patches:
            c=tuple(layout.integers(0,g-p.patch_size,2))
            if all(abs(c[0]-o[0])>=p.patch_size+2 or abs(c[1]-o[1])>=p.patch_size+2 for o in corners):corners.append(c)
        self.cells=[]
        for k,(x,y) in enumerate(corners):
            for i in range(p.patch_size):
                for j in range(p.patch_size):self.patch_of[x+i,y+j]=k;self.cells.append((x+i,y+j))
        self.q=layout.uniform(p.q_lo,p.q_hi,p.patches);self.f=np.ones(p.patches);self.f_t=np.zeros(p.patches)
        self.present=np.zeros((g,g),bool);self.regrow=np.full((g,g),np.inf)
        for c in self.cells:self.present[c]=True
        self.prior=np.zeros(p.patches);self.known=np.zeros(p.patches,bool)
        self.x,self.y=layout.uniform(0,g,2);self.heading=layout.uniform(0,2*math.pi)
        self.h=0.;self.kappa=0.;self.rbar=0.;self.t=0.;self.cell=(int(self.x),int(self.y));self.rate=self.enter(self.cell)
        self.deg_k=np.ones(len(self.q));self.delay_k=np.ones(len(self.q))
        self.rich_rng=np.random.default_rng(seed*7919+2);self.rich=[]
        if p.rich_patches and p.rich_appear<=0:self.add_rich(self.rich_rng)
        self.near=-np.ones((g,g),int)
        for (cx,cy) in self.cells:
            for dx in (-1,0,1):
                for dy in (-1,0,1):self.near[(cx+dx)%g,(cy+dy)%g]=self.patch_of[cx,cy]
        self.log=dict(samples=[],entries=[],harvests=[])
    def add_rich(self,rng):
        p=self.p;g=p.grid;size=p.patch_size;added=0
        def tor(a,b):d=abs(a-b)%g;return min(d,g-d)
        while added<p.rich_patches:
            x,y=(int(v) for v in rng.integers(0,g-size,2))
            far=math.hypot(tor(x+size/2,self.x),tor(y+size/2,self.y))>=p.rich_min_dist
            clear=all(self.patch_of[(x+i)%g,(y+j)%g]<0 for i in range(-2,size+2) for j in range(-2,size+2))
            if not (far and clear):continue
            k=len(self.q)
            for i in range(size):
                for j in range(size):self.patch_of[x+i,y+j]=k;self.cells.append((x+i,y+j));self.present[x+i,y+j]=True
            self.q=np.append(self.q,rng.uniform(p.rich_q_lo,p.rich_q_hi));self.f=np.append(self.f,1.);self.f_t=np.append(self.f_t,0.)
            self.prior=np.append(self.prior,0.);self.known=np.append(self.known,False);added+=1
            self.deg_k=np.append(self.deg_k,p.rich_degrade);self.delay_k=np.append(self.delay_k,p.rich_delay)
            if hasattr(self,'near'):
                for i in range(-1,size+1):
                    for j in range(-1,size+1):self.near[(x+i)%g,(y+j)%g]=k
        self.rich=list(range(len(self.q)-p.rich_patches,len(self.q)))
    def maybe_add_rich(self):
        p=self.p
        if p.rich_patches and not self.rich and p.rich_appear>0 and self.t>=p.rich_appear:self.add_rich(self.rich_rng)
    # --- landscape ---
    def n_eff(self,c):
        return self.n[c]*0.5**((self.t-self.last[c])/self.p.wear_half_life)
    def n_eff_grid(self,xs,ys):
        return self.n[xs,ys]*0.5**((self.t-self.last[xs,ys])/self.p.wear_half_life)
    def cost(self,n):
        p=self.p;return p.c_min+(p.c0-p.c_min)*np.exp(-n/p.wear_k)
    def enter(self,c):
        before=self.n_eff(c);self.n[c]=before+1;self.last[c]=self.t
        self.fam=1-math.exp(-before/self.p.n0)
        if hasattr(self,'log'):self.log['entries'].append((self.t,c[0],c[1]))
        return float(self.cost(before))
    def fertility(self,k):
        f=1-(1-self.f[k])*math.exp(-(self.t-self.f_t[k])/self.p.recover_tau);self.f[k]=f;self.f_t[k]=self.t;return f
    # --- perception ---
    def perceived_depth(self):return 0. if self.wide else (1-self.dose(self.t))*self.kappa
    def perceive(self):
        p=self.p;kp=self.perceived_depth();fov=math.radians(p.fov_deg)*(1-p.fov_narrow*kp);L=p.range_tiles*(1-p.range_narrow*kp)
        angles=self.heading+np.linspace(-fov/2,fov/2,p.rays);s=np.arange(p.ray_step,L+1e-9,p.ray_step)
        xs=((self.x+np.cos(angles)[:,None]*s[None,:])%p.grid).astype(int);ys=((self.y+np.sin(angles)[:,None]*s[None,:])%p.grid).astype(int)
        k=self.patch_of[xs,ys];pi=min(1.,p.prior_gain*kp)
        fam=1-np.exp(-self.n_eff_grid(xs,ys)/p.n0)
        prior_on=(k>=0)&(fam>p.familiar_fam)&self.known[np.maximum(k,0)]
        actual=np.where((k>=0)&self.present[xs,ys],self.q[np.maximum(k,0)],0.)
        value=np.where(prior_on,(1-pi)*actual+pi*self.prior[np.maximum(k,0)],actual)
        hit=value>0;first=np.where(hit.any(1),hit.argmax(1),-1)
        food=np.array([value[r,i]/(s[i]) if i>=0 else 0. for r,i in enumerate(first)])
        cost=self.cost(self.n_eff_grid(xs,ys)).mean(1)
        self.breadth=(fov/math.radians(p.fov_deg))*(L/p.range_tiles)
        sigma=p.perc_noise*self.breadth+(0. if self.wide else p.dec_noise*self.dose(self.t))
        z=self.rng.normal(size=(2,p.rays))   # drawn every tick so paired streams stay aligned
        food=food*np.exp(sigma*z[0]);cost=cost*np.exp(sigma*z[1])
        return angles-self.heading,food,cost,fov,L
    # --- control (stage A heuristic) ---
    def heuristic(self,rel,food,cost):
        p=self.p
        if self.h<p.h_seek and food.max()>0:target=rel[food.argmax()]   # hungry: best perceived food
        elif self.h>p.h_avoid:target=rel[np.argmin(food*10+cost)]        # sated: steer away from food
        else:target=rel[np.argmin(cost)]                                  # otherwise: cheapest (worn) path
        return float(np.clip(target/math.radians(p.max_turn_deg),-1,1)+self.rng.normal(0,p.steer_noise))
    # --- dynamics ---
    def step(self,action=None):
        self.maybe_add_rich();p=self.p;rel,food,cost,fov,L=self.perceive()
        a=self.heuristic(rel,food,cost) if action is None else action
        self.heading+=math.radians(p.max_turn_deg)*float(np.clip(a,-1,1))
        self.x=(self.x+math.cos(self.heading)*p.speed*p.dt)%p.grid;self.y=(self.y+math.sin(self.heading)*p.speed*p.dt)%p.grid
        self.t+=p.dt;c=(int(self.x),int(self.y))
        if c!=self.cell:self.cell=c;self.rate=self.enter(c)
        self.h-=(self.rate+p.passive+p.breadth_cost*self.breadth)*p.dt
        for cc in self.cells_due():self.present[cc]=True;self.regrow[cc]=np.inf
        k=self.patch_of[c]
        if k>=0 and self.present[c]:
            self.h+=self.q[k];self.present[c]=False;f=self.fertility(k)*(1-p.degrade*self.deg_k[k]);self.f[k]=f
            self.regrow[c]=self.t+p.base_delay*self.delay_k[k]/max(f,1e-3);self.prior[k]=self.q[k];self.known[k]=True
            self.log['harvests'].append((self.t,int(k),c[0],c[1],float(self.q[k]),f))
        harvested=bool(self.log['harvests']) and self.log['harvests'][-1][0]==self.t
        self.rbar+=(float(harvested)/p.dt-self.rbar)*p.dt/p.tau_r
        drive=self.fam*((1-p.kappa_reward)+p.kappa_reward*min(1.,self.rbar/p.r_ref))
        self.kappa+=p.dt*(p.alpha*drive*(1-self.kappa)-p.beta*self.kappa)
        if round(self.t/p.dt)%10==0:
            kn=self.near[c];degraded=bool(kn>=0 and self.fertility(kn)<0.5)
            self.log['samples'].append((self.t,self.x,self.y,self.h,self.kappa,math.degrees(fov),L,degraded))
    def cells_due(self):
        due=np.argwhere(self.regrow<=self.t);return [tuple(d) for d in due]
    def run(self):
        total=sum(self.p.phases)
        while self.t<total-1e-9:self.step()
        return self

def phase_dose(p,arm,d=1.,duration=None):
    a,b,_=p.phases;end=a+(duration if duration is not None else b)
    if arm=='CCC':return lambda t:0.
    if arm=='WIDE':
        f=lambda t:0.;f.wide=True;return f
    if arm in ('CDR','DOSE'):return lambda t:d if a<=t<end else 0.
    raise ValueError(arm)
