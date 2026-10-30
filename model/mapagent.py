"""Cognitive-map agent: canalization from learned precision (see revisionPlan/canalizationModel/COGNITIVE_MAP_BOOKMARK.md).

Same world as world.py; fixed perception (no imposed narrowing). The agent keeps a map of
expected pellet value V and precision P per cell.
- Sighting a cell gives a sample (q if a pellet is present, else 0) with precision P_see per
  tick; standing on a cell gives P_touch; a harvest gives P_reward. Beliefs update by
  precision weighting: V <- (P*V + Ps*v) / (P + Ps), P <- P + Ps. P decays with time constant
  tau_P. Rewarded cells become precise, so later contrary evidence moves them little: an
  exhausted rut keeps looking rich. Canalization is learned, not imposed.
- Expected availability: after a harvest the agent believes a cell regrows over base_delay, so
  worth = V * (1 - exp(-(t - t_harvest) / base_delay)). A degraded patch (slower real regrowth)
  disappoints on each return; precise beliefs make it keep returning.
- Planning: when hungry, head for the cell maximising worth / (1 + distance) if its worth
  is at least v_explore, else explore straight ahead; when sated,
  avoid visible food; otherwise cruise along the cheapest ray.
- Decanalization (REBUS flattening), dose d at the window start: P <- (1-d)*P and
  V <- (1-d)*V + d*mean(V over known cells); during the window inference uses (1-d)*P.
  A full dose erases map structure as well as rigidity. flatten_values = 0 scales precision only.
"""
from dataclasses import dataclass
import math
import numpy as np
from world import Params,Trial

@dataclass
class MapParams(Params):
    P_see:float=0.05;P_touch:float=0.5;P_reward:float=2.;tau_P:float=200.;target_min:float=1.;v_explore:float=4.;flatten_values:float=1.
    # Decanalization levers: prec_lever scales the precision reduction (1 = on); explore_boost raises the
    # exploration threshold during the window, v_explore * (1 + explore_boost * d) (REBUS: relaxed priors, more exploration).
    prec_lever:float=1.;explore_boost:float=0.
    # directed_explore: when exploring, head for the nearest never-seen cell at least 3 tiles away (frontier), not straight ahead.
    directed_explore:float=0.
    fov_narrow:float=0.;range_narrow:float=0.;prior_gain:float=0.

class MapTrial(Trial):
    def __init__(self,p,seed,dose):
        super().__init__(p,seed,dose);g=p.grid
        self.V=np.zeros((g,g));self.P=np.zeros((g,g));self.last_harvest=np.full((g,g),-np.inf);self.seen=np.zeros((g,g),bool);self.P_t=0.;self.flattened=False
    def decay(self):
        dt=self.t-self.P_t
        if dt>0:self.P*=math.exp(-dt/self.p.tau_P);self.P_t=self.t
    def observe(self,xs,ys,values,ps):
        P=self.P[xs,ys];self.V[xs,ys]=(P*self.V[xs,ys]+ps*values)/(P+ps);self.P[xs,ys]=P+ps
    def flatten(self,d):
        known=self.P>1e-6
        if known.any() and self.p.flatten_values:self.V[known]=(1-d)*self.V[known]+d*self.V[known].mean()
        self.P*=(1-d)
    def believed(self):
        d=self.dose(self.t);return self.V,self.P*(1-d)
    def step(self,action=None):
        self.maybe_add_rich();p=self.p;d=self.dose(self.t)
        dp=d*p.prec_lever
        if d>0 and not self.flattened:self.flatten(dp);self.flattened=True
        if d==0:self.flattened=False
        self.decay()
        # perception: fixed field; sightings update the map, weighted by current (dose-scaled) precision
        fov=math.radians(p.fov_deg);L=p.range_tiles
        angles=self.heading+np.linspace(-fov/2,fov/2,p.rays);s=np.arange(p.ray_step,L+1e-9,p.ray_step)
        xs=((self.x+np.cos(angles)[:,None]*s[None,:])%p.grid).astype(int).ravel();ys=((self.y+np.sin(angles)[:,None]*s[None,:])%p.grid).astype(int).ravel()
        cells=np.unique(np.stack([xs,ys],1),axis=0);cx,cy=cells[:,0],cells[:,1];self.seen[cx,cy]=True
        k=self.patch_of[cx,cy];sample=np.where((k>=0)&self.present[cx,cy],self.q[np.maximum(k,0)],0.)
        if dp>0:   # relaxed inference: prior weight scaled by (1-dp)
            saved=self.P[cx,cy].copy();self.P[cx,cy]*=(1-dp);self.observe(cx,cy,sample,p.P_see);self.P[cx,cy]+=saved*dp
        else:self.observe(cx,cy,sample,p.P_see)
        cost=self.cost(self.n_eff_grid(xs.reshape(p.rays,-1),ys.reshape(p.rays,-1))).mean(1);rel=angles-self.heading
        # control
        self._target=None
        if self.h<p.h_seek:
            gx,gy=np.meshgrid(np.arange(p.grid)+0.5,np.arange(p.grid)+0.5,indexing='ij')
            dx=(gx-self.x+p.grid/2)%p.grid-p.grid/2;dy=(gy-self.y+p.grid/2)%p.grid-p.grid/2;dist=np.hypot(dx,dy)
            # expected availability: a harvested cell is believed to regrow over base_delay
            avail=1-np.exp(-(self.t-self.last_harvest)/p.base_delay)
            worth=self.V*avail
            score=np.where(worth>=p.target_min,worth/(1+dist),0.)
            best=np.unravel_index(score.argmax(),score.shape)
            if score.max()>0 and worth[best]>=p.v_explore*(1+p.explore_boost*d):   # a remembered cell worth more than exploring
                i=np.unravel_index(score.argmax(),score.shape);want=math.atan2(dy[i],dx[i]);self._target=(i,float(worth[i]),float(self.P[i]),float(dist[i]))
                target=(want-self.heading+math.pi)%(2*math.pi)-math.pi
            elif p.directed_explore and (~self.seen).any():   # nothing worth more than exploring: go to the frontier
                fr=np.where((~self.seen)&(dist>=3),dist,np.inf);i=np.unravel_index(fr.argmin(),fr.shape)
                target=(math.atan2(dy[i],dx[i])-self.heading+math.pi)%(2*math.pi)-math.pi if np.isfinite(fr[i]) else 0.
            else:target=0.   # nothing worth more than exploring: head on into the unknown (noise turns it)
        else:
            vis=sample>0
            if self.h>p.h_avoid and vis.any():
                fx=(cx[vis]+0.5-self.x+p.grid/2)%p.grid-p.grid/2;fy=(cy[vis]+0.5-self.y+p.grid/2)%p.grid-p.grid/2
                away=math.atan2(-fy.mean(),-fx.mean());target=(away-self.heading+math.pi)%(2*math.pi)-math.pi
            else:target=rel[np.argmin(cost)]
        a=float(np.clip(target/math.radians(p.max_turn_deg),-1,1)+self.rng.normal(0,p.steer_noise))
        self.rng.normal(size=(2,p.rays))   # keep the random stream aligned with world.Trial
        self.heading+=math.radians(p.max_turn_deg)*float(np.clip(a,-1,1))
        self.x=(self.x+math.cos(self.heading)*p.speed*p.dt)%p.grid;self.y=(self.y+math.sin(self.heading)*p.speed*p.dt)%p.grid
        self.t+=p.dt;c=(int(self.x),int(self.y))
        if c!=self.cell:self.cell=c;self.rate=self.enter(c)
        self.h-=(self.rate+p.passive)*p.dt
        for cc in self.cells_due():self.present[cc]=True;self.regrow[cc]=np.inf
        kk=self.patch_of[c]
        if kk>=0 and self.present[c]:
            self.h+=self.q[kk];self.present[c]=False;f=self.fertility(kk)*(1-p.degrade*self.deg_k[kk]);self.f[kk]=f
            self.regrow[c]=self.t+p.base_delay*self.delay_k[kk]/max(f,1e-3)
            self.observe(np.array([c[0]]),np.array([c[1]]),np.array([self.q[kk]]),p.P_reward);self.last_harvest[c]=self.t
            self.log['harvests'].append((self.t,int(kk),c[0],c[1],float(self.q[kk]),f))
        else:
            self.observe(np.array([c[0]]),np.array([c[1]]),np.array([0.]),p.P_touch*p.dt)
        self.kappa=float(self.P[self.P>1e-6].max()/(1+self.P[self.P>1e-6].max())) if (self.P>1e-6).any() else 0.
        if round(self.t/p.dt)%10==0:
            kn=self.near[c];degraded=bool(kn>=0 and self.fertility(kn)<0.5)
            self.log['samples'].append((self.t,self.x,self.y,self.h,self.kappa,math.degrees(fov),L,degraded))
            if getattr(self,'diag',False):   # diagnostic: what the agent is pursuing
                mode='seek' if self.h<p.h_seek else 'sated'
                if self._target is None:self.log.setdefault('diag',[]).append((self.t,self.h,mode,None))
                else:
                    (ix,iy),w,P,dist=self._target;kp=int(self.patch_of[ix,iy])
                    self.log.setdefault('diag',[]).append((self.t,self.h,mode,(ix,iy),w,P,dist,bool(self.present[ix,iy]),kp,
                        float(self.fertility(kp)) if kp>=0 else None,float(self.q[kp]) if kp>=0 else 0.))
