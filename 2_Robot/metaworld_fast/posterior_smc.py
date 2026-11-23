"""Joint tempered resample-move SMC with exact delayed acceptance.

All arrays and reductions are float64. Optional CUDA accelerates full-history
target evaluation only; proposals and stable random streams stay on the CPU.
Gaussian densities are a frozen first-stage screen, never the target.
"""
from __future__ import annotations
import time
import numpy as np
from scipy.special import logsumexp
from .model import log_probabilities
from .metrics import integrated_eig


def effective_sample_size(logweights):
    normalized = logweights-logsumexp(logweights)
    return float(np.exp(-logsumexp(2*normalized)))


class SMCPosterior:
    factorized = False

    def __init__(self, d=4, beta=4., particles=4096, rng=None, device="cpu"):
        self.d, self.beta, self.n = int(d), float(beta), int(particles)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.device = str(device)
        self.particles = self.rng.uniform(-1, 1, (self.n, self.d))
        self.logweights = np.full(self.n, -np.log(self.n))
        self.log_target = np.zeros(self.n)
        self.history = []
        self.proposal_scale = 2.38 / np.sqrt(d)
        self.last_diagnostics = {}
        self._coefficients = None
        self._torch_coefficients = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_torch_coefficients"] = None
        return state

    @property
    def weights(self):
        return np.exp(self.logweights-logsumexp(self.logweights))

    def moments(self):
        weights = self.weights
        mean = weights @ self.particles
        centered = self.particles-mean
        cov = centered.T @ (centered*weights[:,None])
        return mean, (cov+cov.T)/2

    def projected_quantiles(self, direction, probabilities):
        projection = self.particles @ direction
        order = np.argsort(projection, kind="stable")
        cdf = np.cumsum(self.weights[order])
        cdf[-1] = 1.
        # The specification chooses the first cumulative weight strictly above
        # the requested probability (including exact dyadic boundaries).
        indices = np.searchsorted(cdf, probabilities, side="right")
        return projection[order[np.minimum(indices,self.n-1)]]

    def sample(self, count, rng):
        return self.particles[rng.choice(self.n, size=count, replace=True, p=self.weights)].copy()

    def eig(self, q):
        return integrated_eig(self.particles, self.weights, q, self.beta)

    def hypothetical_means(self, q):
        joint = self.logweights[:,None]+log_probabilities(self.particles,q,self.beta)
        return np.exp(joint-logsumexp(joint,axis=0)).T @ self.particles

    def _refresh_coefficients(self):
        if not self.history:
            self._coefficients = None
        else:
            q = np.stack([entry[0] for entry in self.history])
            y = np.array([entry[1] for entry in self.history])
            # Subtract q_0 to reduce logits magnitudes; retain all norm offsets.
            qc = q-q[:,0:1,:]
            coeff = 2*self.beta*qc
            intercept = -self.beta*np.sum(qc*qc,axis=2)-np.einsum("hmd,hd->hm",coeff,q[:,0,:])
            self._coefficients = coeff, intercept, y
        self._torch_coefficients = None

    def target_logdensity(self, points, alpha=1.):
        """Exact full-history target, with final observation power alpha."""
        points = np.asarray(points,dtype=np.float64)
        output = np.full(len(points), -np.inf)
        valid = np.all((points >= -1) & (points <= 1),axis=1)
        if not np.any(valid):
            return output
        if self._coefficients is None:
            output[valid]=0.
            return output
        coeff, intercept, y = self._coefficients
        if self.device.startswith("cuda"):
            import torch
            if self._torch_coefficients is None:
                self._torch_coefficients = (
                    torch.as_tensor(coeff,dtype=torch.float64,device=self.device),
                    torch.as_tensor(intercept,dtype=torch.float64,device=self.device),
                    torch.as_tensor(y,dtype=torch.long,device=self.device))
            tc, ti, ty = self._torch_coefficients
            with torch.inference_mode():
                tx = torch.as_tensor(points[valid],dtype=torch.float64,device=self.device)
                out = torch.zeros(len(tx),dtype=torch.float64,device=self.device)
                for start in range(0,len(y),32):
                    end = min(start+32,len(y))
                    c = tc[start:end]
                    logits = (tx @ c.reshape(-1,self.d).T).reshape(len(tx),end-start,-1)+ti[start:end]
                    chosen = torch.gather(logits,2,ty[start:end][None,:,None].expand(len(tx),-1,1)).squeeze(2)
                    ll = chosen-torch.logsumexp(logits,dim=2)
                    if end == len(y) and alpha != 1:
                        ll[:,-1] *= float(alpha)
                    out += ll.sum(dim=1)
                output[valid] = out.cpu().numpy()
        else:
            x = points[valid]
            out = np.zeros(len(x))
            for start in range(0,len(y),16):
                end = min(start+16,len(y))
                c = coeff[start:end]
                logits = (x @ c.reshape(-1,self.d).T).reshape(len(x),end-start,-1)+intercept[start:end]
                chosen = np.take_along_axis(logits,y[start:end][None,:,None],axis=2)[:,:,0]
                ll = chosen-logsumexp(logits,axis=2)
                if end == len(y) and alpha != 1:
                    ll[:,-1] *= float(alpha)
                out += ll.sum(axis=1)
            output[valid] = out
        return output

    def _resample_move(self, alpha, newq, newy, diagnostics):
        cdf = np.cumsum(self.weights)
        cdf[-1] = 1.
        positions = (self.rng.random()+np.arange(self.n))/self.n
        indices = np.searchsorted(cdf,positions,side="right")
        self.particles = self.particles[indices].copy()
        self.log_target = self.log_target[indices].copy()
        self.logweights.fill(-np.log(self.n))
        diagnostics["resamplings"] += 1
        for sweep in range(10):
            mean, covariance = self.moments()
            try:
                factor = np.linalg.cholesky(covariance)
            except np.linalg.LinAlgError as exc:
                raise FloatingPointError("Degenerate SMC proposal covariance; no ridge added") from exc
            inverse = np.linalg.inv(covariance)
            scale = self.proposal_scale
            proposals = self.particles + scale*self.rng.standard_normal(self.particles.shape)@factor.T
            current_delta = self.particles-mean
            proposal_delta = proposals-mean
            current_g = -.5*np.einsum("nd,de,ne->n",current_delta,inverse,current_delta)
            proposal_g = -.5*np.einsum("nd,de,ne->n",proposal_delta,inverse,proposal_delta)
            log_screen_ratio = proposal_g-current_g
            valid = np.all((proposals>=-1)&(proposals<=1),axis=1)
            pass1 = valid & (np.log(self.rng.random(self.n)) < np.minimum(0.,log_screen_ratio))
            selected = np.flatnonzero(pass1)
            newtarget = self.target_logdensity(proposals[selected],alpha)
            correction = newtarget-self.log_target[selected]-log_screen_ratio[selected]
            pass2 = np.log(self.rng.random(len(selected))) < np.minimum(0.,correction)
            accepted = selected[pass2]
            self.particles[accepted] = proposals[accepted]
            self.log_target[accepted] = newtarget[pass2]
            rate = len(accepted)/self.n
            diagnostics["sweeps"].append({"sweep":sweep,"alpha":float(alpha),"scale":float(scale),"stage1":len(selected)/self.n,"acceptance":rate})
            # Proposal/screen are frozen within each sweep. Adapt only between sweeps.
            self.proposal_scale = float(scale*np.exp(.10*(rate-.234)))
        return log_probabilities(self.particles,newq,self.beta)[:,newy]

    def update(self,q,y):
        start_time = time.monotonic()
        q,y = np.asarray(q,dtype=np.float64),int(y)
        self.history.append((q.copy(),y))
        self._refresh_coefficients()
        newll = log_probabilities(self.particles,q,self.beta)[:,y]
        alpha = 0.
        diagnostics = {"ess_before":effective_sample_size(self.logweights),"increments":[],"resamplings":0,"sweeps":[]}
        target_ess = .8*self.n
        attempts = 0
        while alpha < 1.-1e-14:
            attempts += 1
            if attempts > 512:
                raise FloatingPointError("SMC tempering failed to progress in 512 steps")
            remaining = 1.-alpha
            if effective_sample_size(self.logweights+remaining*newll) >= target_ess:
                delta = remaining
            elif effective_sample_size(self.logweights) <= target_ess*(1+1e-10):
                newll = self._resample_move(alpha,q,y,diagnostics)
                continue
            else:
                low,high = 0.,remaining
                for _ in range(45):
                    middle = (low+high)/2
                    if effective_sample_size(self.logweights+middle*newll) >= target_ess:
                        low=middle
                    else:
                        high=middle
                delta=low
                if delta <= 1e-12:
                    newll = self._resample_move(alpha,q,y,diagnostics)
                    continue
            self.logweights += delta*newll
            self.logweights -= logsumexp(self.logweights)
            self.log_target += delta*newll
            alpha += delta
            if 1-alpha < 1e-14:
                alpha=1.
            diagnostics["increments"].append({"delta":float(delta),"exponent":float(alpha),"ess":effective_sample_size(self.logweights)})
            if alpha < 1:
                newll = self._resample_move(alpha,q,y,diagnostics)
        if effective_sample_size(self.logweights) < .5*self.n:
            self._resample_move(1.,q,y,diagnostics)
        diagnostics.update(ess_after=effective_sample_size(self.logweights),final_exponent=alpha,seconds=time.monotonic()-start_time)
        self.last_diagnostics=diagnostics
