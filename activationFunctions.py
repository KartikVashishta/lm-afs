# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Damien Teney damien.teney@idiap.ch
# SPDX-License-Identifier: MIT

'''
Activation functions using learneable linear splines.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

# Evaluate, at points x, a function defined as the linear interpolation of anchors of coordinates 'anchors' and values 'vals'
torch.compile(dynamic=False)
def evalSpline(x, anchors, vals):
    idx = torch.bucketize(x, anchors) - 1 # Find which interval each x falls into
    idx = idx.clamp(0, len(anchors) - 2)

    stepSize = anchors[1] - anchors[0]
    x0 = anchors[0] + idx * stepSize
    frac = (x - x0) / stepSize
    frac = frac.clamp(0.0, 1.0) # Constant extrapolation

    y0 = vals[idx]
    y1 = vals[idx + 1]
    out = y0 + frac * (y1 - y0) # Linear interpolation
    return out.to(x.dtype) # Cast back to bfloat16; anchors/frac/out were float32

# Same for *channel-specific* splines
@torch.compile(dynamic=False)
def evalSplinePerDim(x, anchors, vals):
    # x:       [batchSize, seqLength, dimHidden]
    # anchors: [nAnchors]
    # vals:    [dimHidden, nAnchors]

    # Sanity checks
    assert anchors.ndim == 1,                 (f"x.shape={x.shape}, anchors.shape={anchors.shape}, vals.shape={vals.shape}")
    assert vals.ndim == 2,                    (f"x.shape={x.shape}, anchors.shape={anchors.shape}, vals.shape={vals.shape}")
    assert vals.shape[1] == anchors.shape[0], (f"x.shape={x.shape}, anchors.shape={anchors.shape}, vals.shape={vals.shape}")
    dimHidden = vals.shape[0]
    if x.ndim == 1: # Function called to plot the AF (not evaluating MLP activations)
        x = x.view(1, -1, 1).expand(-1, -1, dimHidden) # [nPts] -> [1, nPts, dimHidden], replicate to eval on all dimensions
        squeezeOut = True
    else:
        squeezeOut = False
    assert x.ndim == 3,                       (f"x.shape={x.shape}, anchors.shape={anchors.shape}, vals.shape={vals.shape}")
    assert x.shape[2] == dimHidden,           (f"x.shape={x.shape}, anchors.shape={anchors.shape}, vals.shape={vals.shape}")

    idx = torch.bucketize(x, anchors) - 1 # Find which interval each x falls into
    idx = idx.clamp(0, len(anchors) - 2)

    stepSize = anchors[1] - anchors[0]
    x0 = anchors[0] + idx * stepSize
    frac = (x - x0) / stepSize
    frac = frac.clamp(0.0, 1.0) # Constant extrapolation

    # This is the part that differs from evalSpline()
    dimIds = torch.arange(dimHidden, device=x.device).view(1, 1, dimHidden) # [1, 1, dimHidden]
    y0 = vals[dimIds, idx]     # [batchSize, seqLength, dimHidden
    y1 = vals[dimIds, idx + 1] # [batchSize, seqLength, dimHidden

    out = y0 + frac * (y1 - y0) # Linear interpolation
    if squeezeOut:
        out = out.squeeze(0) # [1, nPts, dimHidden] -> [nPts, dimHidden]
        out = out[:, 0] # [nPts] Will only plot the AF for the 1st dimension
    return out.to(x.dtype) # Cast back to bfloat16; anchors/frac/out were float32

class af(nn.Module):
    def __init__(self, afType="gelu", afRange=15, afNAnchors=128, afInit=0.01, dim=128, dtype=torch.float32):
        super().__init__()
        self.afType = afType
        self.dim = dim
        assert ("-" not in afType) and (";" not in afType), f"Invalid afType: {afType}"

        if not self.isParameterized: # Non-parametrized (e.g. GeLU)
            self.afAnchors = None
            self.afVals = None
            self.isTrained = False

        else: # Learnable AF (spline)
            self.isTrained = True
            self.afAnchors = torch.linspace(-afRange, afRange, afNAnchors, dtype=torch.float32) # Will remain in float32
            diffs = torch.diff(self.afAnchors); assert torch.allclose(diffs, diffs[0].expand_as(diffs), rtol=1e-4, atol=1e-4), f"afAnchors not evenly spaced: {self.afAnchors}" # Sanity check: anchors are evenly spaced
            if (afInit == -1.0) and ("splinePerDim" not in self.afType): # Special value: init as linear function
                self.afVals = nn.Parameter(torch.linspace(-afRange, afRange, afNAnchors, dtype=dtype))
            elif (afInit == -1.0) and ("splinePerDim" in self.afType): # Special value: init as linear function
                self.afVals = nn.Parameter(torch.linspace(-afRange, afRange, afNAnchors, dtype=dtype).unsqueeze(0).expand(self.dim, -1))
            elif "splinePerDim" in self.afType: # Standard init as constants, with different vals per dim
                assert afInit > 0 # Initial constant value
                self.afVals = nn.Parameter(afInit * torch.ones(self.dim, afNAnchors, dtype=dtype))
            else: # Standard init as constants
                assert afInit > 0 # Initial constant value
                self.afVals = nn.Parameter(afInit * torch.ones(afNAnchors, dtype=dtype))

    def loadTrainedAf(self, afType, afAnchors, afVals, fineTune=False):
        assert self.afType == afType, f"{self.afType} != {afType}"
        assert self.isParameterized and (afAnchors is not None) and (afVals is not None)
        self.afAnchors = afAnchors.detach().to(dtype=torch.float32) # Will remain in float32
        assert isinstance(afVals, nn.Parameter), f"afVals: {type(afVals)}"
        self.afVals = nn.Parameter(afVals.detach(), requires_grad=fineTune)
        self.isTrained = fineTune

    @property
    def isParameterized(self):
        return self.afType.startswith("spline")

    def forward(self, x):
        if (self.afAnchors is not None) and (self.afAnchors.device != x.device):
            self.afAnchors = self.afAnchors.to(x.device, dtype=torch.float32)
        if   self.afType == "linear":       y = x
        elif self.afType == "gelu":         y = F.gelu(x)
        elif self.afType == "relu":         y = F.relu(x)
        elif self.afType == "relu2":        y = F.relu(x).square() # https://arxiv.org/abs/2109.08668
        elif self.afType == "silu":         y = F.silu(x)
        elif self.afType == "tanh":         y = torch.tanh(x)
        elif self.afType == "spline":       y = evalSpline(x, self.afAnchors, self.afVals)
        elif self.afType == "splinePerDim": y = evalSplinePerDim(x, self.afAnchors, self.afVals)
        else: raise ValueError(f"Unknown activation type: {self.afType}")    
        torch._assert(x.dtype == y.dtype, f"Expected dtype {x.dtype}, got {y.dtype} after activation {self.afType}") # Check that the type of the input is conserved
        return y

def isGroupedAfType(afType): return isinstance(afType, str) and afType.startswith("group:")

def parseGroupedAfType(afType):
    assert isGroupedAfType(afType), f"Invalid grouped AF spec: {afType}"
    groupAfTypes = [s.strip() for s in afType.split(":", 1)[1].split(",") if s.strip()]
    assert len(groupAfTypes)>=2, f"Need >=2 group activations in grouped spec: {afType}"
    return groupAfTypes

class afSplit(nn.Module):
    # split the last dim into equal groups and apply one activation per group
    # syntax: group:relu2,gelu,relu,silu
    def __init__(self, afType="group:relu2,gelu,relu,silu", afRange=15, afNAnchors=128, afInit=0.01, dim=128, dtype=torch.float32):
        super().__init__()
        self.afType=afType
        self.groupAfTypes = parseGroupedAfType(afType)
        self.nGroups=len(self.groupAfTypes)
        self.dim=dim
        ## todo: graceful division of the activations within the groups 
        assert self.dim % self.nGroups == 0, f"dim={self.dim} should be divisible by nGroups={self.nGroups} for {afType}"
        self.dimPerGroup=self.dim//self.nGroups
        self.afs = nn.ModuleList([af(groupAfType, afRange, afNAnchors, afInit, self.dimPerGroup, dtype) for groupAfType in self.groupAfTypes])
    
    @property
    def isParameterized(self):
        return any(m.isParameterized for m in self.afs)

    def forward(self, x):
        #only used as a fallback, proper plotting should use forwardForPlot()
        if x.ndim==1:
            ys = [m(x) for m in self.afs]
            y=torch.stack(ys, dim=0).mean(dim=0)
        else:
            assert x.shape[-1]==self.dim, f"Expected last dim {self.dim}, got {x.shape[-1]}"
            xChunks = x.split(self.dimPerGroup, dim=-1)
            yChunks = [m(xChunk) for m, xChunk in zip(self.afs, xChunks)]
            y=torch.cat(yChunks, dim=-1)
        torch._assert(x.dtype==y.dtype, f"Expected dtype {x.dtype}, got {y.dtype} after grouped activation {self.afType}")
        return y
    
    def forwardForPlot(self, x): return [m(x) for m in self.afs]