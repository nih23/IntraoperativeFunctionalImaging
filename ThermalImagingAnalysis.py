import ActivityPatterns as ap
import numpy as np
import h5py
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import bspline

def semiparamRegression(S, X, B, P):
    """Apply semiparametric regression framework to imaging data.
    S: m x n data cube with m time series of length n
    X: length m vector of discretized parametric function
    B: non parametric basis
    P: penalty matrix of non-parametric basis
    """
    [noTimepoints, noPixels] = S.shape
    m = np.mean(S,axis=0)
    S2 = S - m
    G = np.concatenate([X, B]).transpose();
    [noFixedComponents, noTimepoints] = X.shape
    assert (noFixedComponents == 1), "The hypothesis test only works for a single parametric component."
    # compute Penalty term
    E1 = 0 * np.eye(noFixedComponents)
    S_P = linalg.block_diag(E1,P)
    Pterm = S_P.transpose().dot(S_P)
    # allocate intermediate storage
    lambdas= np.linspace(0.1,10,10)
    GtG = G.transpose().dot(G)
    AIC = np.zeros([len(lambdas),noPixels])
    Z = np.zeros([len(lambdas),noPixels])
    for i in range(0,len(lambdas)):
        # fit model using penalised normal equations
        lambda_i = lambdas[i]
        GtGpD = GtG + lambda_i * Pterm;
        GTGpDsG = linalg.solve(GtGpD,G.transpose())
        beta = GTGpDsG.dot(S2)
        seqF = G.dot(beta)
        # compute model statistics
        eGlobal = S2 - seqF
        RSS = np.sum(eGlobal ** 2, axis=0)
        df = np.trace(GTGpDsG.dot(G));
        covA_1 = linalg.solve(GtGpD,GtG)
        covA = linalg.solve(GtGpD.transpose(),covA_1.transpose()).transpose()
        # covariance matrix of our components
        s_square = RSS / (noTimepoints-df-1)
        # Z-value of our parametric component
        Z[i,] = beta[0,:] / np.sqrt(s_square * covA[0,0])
        # compute AICc
        AIC_i = np.log(RSS) + (2 * (df+1)) / (noTimepoints-df-2)
        AIC[i,] = AIC_i

    minAICcIdx = np.argmin(AIC,axis=0)
    Z = Z.transpose()
    Z_minAIC = Z[np.arange(Z.shape[0]), minAICcIdx]

    return Z_minAIC

def semiparamRegressio_VCM(S, T, B, P):
    """Apply semiparametric regression framework with varying coefficient model to imaging data.
    S: m x n data cube with m time series of length n
    T: length n vector of timestamps
    B: non parametric basis
    P: penalty matrix of non-parametric basis
    """
    [noTimepoints, noPixels] = S.shape
    m = np.mean(S,axis=0)
    S2 = S - m

    """
    Build VCM basis
    """
    val = ap.computeBoxcarActivityPattern(T,sigma=30)
    val_neg = np.where(val == 0)[0]

    csep = np.cos((2*np.pi*1/60000) * T);
    ssep = np.sin((2*np.pi*1/60000) * T);
    csep[val_neg] = 0;
    ssep[val_neg] = 0;
    ycos = np.diag(csep);
    ysin = np.diag(ssep);

    """
    modulate B-Spline basis
    magic script to create spline basis [Bsep,~,Dsep] = computePenBSplineBasis(noTimepoints,2,1,10);
    """
    Bsep = bspline.createBasis(orderSpline = 2, noKnots = 10);
    BcosSEP = ycos.dot(Bsep).transpose();
    BsinSEP = ysin.dot(Bsep).transpose();

    nothing,noFixedEffects = Bsep.shape
    noFixedEffects *= 2

    """
    create design matrix
    """
    G = np.concatenate([BcosSEP, BsinSEP, B],axis=0).transpose();
    GwithoutFixedEffects = B.transpose()
    """
    fit model
    """
    # compute Penalty term
    #E1 = 0 * np.eye(noFixedEffects)
    E1 = np.diff(np.eye(int(noFixedEffects / 2))).transpose()
    S_P = linalg.block_diag(E1,E1,P)
    Pterm = S_P.transpose().dot(S_P)
    # allocate intermediate storage
    lambdas= np.linspace(0.1,10,10)
    lambdas = [1]
    GtG = G.transpose().dot(G)
    GtGwoFE = GwithoutFixedEffects.transpose().dot(GwithoutFixedEffects)
    AIC = np.zeros([len(lambdas),noPixels])
    F = np.zeros([len(lambdas),noPixels])
    for i in range(0,len(lambdas)):
        """
        fixed effects
        """
        lambda_i = lambdas[i]
        GtGpD = GtG + lambda_i * Pterm;
        GTGpDsG = linalg.solve(GtGpD,G.transpose())
        beta = GTGpDsG.dot(S2)
        seqF = G.dot(beta)
        # compute model statistics
        eGlobal = S2 - seqF
        RSS = np.sum(eGlobal ** 2, axis=0)
        df = np.trace(GTGpDsG.dot(G));

        """
        without fixed effects
        """
        GtGpD = GtGwoFE + lambda_i * P;
        GTGpDsG = linalg.solve(GtGpD,GwithoutFixedEffects.transpose())
        beta = GTGpDsG.dot(S2)
        seqF = GwithoutFixedEffects.dot(beta)
        eGlobal = S2 - seqF
        RSS_woFE = np.sum(eGlobal ** 2, axis=0)
        df_woFE = df - np.trace(GTGpDsG.dot(G));

        """
        AICc, F-value computation
        """

        F_i = ((RSS_woFE - RSS) / df_woFE)  /    (RSS / (noTimepoints - df))
        F[i,] = F_i

        AIC_i = np.log(RSS) + (2 * (df+1)) / (noTimepoints-df-2)
        AIC[i,] = AIC_i

    minAICcIdx = np.argmin(AIC,axis=0)
    F = F.transpose()
    F_minAIC = F[np.arange(F.shape[0]), minAICcIdx]

    return F_minAIC