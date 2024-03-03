#References：
#1. H. Li, Z. Wang, N. Zou, M. Ye, R. Xu, X. Gong, W. Duan, Y. Xu, Deep-learning density functional theory Hamiltonian for efficient ab initio electronic-structure calculation, Nat. Comput. Sci. 2(6) (2022) 367-377.

#This code is extended from https://github.com/mzjb/DeepH-pack.git, which has the GNU LESSER GENERAL PUBLIC LICENSE. 

# Thanks to the original authors for their generous sharing. If you need to use the code, please refer to the original version of the code and literature.

import torch
import numpy as np


def semifactorial(x):
    """Compute the semifactorial function x!!.

    x!! = x * (x-2) * (x-4) *...

    Args:
        x: positive int
    Returns:
        float for x!!
    """
    y = 1.
    for n in range(x, 1, -2):
        y *= n
    return y


def pochhammer(x, k):
    """Compute the pochhammer symbol (x)_k.

    (x)_k = x * (x+1) * (x+2) *...* (x+k-1)

    Args:
        x: positive int
    Returns:
        float for (x)_k
    """
    xf = float(x)
    for n in range(x+1, x+k):
        xf *= n
    return xf

def lpmv(l, m, x):
    """Associated Legendre function including Condon-Shortley phase.

    Args:
        m: int order 
        l: int degree
        x: float argument tensor
    Returns:
        tensor of x-shape
    """
    m_abs = abs(m)
    if m_abs > l:
        return torch.zeros_like(x)

    # Compute P_m^m
    yold = ((-1)**m_abs * semifactorial(2*m_abs-1)) * torch.pow(1-x*x, m_abs/2)
    
    # Compute P_{m+1}^m
    if m_abs != l:
        y = x * (2*m_abs+1) * yold
    else:
        y = yold

    # Compute P_{l}^m from recursion in P_{l-1}^m and P_{l-2}^m
    for i in range(m_abs+2, l+1):
        tmp = y
        # Inplace speedup
        y = ((2*i-1) / (i-m_abs)) * x * y
        y -= ((i+m_abs-1)/(i-m_abs)) * yold
        yold = tmp

    if m < 0:
        y *= ((-1)**m / pochhammer(l+m+1, -2*m))

    return y

def tesseral_harmonics(l, m, theta=0., phi=0.):
    """Tesseral spherical harmonic with Condon-Shortley phase.

    The Tesseral spherical harmonics are also known as the real spherical
    harmonics.

    Args:
        l: int for degree
        m: int for order, where -l <= m < l
        theta: collatitude or polar angle
        phi: longitude or azimuth
    Returns:
        tensor of shape theta
    """
    assert abs(m) <= l, "absolute value of order m must be <= degree l"

    N = np.sqrt((2*l+1) / (4*np.pi))
    leg = lpmv(l, abs(m), torch.cos(theta))
    if m == 0:
        return N*leg
    elif m > 0:
        Y = torch.cos(m*phi) * leg
    else:
        Y = torch.sin(abs(m)*phi) * leg
    N *= np.sqrt(2. / pochhammer(l-abs(m)+1, 2*abs(m)))
    Y *= N
    return Y

class SphericalHarmonics(object):
    def __init__(self):
        self.leg = {}

    def clear(self):
        self.leg = {}

    def negative_lpmv(self, l, m, y):
        """Compute negative order coefficients"""
        if m < 0:
            y *= ((-1)**m / pochhammer(l+m+1, -2*m))
        return y

    def lpmv(self, l, m, x):
        """Associated Legendre function including Condon-Shortley phase.

        Args:
            m: int order 
            l: int degree
            x: float argument tensor
        Returns:
            tensor of x-shape
        """
        # Check memoized versions
        m_abs = abs(m)
        if (l,m) in self.leg:
            return self.leg[(l,m)]
        elif m_abs > l:
            return None
        elif l == 0:
            self.leg[(l,m)] = torch.ones_like(x)
            return self.leg[(l,m)]
        
        # Check if on boundary else recurse solution down to boundary
        if m_abs == l:
            # Compute P_m^m
            y = (-1)**m_abs * semifactorial(2*m_abs-1)
            y *= torch.pow(1-x*x, m_abs/2)
            self.leg[(l,m)] = self.negative_lpmv(l, m, y)
            return self.leg[(l,m)]
        else:
            # Recursively precompute lower degree harmonics
            self.lpmv(l-1, m, x)

        # Compute P_{l}^m from recursion in P_{l-1}^m and P_{l-2}^m
        # Inplace speedup
        y = ((2*l-1) / (l-m_abs)) * x * self.lpmv(l-1, m_abs, x)
        if l - m_abs > 1:
            y -= ((l+m_abs-1)/(l-m_abs)) * self.leg[(l-2, m_abs)]
        #self.leg[(l, m_abs)] = y
        
        if m < 0:
            y = self.negative_lpmv(l, m, y)
        self.leg[(l,m)] = y

        return self.leg[(l,m)]

    def get_element(self, l, m, theta, phi):
        """Tesseral spherical harmonic with Condon-Shortley phase.

        The Tesseral spherical harmonics are also known as the real spherical
        harmonics.

        Args:
            l: int for degree
            m: int for order, where -l <= m < l
            theta: collatitude or polar angle
            phi: longitude or azimuth
        Returns:
            tensor of shape theta
        """
        assert abs(m) <= l, "absolute value of order m must be <= degree l"

        N = np.sqrt((2*l+1) / (4*np.pi))
        leg = self.lpmv(l, abs(m), torch.cos(theta))
        if m == 0:
            return N*leg
        elif m > 0:
            Y = torch.cos(m*phi) * leg
        else:
            Y = torch.sin(abs(m)*phi) * leg
        N *= np.sqrt(2. / pochhammer(l-abs(m)+1, 2*abs(m)))
        Y *= N
        return Y

    def get(self, l, theta, phi, refresh=True):
        """Tesseral harmonic with Condon-Shortley phase.

        The Tesseral spherical harmonics are also known as the real spherical
        harmonics.

        Args:
            l: int for degree
            theta: collatitude or polar angle
            phi: longitude or azimuth
        Returns:
            tensor of shape [*theta.shape, 2*l+1]
        """
        results = []
        if refresh:
            self.clear()
        for m in range(-l, l+1):
            results.append(self.get_element(l, m, theta, phi))
        return torch.stack(results, -1)
