from scipy import special
import mpmath as mp

def spherical_jn_mp(n, z):
    # j_n(z) = sqrt(pi/(2z)) * J_{n+1/2}(z)
    return mp.sqrt(mp.pi/(2*z)) * mp.besselj(n + mp.mpf('0.5'), z)

def spherical_jn_prime_mp(n, z):
    # j_n'(z) = j_{n-1}(z) - (n+1)/z * j_n(z)   (for n>=1)
    jn = spherical_jn_mp(n, z)
    if n == 0:
        # j0'(z) = -j1(z)
        return -spherical_jn_mp(1, z)
    jn_minus1 = spherical_jn_mp(n-1, z)
    return jn_minus1 - (n + 1)/z * jn

def Dn(z, n):
    # D_n(z) = psi'(z)/psi(z), psi=z*j_n
    jn  = spherical_jn_mp(n, z)
    jnd = spherical_jn_prime_mp(n, z)
    psi  = z * jn
    psid = jn + z * jnd
    return psid / psi
"""
def Dn(z,n):
    jn = special.spherical_jn(n,z)
    jnd = special.spherical_jn(n,z, derivative=True)
    
    psi = z * jn
    psid = jn + z * jnd
    return psid/psi 
"""

def ricabes1(v,z):
    jn = mp.sqrt(0.5*mp.pi/z) * mp.besselj(v+0.5,z, 0)
    jnl = mp.sqrt(0.5*mp.pi/z)* mp.besselj(v-0.5, z,0)
    
    psi1 = z * jn
    psid = z * jnl - v * jn # derive from analytic
    psi2 = psid + psi1 * v/z

    return psi1, psi2

def ricabes3(v,z):
    jn = mp.sqrt(0.5*mp.pi/z)*mp.besselj(v+0.5,z, 0)
    yn = mp.sqrt(0.5*mp.pi/z)*mp.bessely(v+0.5,z, 0)
    hn = jn + 1j*yn
    
    jnl = mp.sqrt(0.5*mp.pi/z)*mp.besselj(v-0.5,z, 0)
    ynl = mp.sqrt(0.5*mp.pi/z)*mp.bessely(v-0.5,z, 0)
    hnl = jnl + 1j*ynl

    zeta1 = z * hn
    zetad = z * hnl - v * hn
    zeta2 = zetad + zeta1 * v/z

    return zeta1, zeta2

def coef_ab(x,m):
    res_a1 = (m**2-1)/3
    res_a11 = 1-(x**2/10)+(4 * m**2 + 5) * x**4/1400
    
    res_D3 = (8 * m**4 - 385 * m**2 + 350) * x**4 / 1400
    res_D4 = 2j * ( m**2 - 1 ) * x**3/3 * (1 - x**2/10)
    D = m**2 + 2 + (1-(7 * m**2)/10) * x**2 - res_D3 + res_D4
    a1 = 2j*res_a1*res_a11/D

    res_b1 = 1+(2 * m**2 - 5) * x**2/70
    res_b11 = 1-(2 * m**2 - 5) * x**2/30
    b1 = 1j * x**2 * (m**2 - 1)/45 * res_b1/res_b11

    res_a2 = 1 - x**2/14
    res_a22 = 2 * m**2 + 3 - (2 * m**2 -7) * x**2/14
    a2 = 1j * x**2 * (m**2-1)/15 * res_a2/res_a22

    return a1,b1,a2


