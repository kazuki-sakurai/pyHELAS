import os, sys, math
import numpy as np
import vector
from math import sqrt, cos, sin, pi

# This code is based on the HELAS paper: https://lib-extopc.kek.jp/preprints/PDF/1991/9124/9124011.pdf
# We use the convention in which the 1j factor is absent in the coupling and the propagators.

####################
# External lines 
####################
# IXXXXX(p, hel, is_fermion): fermion or anti-fermion line flowing-in the blob. 
# OXXXXX(p, hel, is_fermion): fermion anti-fermion flowing-out the blob
# VXXXXX(p, hel, in_out): vector boson line, in or out
# SXXXXX(p, in_out): scalar boson line, in or out

####################
# FFV vertex 
####################
# IOVXXXX(FI,FO,VC,G): AMPLITUDE out of FI FO VC 
# FVIXXX(FI,VC,G,FMASS,FWIDTH): FI out of FI VC  
# FVOXXX(FO,VC,G,FMASS,FWIDTH): FO out of FO VC
# JIOXXX(FI,FO,G,VMASS,VWIDTH) J^mu out of FI FO
# J3XXXX(FI,FO, GAF,GZF ,m,width, Info): J3^mu out of FI FO

####################
# FFS vertex 
####################
# IOSXXX(FI,FO,SC,GC): AMPLITUDE out of FI FO SC
# FSIXXX(FI,SC,GC,FMASS,FWIDTH): FI out of FI SC
# FSOXXX(FO,SC,GC,FMASS,FWIDTH): FO out of FO SC
# HIOXXX(FI,FO,GC,SMASS,SWIDTH): SC out of FI FO

####################
# VVV vertex 
####################
# VVVXXX(WM,WP,W3,G): AMPLITUDE out of WM WP W3
# JVVXXX(V1,V2,G,VMASS,VWIDTH): J^mu out of V1 V2

####################
# VVS vertex 
####################
# VVSXXX(V1,V2,SC,G): AMPLITUDE out of V1 V2 SC
# JVSXXX(VC,SC,G,VMASS,VWIDTH): J^mu out of VC SC
# HVVXXX(V1,V2,G,SMASS,SWIDTH): SC out of V1 V2


class Structure: pass

Gf = 1.16639e-5
aEM = 1/132.507
mZ = 91.188
r2 = sqrt(2.)
thW = 1/2 *np.arcsin(2*sqrt( pi/r2 *aEM /(Gf *mZ**2)))
sinW = np.sin(thW)
cosW = np.cos(thW)
gEM = np.sqrt(4*pi*aEM)
gW =gEM/sinW
vev = (sqrt(2) * Gf)**(-1/2)
mW = mZ*cosW

gev2tobarn = 0.3894e-3

Info = {
    'e': {'m':0, 'w':0, 'Q':-1, 'T3':-1/2},
    't': {'m':173, 'w':1.4915, 'Q':2/3, 'T3':1/2}, 
    'b': {'m':4.2, 'w':0, 'Q':-1/3, 'T3':-1/2}, 
    'mu': {'m':0.106, 'w':0, 'Q':-1, 'T3':-1/2},
    'gamma': {'m':0, 'w':0},
    'Z': {'m':91.188, 'w':2.441404}, 
    'W': {'m':80.4, 'w':2.1},  
    'const': {'Gf':Gf, 'aEM':aEM, 'thW':thW, 'sinW':sinW, 'cosW':cosW, 'gEM':gEM, 'gW':gW, 'vev': vev },
    'G': {'WF': [gW, 0], 'WWA': gW*sinW, 'WWZ': gW*cosW, 'hZZ': 2j*mZ**2/vev, 'hWW': 2j*mW**2/vev}
    }

Q, T3 = 2/3, 1/2
Info['G']['AUU'] = gEM*Q * np.array([1, 1])
Info['G']['ZUU'] = gW/cosW * np.array([(T3 - sinW**2 * Q), (0 - sinW**2 * Q)])

Q, T3 = -1/3, -1/2
Info['G']['ADD'] = gEM*Q * np.array([1, 1])
Info['G']['ZDD'] = gW/cosW * np.array([(T3 - sinW**2 * Q), (0 - sinW**2 * Q)])

Q, T3 = -1, -1/2
Info['G']['AEE'] = gEM*Q * np.array([1, 1])
Info['G']['ZEE'] = gW/cosW * np.array([(T3 - sinW**2 * Q), (0 - sinW**2 * Q)])

Q, T3 = 0, 1/2
Info['G']['ANN'] = gEM*Q * np.array([1, 1])
Info['G']['ZNN'] = gW/cosW * np.array([(T3 - sinW**2 * Q), (0 - sinW**2 * Q)])


gmet = np.eye(4); gmet[1,1], gmet[2,2], gmet[3,3] = -1, -1, -1
gmetD = np.array([1, -1, -1, -1])

I2 = np.eye(2)
I4 = np.eye(4)
Z2 = np.zeros((2, 2))

Pau0 = np.eye(2)
Pau1 = np.array([[0,1],[1,0]])
Pau2 = np.array([[0,-1j],[1j,0]])
Pau3 = np.array([[1,0],[0,-1]])
Pau = [Pau0, Pau1, Pau2, Pau3]

gam0 = np.block( [[Z2, Pau0],[Pau0, Z2]] )
gam1 = np.block( [[Z2, Pau1],[-Pau1, Z2]] )
gam2 = np.block( [[Z2, Pau2],[-Pau2, Z2]] )
gam3 = np.block( [[Z2, Pau3],[-Pau3, Z2]] )
gam = [gam0, gam1, gam2, gam3]
gam5 = np.block( [[-I2, Z2],[Z2, I2]] )

PL = (I4 - gam5)/2
PR = (I4 + gam5)/2

def pmag(p): return np.sqrt(p[1]**2 + p[2]**2 + p[3]**2)

def get_angles(p):
    if pmag(p) == 0: return 0, 0
    theta = np.arccos(p[3] / pmag(p))       # polar angle
    phi = np.arctan2(p[2], p[1])        # azimuthal angle    
    return theta, phi

def get_shashed( vec ): return vec[0]*gam0 - vec[1]*gam1 - vec[2]*gam2 - vec[3]*gam3

def Ldot( v1, v2 ): return v1[0]*v2[0] - v1[1]*v2[1] - v1[2]*v2[2] - v1[3]*v2[3]

def get_mass(p): 
    msq = Ldot( p,p )
    if msq >= 0: return np.sqrt(msq)
    if msq < 0: return - np.sqrt( abs(msq) )

#def get_vector( ar ): return vector.obj(E=ar[0],px=ar[1],py=ar[2],pz=ar[3])

def get_scalar_propagator(p, m, width):       
    #return 1j/( Ldot(p,p) - m**2 + 1j*m*width )     
    return 1/( Ldot(p,p) - m**2 + 1j*m*width )     

def get_fermion_propagator(p, m, width):
    pslash = get_shashed( p )    
    mmat = m * I4
    #return 1j/( Ldot(p,p) - m**2 + 1j*m*width ) * (pslash + mmat)    
    return 1/( Ldot(p,p) - m**2 + 1j*m*width ) * (pslash + mmat)    

def get_gauge_propagator(p, m, width):     
    denom = (Ldot(p,p) - m**2 + 1j*m*width)
    if m == 0:
        matrix = -gmet
    else:
        matrix = -gmet + (1/m**2)*np.outer(p, p)
    #return 1j/denom * matrix
    return 1/denom * matrix

def chi_plus(p):
    theta, phi = get_angles(p) 
    return np.array([cos(theta/2), np.exp( 1j*phi ) * sin(theta/2)])
def chi_minus(p): 
    theta, phi = get_angles(p) 
    return np.array([- np.exp( 1j*phi ) * sin(theta/2), cos(theta/2)])
def get_chi(p):
    chi = {}
    chi[ 1] = chi_plus(p)
    chi[-1] = chi_minus(p)
    return chi

def get_omeg(p):
    omeg = {}
    omeg[ 1] = np.sqrt(p[0] + pmag(p))
    omeg[-1] = np.sqrt(p[0] - pmag(p))
    return omeg    

def get_pol(p, hel): 
    E, m = p[0], get_mass(p)
    th, ph = get_angles(p)
    if hel not in [1, -1, 0]:
        print('ERROR: vector boson helicity must be in [1, 0, -1]!!!')
        exit()
    if hel in [1, -1]:
        e1 = np.array([0, cos(th)*cos(ph), cos(th)*sin(ph), -sin(th)])
        e2 = np.array([0, -sin(ph), cos(ph), 0])
        ep = ( - hel * e1 + 1j*e2 )/sqrt(2) 
    if hel == 0:
        ep = np.array([pmag(p)/m, (E/m)*sin(th)*cos(ph), (E/m)*sin(th)*sin(ph), (E/m)*cos(th)])
    return ep

def IXXXXX(p, hel, is_fermion): 
    omeg = get_omeg(p)
    chi = get_chi(p)
    if is_fermion in ['fermion', 'f', 'fer']: 
        sp = [ omeg[-hel]*chi[hel], omeg[hel]*chi[hel] ]
        p_save = p
    else: 
        sp = [ -hel*omeg[hel]*chi[-hel], hel*omeg[-hel]*chi[-hel] ]         
        p_save = - p
    data = Structure()
    data.p = p_save
    data.sp = np.block(sp)
    return data

def OXXXXX(p, hel, is_fermion): 
    dataI = IXXXXX(p, hel, is_fermion)
    spbar = dataI.sp.conj() @ gam0 
    data = Structure()
    data.p = dataI.p
    data.spbar = spbar
    return data

def VXXXXX(p, hel, in_out): 

    pol = get_pol(p, hel)
    if in_out == 'out': 
        p_save = p
        pol = pol.conj()
    if in_out == 'in':
        p_save = -p
    data = Structure()
    data.p = p_save
    data.pol = pol
    return data

def SXXXXX(p, in_out): 
    if in_out == 'out':
        p_save = p
    elif in_out == 'in':
        p_save = -p
    data = Structure()
    data.p = p_save
    data.WF = 1
    return data

###
def IOVXXX(FI, FO, VC, G):
    sp = FI.sp 
    spbar = FO.spbar 
    pol = VC.pol
    COUP = G[0]*PL + G[1]*PR
    vdm = [ spbar @ gam[mu] @ COUP @ sp for mu in range(4) ]
    fgf = np.array( vdm )
    amp = Ldot( fgf, pol )
    return amp

def FVIXXX(FI,VC,G,FMASS,FWIDTH):
    sp = FI.sp 
    pol_slash = get_shashed( VC.pol )
    #COUP = 1j*(G[0]*PL + G[1]*PR)
    COUP = G[0]*PL + G[1]*PR # changed
    p_out = FI.p - VC.p
    prop = get_fermion_propagator(p_out, FMASS, FWIDTH)
    sp_out = prop @ pol_slash @ COUP @ sp    
    data = Structure()
    data.p = p_out
    data.sp = sp_out
    return data

def FVOXXX(FO,VC,G,FMASS,FWIDTH):
    spbar = FO.spbar 
    pol_slash = get_shashed( VC.pol )
    #COUP = 1j*(G[0]*PL + G[1]*PR)
    COUP = G[0]*PL + G[1]*PR # changed
    p_out = FO.p + VC.p
    prop = get_fermion_propagator(p_out, FMASS, FWIDTH)
    spbar_out = spbar @ pol_slash @ COUP @ prop     
    data = Structure()
    data.p = p_out
    data.spbar = spbar_out
    return data

def JIOXXX(FI,FO,G,VMASS,VWIDTH):

    q = - FI.p + FO.p

    sp = FI.sp 
    spbar = FO.spbar 
    #COUP = 1j*(G[0]*PL + G[1]*PR)
    COUP = G[0]*PL + G[1]*PR # changed 
    prop = get_gauge_propagator(q,VMASS,VWIDTH)
    pol_out = []
    for mu in range(4):
        val = 0
        for nu in range(4):
            val += prop[mu,nu] * gmetD[nu] * spbar @ gam[nu] @ COUP @ sp
        pol_out.append(val)
    data = Structure()
    data.p = q
    data.pol = np.array(pol_out)
    return data

def J3XXXX(FI,FO, GAF,GZF ,m,width, Info):

    q = - FI.p + FO.p

    sW = Info['const']['sinW']
    cW = Info['const']['cosW']
    eQ = GAF[0] 

    JL = np.array([FO.spbar @ gam[mu] @ PL @ FI.sp for mu in range(4)])
    JR = np.array([FO.spbar @ gam[mu] @ PR @ FI.sp for mu in range(4)])

    DZ = Ldot(q,q) - m**2 + 1j*m*width
    DA = Ldot(q,q) 

    term1 = (-cW/DZ * GZF[0] - eQ*sW/DA) * JL  
    term2 = cW/(DZ *m**2) * (GZF[0]*Ldot(q,JL) + GZF[1]*Ldot(q,JR)) * q   
    term3 = eQ*sinW * (m**2 - 1j*m*width) / (DA * DZ) * JR  

    #J3 = 1j*(term1 + term2 + term3)
    J3 = term1 + term2 + term3

    data = Structure()
    data.p = q
    data.pol = J3 
    return data

def J3XXXX2(FI,FO,GAF,GZF,ZMASS,ZWIDTH, Info):

    q = - FI.p + FO.p

    Z = JIOXXX(FI,FO,GZF,ZMASS,ZWIDTH)
    A = JIOXXX(FI,FO,GAF,0,0)

    sinW = Info['const']['sinW']
    cosW = Info['const']['cosW']

    pol_out = cosW * Z.pol + sinW * A.pol
    data = Structure()
    data.p = q
    data.pol = np.array(pol_out)
    return data

def J3XXXX3(FI,FO, GAF,GZF ,m,width, Info):

    q = -FI.p + FO.p

    JL = np.array([FO.spbar @ gam[mu] @ PL @ FI.sp for mu in range(4)])
    JR = np.array([FO.spbar @ gam[mu] @ PR @ FI.sp for mu in range(4)])

    sW = Info['const']['sinW']
    cW = Info['const']['cosW']
    eQ = GAF[0] 

    prop_Z = get_gauge_propagator(q,m,width)
    prop_A = get_gauge_propagator(q,0,0)

    term_Z, term_A = [], []
    for mu in range(4):
        tmpZ, tmpA = 0, 0
        for nu in range(4):    
            #tmpZ += cW * prop_Z[mu,nu] * gmetD[nu] * 1j*( GZF[0]*JL[nu] + GZF[1]*JR[nu]) ### extra minus sign!!
            #tmpA += sW * prop_A[mu,nu] * gmetD[nu] * 1j*eQ*( JL[nu] + JR[nu]) ### extra minus sign!!
            tmpZ += cW * prop_Z[mu,nu] * gmetD[nu] * ( GZF[0]*JL[nu] + GZF[1]*JR[nu]) ### 1j removed 
            tmpA += sW * prop_A[mu,nu] * gmetD[nu] * eQ*( JL[nu] + JR[nu]) ### 1j removed
        term_Z.append(tmpZ)
        term_A.append(tmpA)
    term_Z = np.array(term_Z)
    term_A = np.array(term_A)
    J3 = term_Z + term_A

    data = Structure()
    data.p = q
    data.pol = J3 
    return data

def IOSXXX(FI,FO,SC,GC):
    COUP = GC[0]*PL + GC[1]*PR
    amp = SC.WF * FO.spbar @ COUP @ FI.sp
    return amp  

def FSIXXX(FI,SC,GC,FMASS,FWIDTH):
    p_out = FI.p - SC.p    
    #COUP = 1j*(GC[0]*PL + GC[1]*PR) 
    COUP = GC[0]*PL + GC[1]*PR # changed 
    prop = get_fermion_propagator(p_out, FMASS, FWIDTH)
    sp_out = SC.WF * prop @ COUP @ FI.sp 
    data = Structure()
    data.p = p_out
    data.sp = sp_out
    return data
       
def FSOXXX(FO,SC,GC,FMASS,FWIDTH):
    p_out = FO.p + SC.p
    COUP = GC[0]*PL + GC[1]*PR
    prop = get_fermion_propagator(p_out, FMASS, FWIDTH)
    spbar_out = SC.WF * FO.spbar @ COUP @ prop
    data = Structure()
    data.p = p_out
    data.spbar = spbar_out
    return data

def HIOXXX(FI,FO,GC,SMASS,SWIDTH):
    COUP = GC[0]*PL + GC[1]*PR
    p_out = -FI.p + FO.p 
    prop = get_scalar_propagator(p_out, SMASS,SWIDTH)       
    WF = prop * FO.spbar @ COUP @ FI.sp
    data = Structure()
    data.p = p_out
    data.WF = WF
    return data

def VVSXXX(V1,V2,SC,G):
    amp = G * SC.WF * Ldot(V1.pol, V2.pol)
    return amp

###
def JVSXXX(VC,SC,G,VMASS,VWIDTH):
    q = VC.p + SC.p    
    prop = get_gauge_propagator(q,VMASS,VWIDTH)       
    vec = []
    for mu in range(4):    
        tmp = 0
        for nu in range(4):    
            tmp += prop[mu,nu] * gmetD[nu] * VC.pol[nu]            
        vec.append(tmp)
    pol= G * SC.WF * np.array( vec )  
    #pol= 1j * G * SC.WF * np.array( vec )  # changed   
    data = Structure()
    data.p = q
    data.pol = pol
    return data

# def HVVXXX(V1,V2,G,SMASS,SWIDTH):
#     p_out = V1.p + V2.p
#     prop = get_scalar_propagator(p_out,SMASS,SWIDTH)
#     WF = G * prop * Ldot( V1.pol, V2.pol )
#     data = Structure()
#     data.p = p_out
#     data.WF = WF
#     return data


### VVV vertex
def VVVXXX(WM,WP,W3,G):

    term1 = Ldot( WM.p - WP.p, W3.pol ) * Ldot( WM.pol, WP.pol )
    term2 = Ldot( WP.p - W3.p, WM.pol ) * Ldot( WP.pol, W3.pol )
    term3 = Ldot( W3.p - WM.p, WP.pol ) * Ldot( W3.pol, WM.pol )

    # ap = WP.p[0] / WP.pol[0]
    # am = WM.p[0] / WM.pol[0]
    # a3 = W3.p[0] / W3.pol[0]
    # term1 = Ldot( WM.p - am*WM.pol - WP.p + ap*WP.pol, W3.pol ) * Ldot( WM.pol, WP.pol )
    # term2 = Ldot( WP.p - ap*WP.pol - W3.p + a3*W3.pol, WM.pol ) * Ldot( WP.pol, W3.pol )
    # term3 = Ldot( W3.p - a3*W3.pol - WM.p + am*WM.pol, WP.pol ) * Ldot( W3.pol, WM.pol )

    amp = G * (term1 + term2 + term3)

    # The “−” sgin in the manual is written in the incoming-momentum convention.
    # When we convert to the subroutine-style all-outgoing convention, 
    # each tensor picks up a minus sign, so the overall prefactor becomes +G.

    return amp



def JVVXXX(V1,V2,G,VMASS,VWIDTH):
    p1, p2 = V1.p, V2.p
    q = V1.p + V2.p
    J12 = Ldot(V1.pol, V2.pol) * (p1 - p2)   
    J12 += Ldot(p2 - q, V1.pol ) * V2.pol
    J12 += Ldot(q - p1, V2.pol) * V1.pol

    if VMASS == 0:
        JS = 0
    else:
        JS  = (-Ldot(p1,p1) + Ldot(p2,p2)) * Ldot(V1.pol, V2.pol)
        JS += Ldot(p1,V1.pol) * Ldot(p1,V2.pol) 
        JS -= Ldot(p2,V1.pol) * Ldot(p2,V2.pol) 
        JS = JS/VMASS**2

    #prop = -1j/( Ldot(q,q) - VMASS**2 + 1j*VMASS*VWIDTH )
    prop = -1/( Ldot(q,q) - VMASS**2 + 1j*VMASS*VWIDTH ) # changed
    #Jout = 1j*G*prop*( J12 - JS*q ) 
    Jout = G*prop*( J12 - JS*q ) # changed
    data = Structure()
    data.p = q
    data.pol = Jout
    return data


# px, py, pz = [np.random.uniform() for i in range(3)]
# pa = vector.obj(px=px, py=py, pz=pz, mass=ma)

# FO = OXXXXX(pa, 1, 'f')
# FI = IXXXXX(pa, 1, 'fbar')
# VC = VXXXXX(pa, 1, 'out')

# G = [0.1, 0.3]
# amp = IOVXXX(FI, FO, VC, G)

# print('amp', amp )

