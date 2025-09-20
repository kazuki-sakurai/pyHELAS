import numpy as np
from math import sqrt, cos, sin, acos, asin
import pyHELAS 
import vector
import math

def pretty_output(header, lis, mergin=1):
    lis = np.array(lis)
    tlis = lis.transpose()
    widths = []
    for i in range(len(header)):
        str_tlis = [ len(str(r)) for r in tlis[i] ] + [len(header[i])]
        w = np.max( str_tlis ) + mergin
        widths.append(w)

    print(" ".join(f"{h:>{w}}" for h, w in zip(header, widths)))
    print(" ".join("-"*(w) for w in widths))

    for ar in lis:
        print(" ".join(f"{elem:>{w}}" for elem, w in zip(ar, widths)))


def vec2ar(v): return np.array([v.E, v.px, v.py, v.pz])

def get_momenta(cth, msq):

    sth = sqrt(1 - cth**2)

    #msq = np.random.uniform(0,mmax**2)
    mbar = sqrt(msq)

    numsq = M**4 - 2*(mv**2 + mbar**2)*M**2 + (mv**2 - mbar**2)**2  
    q = sqrt(numsq)/(2*M)

    EAB = (M**2 + mbar**2 - mv**2) / (2*M)
    EV = (M**2 - mbar**2 + mv**2) / (2*M)

    print('M',M)
    print('mbar',mbar)
    print('mv',mv)
    print('EV',EV)

    beta = q/EAB
    gam = EAB/mbar

    mstr = sqrt( mbar**2 - 4*mf**2 )

    pp = {}

    pAx, pAz = (1/2)*mstr*sth, (1/2)*gam*(mstr*cth - mbar*beta) 
    vA = vector.obj( px=pAx, py=0, pz=pAz, mass=mf )
    pp['A'] = vec2ar(vA)

    pBx, pBz = -(1/2)*mstr*sth, -(1/2)*gam*(mstr*cth + mbar*beta) 
    vB = vector.obj( px=pBx, py=0, pz=pBz, mass=mf )
    pp['B'] = vec2ar(vB)

    vZ = vector.obj( px=0, py=0, pz=q, mass=mZ )
    pp['Z'] = vec2ar(vZ)

    vH = vector.obj( px=0, py=0, pz=0, mass=mH )
    pp['H'] = vec2ar(vH)

    return pp



def get_rho_pure(cth, msq):

    dim = 2*2*3

    lA_ar = [1, -1]
    lB_ar = [1, -1]
    lZ_ar = [1, -1, 0]

    #cth = np.random.uniform(-1,1)
    sth = sqrt(1 - cth**2)

    #msq = np.random.uniform(0,mmax**2)
    mbar = sqrt(msq)

    numsq = M**4 - 2*(mv**2 + mbar**2)*M**2 + (mv**2 - mbar**2)**2  
    q = sqrt(numsq)/(2*M)

    EAB = (M**2 + mbar**2 - mv**2) / (2*M)
    EV = (M**2 - mbar**2 + mv**2) / (2*M)

    beta = q/EAB
    gam = EAB/mbar

    sA = sth / (gam*(1 - beta*cth))
    cA = (cth - beta)/(1 - beta*cth)
    sB = sth/( gam*(1 + beta*cth) )
    cB = - (cth + beta) / (1 + beta*cth)

    thA = np.sign(sA) * acos(cA)
    thB = np.sign(sB) * acos(cB)

    sp = sin( (thA + thB)/2 )  
    sm = sin( (thA - thB)/2 )  
    cp = cos( (thA + thB)/2 )  
    cm = cos( (thA - thB)/2 )  

    r2 = 1/sqrt(2)

    A = {}
    for lA in lA_ar:
        for lB in lB_ar:
            for lZ in lZ_ar:
                A[(lA,lB,lZ)] = 0

    A[(-1,1,1)] = - r2 * cL * (sp + sm)
    A[(-1,1,-1)] = - r2 * cL * (sp - sm)
    A[(-1,1,0)] = - (1/mv) * cL * (q*cp - EV*cm)

    A[(1,-1,1)] = + r2 * cR * (sp - sm)
    A[(1,-1,-1)] = + r2 * cR * (sp + sm)
    A[(1,-1,0)] = - (1/mv) * cR * (q*cp - EV*cm)

    Nsq = 0
    for lA in lA_ar:
        for lB in lB_ar:
            for lZ in lZ_ar:
                Nsq += A[(lA,lB,lZ)]**2

    MM = [] 
    for lA in lA_ar:
        for lB in lB_ar:
            for lZ in lZ_ar:
                A[(lA,lB,lZ)] = A[(lA,lB,lZ)]/sqrt(Nsq)
                #print( '(',lA,lB,lZ,')', ' ', A[(lA,lB,lZ)] )
                MM.append(A[(lA,lB,lZ)])

    rho = np.zeros((dim,dim))

    tr = 0
    for i in range(dim):
        for j in range(dim):
            rho[i,j] = MM[i]*MM[j]
            if i == j: tr += rho[i,j] 
            #print( '({},{})  {}'.format(i,j,rho[(i,j)])  )
    #print('tr = ',tr)
    return rho, A


mH = M = 125.
mZ = mv = 91.1880
Zwidth = 0
g_HZZ = 1
mf = 10

if M - mv - 2*mf < 0:
    print('ERROR: too large fermion mass')
    print('The fermion mass must be less than', (M - mv)/2 )   
    exit()
#cR, cL = 1, 1
cR, cL = 0.2312, -0.2688

# cth = np.random.uniform(-1,1)
# mmax = (M - mv)
# mmin = 2*mf
# msq = np.random.uniform( mmin**2, mmax**2 )

cth = 1/3
mbar = 25
msq = mbar**2

rho, AA = get_rho_pure(cth, msq)

pp = get_momenta(cth, msq)

norm1_sq = 0
norm2_sq = 0
Amp1 = {}
Amp2 = {}

print('cos(bar theta) = ', cth)
print('mbar = ', mbar)
print('mf = ', mf)
print('cL, cR = ', cL, cR)
print('pA:', pp['A'])
print('pB:', pp['B'])
print('pZ:', pp['Z'])
print('pH:', pp['H'])

for lA in [1, -1]:
    for lB in [1, -1]:
        for lZ in [1, 0, -1]:

            if lA != -1: continue
            if lB != 1: continue
            if lZ != 0: continue
            print('lA, lB, lZ', lA, lB, lZ)

            # Defining external momenta  
            FO = pyHELAS.OXXXXX(pp['A'], lA, 'fermion') # tau-
            FI = pyHELAS.IXXXXX(pp['B'], lB, 'fbar') # tau+
            VC = pyHELAS.VXXXXX(pp['Z'], lZ, 'out') # Z
            SC = pyHELAS.SXXXXX(pp['H'], 'in') # Higgs

            print('Ubar(A)({}) ='.format(lA), FO.spbar.real)
            print('V(B)({}) ='.format(lB), FI.sp.real)
            print('epsilon*({}) ='.format(lZ), VC.pol)
            PL = pyHELAS.PL
            PR = pyHELAS.PR
            gam = pyHELAS.gam

            JL = np.array([ FO.spbar @ gam[mu] @ PL @ FI.sp for mu in range(4)])
            JR = np.array([ FO.spbar @ gam[mu] @ PR @ FI.sp for mu in range(4)])

            print('Real(JL)',JL.real)
            print('Imag(JL)',JL.imag)
            print('Real(JR)',JR.real)
            print('Imag(JR)',JR.imag)


            kk = pp['A'] + pp['B']

            print('k^mu', kk)
            #kk = pp['H'] - pp['Z']

            JK = pyHELAS.Ldot(cL*JL + cR*JR, kk)
            KE = pyHELAS.Ldot(kk, VC.pol)

            print('(cL*JL + cR*JR).k =', JK.real)
            print('k.epsilon*(0) =', KE.real)


            # print('Ubar(A).gam0 =', FO.spbar @ gam[0])
            # print('Real(Ubar.gam[mu].PL.V) =', JL.real)
            # print('Im(Ubar.gam[mu].PL.V) =', JL.imag)
            # print('Real(Ubar.gam[mu].PR.V) =', JR.real)
            # print('Im(Ubar.gam[mu].PR.V) =', JR.imag)

            # HELAS method 1 ( H + Z -> Z*, Amp = Z + f + fbar )
            Zs = pyHELAS.JVSXXX(VC, SC, g_HZZ, mZ, Zwidth)
            amp1 = pyHELAS.IOVXXX(FI, FO, Zs, [cL, cR])
            Amp1[(lA,lB,lZ)] = amp1

            # HELAS method 2 (f + fbar -> Z*, Amp = Z + Z* + H )
            Zs2 = pyHELAS.JIOXXX(FI,FO,[cL, cR], mZ, Zwidth) 
            amp2 = pyHELAS.VVSXXX(Zs2, VC, SC, g_HZZ) 
            Amp2[(lA,lB,lZ)] = amp2

            #print('amp1, amp2 =', amp1, amp2)

            norm1_sq += np.abs(amp1)**2
            norm2_sq += np.abs(amp2)**2

exit()
header = ['lA', 'lB', 'lZ', 'HELAS1', 'HELAS2', 'Analytic']
outlist = []
for lA in [1, -1]:
    for lB in [1, -1]:
        for lZ in [1, 0, -1]:
            print(lA, lB, lZ, ' ', Amp1[(lA,lB,lZ)].imag)

            outlist.append([ lA, lB, lZ, Amp1[(lA,lB,lZ)].imag, Amp2[(lA,lB,lZ)].imag, AA[lA,lB,lZ] ])

pretty_output( header, outlist )

