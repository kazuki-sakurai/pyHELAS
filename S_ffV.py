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


###################
# Diagram-1 
###################
# Method-A: H + Z -> Z*, Amp = Z + f + fbar )
def get_amp1A(externals, COUP ):
    FO, FI, VC, SC = externals 
    Zs = pyHELAS.JVSXXX(VC, SC, g_HZZ, mZ, Zwidth)
    amp = pyHELAS.IOVXXX(FI, FO, Zs, COUP)
    return amp

# Method-B: f + fbar -> Z*, Amp = Z + Z* + H 
def get_amp1B(externals, COUP ):
    FO, FI, VC, SC = externals 
    Zs = pyHELAS.JIOXXX(FI,FO, COUP, mZ, Zwidth) 
    amp = pyHELAS.VVSXXX(Zs, VC, SC, g_HZZ) 
    return amp

###################
# Diagram-2
###################
# Method-A: fbar + H -> f*, Amp = f* + Z + f )
def get_amp2A(externals, COUP ):
    FO, FI, VC, SC = externals 
    FI_off = pyHELAS.FSIXXX(FI,SC,G_Hff,mf,fwidth)
    amp = pyHELAS.IOVXXX(FI_off,FO,VC,COUP)
    return amp

# Method-B: Z + f -> f*, Amp = f* + H + fbar 
def get_amp2B(externals, COUP ):
    FO, FI, VC, SC = externals 
    FO_off = pyHELAS.FVOXXX(FO,VC,COUP,mf,fwidth)
    amp = pyHELAS.IOSXXX(FI,FO_off,SC,G_Hff)
    return amp

###################
# Diagram-3
###################
# Method-A: f + H -> fbar*, Amp = fbar* + Z + fbar )
def get_amp3A(externals, COUP ):
    FO, FI, VC, SC = externals 
    FO_off = pyHELAS.FSOXXX(FO,SC,G_Hff,mf,fwidth)
    amp = pyHELAS.IOVXXX(FI,FO_off,VC,COUP)
    return amp

# Method-B: Z + fbar -> fbar*, Amp = fbar* + H + f 
def get_amp3B(externals, COUP ):
    FO, FI, VC, SC = externals 
    FI_off = pyHELAS.FVIXXX(FI,VC,COUP,mf,fwidth)
    amp = pyHELAS.IOSXXX(FI_off,FO,SC,G_Hff)
    return amp



mH = M = 125.
mZ = mv = 91.1880
Zwidth = 0
g_HZZ = 1
mf = 10
fwidth = 0
vev = pyHELAS.Info['const']['vev']
G_Hff = - mf/vev * np.array([1, 1])

if M - mv - 2*mf < 0:
    print('ERROR: too large fermion mass')
    print('The fermion mass must be less than', (M - mv)/2 )   
    exit()
#cR, cL = 1, 1
cR, cL = 0.2312, -0.2688
COUP = [cL, cR]


# cth = np.random.uniform(-1,1)
# mmax = (M - mv)
# mmin = 2*mf
# msq = np.random.uniform( mmin**2, mmax**2 )

cth = 1/3
mbar = 25
msq = mbar**2

rho, AA = get_rho_pure(cth, msq)

pp = get_momenta(cth, msq)

print('cos(bar theta) = ', cth)
print('mbar = ', mbar)
print('mf = ', mf)
print('cL, cR = ', cL, cR)
print('pA:', pp['A'])
print('pB:', pp['B'])
print('pZ:', pp['Z'])
print('pH:', pp['H'])

header = ['lA', 'lB', 'lZ', 'AmpTot(Graph1 + Graph2 + Graph3)', 'Amp1(Graph1)']
outlist = []

for lA in [1, -1]:
    for lB in [1, -1]:
        for lZ in [1, 0, -1]:

            # if lA != -1: continue
            # if lB != 1: continue
            # if lZ != 0: continue

            # Defining external momenta  
            FO = pyHELAS.OXXXXX(pp['A'], lA, 'fermion') # tau-
            FI = pyHELAS.IXXXXX(pp['B'], lB, 'fbar') # tau+
            VC = pyHELAS.VXXXXX(pp['Z'], lZ, 'out') # Z
            SC = pyHELAS.SXXXXX(pp['H'], 'in') # Higgs
            externals = [FO, FI, VC, SC]

            #check_momentum_conservation
            #print('momentum sum:', FO.p - FI.p + VC.p + SC.p)

            ###################
            # Diagram-1 
            ###################
            amp1A = get_amp1A(externals, COUP) # Method-A: H + Z -> Z*, Amp = Z + f + fbar )
            amp1B = get_amp1B(externals, COUP) # Method-B: f + fbar -> Z*, Amp = Z + Z* + H 
            #check consistency 
            diff = abs(amp1A - amp1B)/max(1e-12, abs(amp1A))
            if diff > 1e-12:
                print('ERROR: amp1A disagrees with amp1B')
                print("|amp1A - amp1B|=", diff)                
                exit()
            else: amp1 = amp1A

            ###################
            # Diagram-2
            ###################
            amp2A = get_amp2A(externals, COUP) # Method-A: H + Z -> Z*, Amp = Z + f + fbar )
            amp2B = get_amp2B(externals, COUP) # Method-B: f + fbar -> Z*, Amp = Z + Z* + H 
            #check consistency 
            diff = abs(amp2A - amp2B)/max(1e-12, abs(amp2A))
            if diff > 1e-12:
                print('ERROR: amp2A disagrees with amp2B')
                print("amp2A",amp2A)
                print("amp2B",amp2B)
                print("|amp2A - amp2B|=", diff)                
                exit()
            else: amp2 = amp2A

            ###################
            # Diagram-3
            ###################
            amp3A = get_amp3A(externals, COUP) # Method-A: H + Z -> Z*, Amp = Z + f + fbar )
            amp3B = get_amp3B(externals, COUP) # Method-B: f + fbar -> Z*, Amp = Z + Z* + H 
            #check consistency 
            diff = abs(amp3A - amp3B)/max(1e-12, abs(amp3A))
            if diff > 1e-12:
                print('ERROR: amp3A disagrees with amp3B')
                print("amp3A",amp3A)
                print("amp3B",amp3B)
                print("|amp3A - amp3B|=", diff)                
                exit()
            else: amp3 = amp3A

            outlist.append([ lA, lB, lZ, amp1+amp2+amp3, amp1 ])

pretty_output( header, outlist )

