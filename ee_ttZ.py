import numpy as np
import vector
from math import sqrt, cos, sin, acos, asin, pi
import pyHELAS 
from decay_vec import Decay_3B

MeV = 10**-3
mH = 125.
mW = 80.369
mZ = 91.188
mt = 173
mtau = 1.777
Twidth = 1.4915
Zwidth = 2.4414
Wwidth = 2.085
Hwidth = 4*MeV

Gf = 1.16639e-5
#aMZ = 1/128
aMZ = 1/132.507
thw = 1/2 *np.arcsin(2*np.sqrt(np.sqrt(2)/2 *pi *aMZ /(Gf *mZ**2)))
sinW = np.sin(thw)
cosW = np.cos(thw)

gev2tobarn = 0.3894e-3

gA = sqrt(4*pi*aMZ)
gW = gA/sinW
gZ = gA/(sinW*cosW)

vev = 2*mZ/gZ

lam = vev**2 / mH**2

COUP = {}
COUP['HZZ'] = gZ * mZ
COUP['HWW'] = gW * mW
COUP['HHH'] = -3 * lam * vev 
COUP['HHHH'] = -3 * lam 
COUP['WWHH'] = gW**2
COUP['ZZHH'] = gZ**2

COUP['HTT'] = -4*gZ*mt/mZ * np.array([1,1])
COUP['HTAUTAU'] = -4*gZ*mtau/mZ * np.array([1,1])


Q =   -1; COUP['AEE'] = -gA*Q * np.array([1,1])
Q =  2/3; COUP['AUU'] = -gA*Q * np.array([1,1])
Q = -1/3; COUP['ADD'] = -gA*Q * np.array([1,1])

Q =    0; COUP['ZNN'] = -gZ * np.array([ 1/2 -Q*sinW**2, -Q*sinW**2])
Q =   -1; COUP['ZEE'] = -gZ * np.array([-1/2 -Q*sinW**2, -Q*sinW**2])
Q =  2/3; COUP['ZUU'] = -gZ * np.array([ 1/2 -Q*sinW**2, -Q*sinW**2])
Q = -1/3; COUP['ZDD'] = -gZ * np.array([-1/2 -Q*sinW**2, -Q*sinW**2])

COUP['WFF'] = np.array([-gW/sqrt(2), 0])



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

def get_momenta(rs, mt, mZ):

    pe  = vector.obj(px=0, py=0, pz = -rs/2, mass = 0 )
    peb = vector.obj(px=0, py=0, pz =  rs/2, mass = 0 )

    p0 = pe + peb
    
    pt, ptb, pz = Decay_3B(p0, mt, mt, mZ)

    pp = {}

    pp['E'] = vec2ar(pe)
    pp['EB'] = vec2ar(peb)
    pp['T'] = vec2ar(pt)
    pp['TB'] = vec2ar(ptb)
    pp['Z'] = vec2ar(pz)

    return pp



###################
# Diagram-1 
###################
# e+ e- > Z -> h Z, h -> t tb
def get_amp1(externals, COUP ):
    EE, EB, TT, TB, VC = externals 
    Zs  = pyHELAS.JIOXXX(EE, EB, COUP['ZEE'], mZ, Zwidth)
    HC  = pyHELAS.HVVXXX(VC, Zs, COUP['HZZ'], mH, Hwidth)
    amp = pyHELAS.IOSXXX(TB,TT,HC,COUP['HTT'])
    return amp

###################
# Diagram-2
###################
# e+ e- > A -> t tb, t -> t Z
def get_amp2(externals, COUP ):
    EE, EB, TT, TB, VC = externals 
    As  = pyHELAS.JIOXXX(EE,EB, COUP['AEE'], 0, 0)
    FO_off = pyHELAS.FVOXXX(TT,VC,COUP['ZUU'],mt,Twidth)
    amp = pyHELAS.IOVXXX(TB,FO_off,As,COUP['AUU'])
    return amp

###################
# Diagram-3
###################
# e+ e- > Z -> t tb, t -> t Z
def get_amp3(externals, COUP ):
    EE, EB, TT, TB, VC = externals 
    Zs  = pyHELAS.JIOXXX(EE,EB, COUP['ZEE'], mZ, Zwidth)
    FO_off = pyHELAS.FVOXXX(TT,VC,COUP['ZUU'],mt,Twidth)
    amp = pyHELAS.IOVXXX(TB,FO_off,Zs,COUP['ZUU'])
    return amp

###################
# Diagram-4
###################
# e+ e- > A -> t tb, tb -> tb Z
def get_amp4(externals, COUP ):
    EE, EB, TT, TB, VC = externals 
    As  = pyHELAS.JIOXXX(EE,EB, COUP['AEE'], 0, 0)
    FI_off = pyHELAS.FVIXXX(TB,VC,COUP['ZUU'],mt,Twidth)
    amp = pyHELAS.IOVXXX(FI_off,TT,As,COUP['AUU'])
    return amp

###################
# Diagram-5
###################
# e+ e- > Z -> t tb, tb -> tb Z
def get_amp5(externals, COUP ):
    EE, EB, TT, TB, VC = externals 
    Zs  = pyHELAS.JIOXXX(EE,EB, COUP['ZEE'], mZ, Zwidth)
    FI_off = pyHELAS.FVIXXX(TB,VC,COUP['ZUU'],mt,Twidth)
    amp = pyHELAS.IOVXXX(FI_off,TT,Zs,COUP['ZUU'])
    return amp


###################
# Diagram-6
###################
# e- e+ > A- Z+, A- -> t tb
def get_amp6(externals, COUP ):
    EE, EB, TT, TB, VC = externals 
    As  = pyHELAS.JIOXXX(TB,TT, COUP['AUU'], 0, 0)
    FI_off = pyHELAS.FVIXXX(EE,As,COUP['AEE'],0,0)
    amp = pyHELAS.IOVXXX(FI_off,EB,VC,COUP['ZEE'])
    return amp

###################
# Diagram-7
###################
# e- e+ > Z- Z+, Z- -> t tb
def get_amp7(externals, COUP ):
    EE, EB, TT, TB, VC = externals 
    Zs  = pyHELAS.JIOXXX(TB,TT, COUP['ZUU'], mZ, Zwidth)
    FI_off = pyHELAS.FVIXXX(EE,Zs,COUP['ZEE'],0,0)
    amp = pyHELAS.IOVXXX(FI_off,EB,VC,COUP['ZEE'])
    return amp

###################
# Diagram-8
###################
# e- e+ > Z- A+, A+ -> t tb
def get_amp8(externals, COUP ):
    EE, EB, TT, TB, VC = externals 
    As  = pyHELAS.JIOXXX(TB,TT, COUP['AUU'], 0, 0)
    FO_off = pyHELAS.FVOXXX(EB,As,COUP['AEE'],0,0)
    amp = pyHELAS.IOVXXX(EE,FO_off,VC,COUP['ZEE'])
    return amp

###################
# Diagram-8
###################
# e- e+ > Z- A+, A+ -> t tb
def get_amp9(externals, COUP ):
    EE, EB, TT, TB, VC = externals 
    Zs  = pyHELAS.JIOXXX(TB,TT, COUP['ZUU'], mZ, Zwidth)
    FO_off = pyHELAS.FVOXXX(EB,Zs,COUP['ZEE'],0,0)
    amp = pyHELAS.IOVXXX(EE,FO_off,VC,COUP['ZEE'])
    return amp

rs = 1000

pp = get_momenta(rs, mt, mZ)

print('pE:', pp['E'])
print('pEB:', pp['EB'])
print('pT:', pp['T'])
print('pTB:', pp['TB'])
print('pZ:', pp['Z'])
print('pT + pTB + pZ:', pp['T'] + pp['TB'] + pp['Z'])

header = ['lA', 'lB', 'lZ', 'AmpTot(Graph1 + Graph2 + Graph3)', 'Amp1(Graph1)']
outlist = []

lE, lEB = 1, -1
for lT in [1, -1]:
    for lTB in [1, -1]:
        for lZ in [1, 0, -1]:

            # if lA != -1: continue
            # if lB != 1: continue
            # if lZ != 0: continue

            # Defining external momenta  
            EE = pyHELAS.IXXXXX(pp['E'], lE, 'fermion') # t
            EB = pyHELAS.OXXXXX(pp['EB'], lEB, 'fbar') # tbar

            TT = pyHELAS.OXXXXX(pp['T'], lT, 'fermion') # t
            TB = pyHELAS.IXXXXX(pp['TB'], lTB, 'fbar') # tbar
            VC = pyHELAS.VXXXXX(pp['Z'], lZ, 'out') # Z
            externals = [EE, EB, TT, TB, VC]

            #check_momentum_conservation
            #print('momentum sum:', FO.p - FI.p + VC.p + SC.p)

            ###################
            # Diagram-1 
            ###################
            amp1 = get_amp1(externals, COUP) 
            amp2 = get_amp2(externals, COUP) 
            amp3 = get_amp3(externals, COUP) 
            amp4 = get_amp4(externals, COUP) 
            amp5 = get_amp5(externals, COUP) 
            amp6 = get_amp6(externals, COUP) 
            amp7 = get_amp7(externals, COUP) 
            amp8 = get_amp8(externals, COUP) 
            amp9 = get_amp9(externals, COUP) 

            print(lE, lEB, lT, lTB, lZ)
            print('amp1:', amp1)
            print('amp2:', amp2)
            print('amp3:', amp3)
            print('amp4:', amp4)
            print('amp5:', amp5)
            print('amp6:', amp6)
            print('amp7:', amp7)
            print('amp8:', amp8)
            print('amp9:', amp9)

            #outlist.append([ lA, lB, lZ, amp1+amp2+amp3, amp1 ])

#pretty_output( header, outlist )

