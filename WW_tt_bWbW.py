import numpy as np
from math import sqrt, cos, sin, acos, asin
import pyHELAS 
import vector
import math

pi = math.pi

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

def get_mom_M_2m(M, m):
    return sqrt(M**2/4 - m**2)

def get_mom_M_ma_mb(M, a, b):
    return sqrt(M**4 - 2*M**2*(a**2 + b**2) + (a**2 - b**2)**2)/(2*M)

def get_momenta(ECOM, th, th0, th0b):

    pp = {}
    pv = {}

    Ein = ECOM/2
    pin = sqrt( Ein**2 - mW**2 )
    pWp_in = vector.obj( px=0, py=0, pz=pin, E=Ein )
    pWm_in = vector.obj( px=0, py=0, pz=-pin, E=Ein )

    # M -> 2m
    ptop = get_mom_M_2m(ECOM, mtop) 

    #th = np.random.uniform( 2*pi )

    pt    = vector.obj( px=ptop*sin(th), py=0, pz=ptop*cos(th), E=ECOM/2 )
    ptbar = vector.obj( px=-ptop*sin(th), py=0, pz=-ptop*cos(th), E=ECOM/2 )

    # M -> ma + mb
    p0 = get_mom_M_ma_mb(mtop, mW, mb) 

    #th0 = np.random.uniform( 2*pi )
    #th0b = np.random.uniform( 2*pi )

    pWp0 = vector.obj( px=p0*sin(th0), py=0, pz=p0*cos(th0), mass=mW )
    pb0 = vector.obj( px=-p0*sin(th0), py=0, pz=-p0*cos(th0), mass=mb )

    pWm0 = vector.obj( px=p0*sin(th0b), py=0, pz=p0*cos(th0b), mass=mW )
    pbbar0 = vector.obj( px=-p0*sin(th0b), py=0, pz=-p0*cos(th0b), mass=mb )

    pWp_out = pWp0.boost_p4(pt)
    pb = pb0.boost_p4(pt)

    pWm_out = pWm0.boost_p4(ptbar)
    pbbar = pbbar0.boost_p4(ptbar)

    #pp

    pp['t'] = vec2ar(pt)
    pp['tbar'] = vec2ar(ptbar)

    pp['b'] = vec2ar(pb)
    pp['bbar'] = vec2ar(pbbar)

    pp['Wp_out'] = vec2ar(pWp_out)
    pp['Wm_out'] = vec2ar(pWm_out)

    pp['Wp_in'] = vec2ar(pWp_in)
    pp['Wm_in'] = vec2ar(pWm_in)

    #pv

    pv['t'] = pt
    pv['tbar'] = ptbar

    pv['b'] = pb
    pv['bbar'] = pbbar

    pv['Wp_out'] = pWp_out
    pv['Wm_out'] = pWm_out

    pv['Wp_in'] = pWp_in
    pv['Wm_in'] = pWm_in

    print('momentum conservation')
    print( pp['t'] + pp['tbar'] )
    print( pp['Wp_out'] + pp['b'] - pp['t'])
    print( pp['Wm_out'] + pp['bbar'] - pp['tbar'])

    return pp, pv


ECOM = 500
Twid = pyHELAS.Info['t']['w']
Zwid = pyHELAS.Info['Z']['w']
mtop = pyHELAS.Info['t']['m']
mb = pyHELAS.Info['b']['m']
mW = pyHELAS.Info['W']['m']
mZ = pyHELAS.Info['Z']['m']
gW = pyHELAS.Info['const']['gW']
gEM = pyHELAS.Info['const']['gEM']
cosW = pyHELAS.Info['const']['cosW']
sinW = pyHELAS.Info['const']['sinW']

COUP = pyHELAS.Info['G']
Info = pyHELAS.Info

# G_WF = pyHELAS.Info['G']['WF']
# G_WWA = pyHELAS.Info['G']['WWA']
# G_WWZ = pyHELAS.Info['G']['WWZ']
# G_hZZ = pyHELAS.Info['G']['hZZ']
# G_hWW = pyHELAS.Info['G']['hWW']

#th, th0, th0b = np.random.uniform(0, 2*pi, 3)
th, th0, th0b = 1/3, 1/5, 6/5

pp, pv = get_momenta(ECOM, th, th0, th0b)

lW1 = 1
lW2 = 1

lb1 = 1
lb2 = 1

lWp = 1
lWm = 1

for lW1 in [1, -1, 0]:
    for lW2 in [1, -1, 0]:
        for lb1 in [1, -1]:
            for lb2 in [1, -1]:
                for lWp in [1, -1, 0]:
                    for lWm in [1, -1, 0]:

                        if (lW1, lW2, lb1, lb2, lWp, lWm) != (1, 1, 1, 1, 0, 0): continue

                        # External States
                        Wp_in = pyHELAS.VXXXXX(pp['Wp_in'],lW1,'in') 
                        Wm_in = pyHELAS.VXXXXX(pp['Wm_in'],lW2,'in')

                        B = pyHELAS.OXXXXX(pp['b'], lb1, 'fer') 
                        Bbar = pyHELAS.IXXXXX(pp['bbar'], lb2, 'fbar') 

                        Wp_out = pyHELAS.VXXXXX(pp['Wp_out'], lWp, 'out') 
                        Wm_out = pyHELAS.VXXXXX(pp['Wm_out'], lWm, 'out') 

                        # Calculation Common
                        Top = pyHELAS.FVOXXX(B, Wp_out, COUP['WF'], mtop, Twid)
                        Tbar = pyHELAS.FVIXXX(Bbar, Wm_out, COUP['WF'], mtop, Twid)

                        # Calculation 1
                        CURR_A = pyHELAS.JVVXXX(Wm_in,Wp_in,COUP['WWA'], 0, 0)
                        CURR_Z = pyHELAS.JVVXXX(Wm_in,Wp_in,COUP['WWZ'], mZ, Zwid)

                        amp_A = pyHELAS.IOVXXX(Tbar, Top, CURR_A, COUP['AUU'])
                        amp_Z = pyHELAS.IOVXXX(Tbar, Top, CURR_Z, COUP['ZUU'])

                        amp0 = amp_A + amp_Z

                        # Calculation 2
                        J3 = pyHELAS.J3XXXX(Tbar,Top,COUP['AUU'],COUP['ZUU'],mZ,Zwid,Info)
                        J3_2 = pyHELAS.J3XXXX2(Tbar,Top,COUP['AUU'],COUP['ZUU'],mZ,Zwid,Info)

                        amp1 = pyHELAS.VVVXXX(Wm_in,Wp_in,J3,Info['const']['gW'])
                        amp2 = pyHELAS.VVVXXX(Wm_in,Wp_in,J3_2,Info['const']['gW'])

                        J3_3 = pyHELAS.J3XXXX3(Tbar,Top,COUP['AUU'],COUP['ZUU'],mZ,Zwid,Info)
                        amp3 = pyHELAS.VVVXXX(Wm_in,Wp_in,J3_3,Info['const']['gW'])

                        print( 'amp_0 =', amp0 )
                        print( 'amp1 =', amp1 )
                        print( 'amp2 =', amp2)
                        print( 'amp3 =', amp3)

                        #assert np.allclose(J3.p + Wm_in.p + Wp_in.p, 0, atol=1e-12)

                        # 2) A 単独・Z 単独も一致するはず
                        # JA = pyHELAS.JIOXXX(Tbar, Top, COUP['AUU'], 0, 0)
                        # JZ = pyHELAS.JIOXXX(Tbar, Top, COUP['ZUU'], mZ, Zwid)

                        # JA_in = pyHELAS.JVVXXX(Wm_in, Wp_in, COUP['WWA'], 0, 0)
                        # JZ_in = pyHELAS.JVVXXX(Wm_in, Wp_in, COUP['WWZ'], mZ, Zwid)
                        # ampA_1 = pyHELAS.IOVXXX(Tbar, Top, JA_in, COUP['AUU'])
                        # ampA_2 = pyHELAS.VVVXXX(Wm_in, Wp_in, JA, COUP['WWA'])

                        # ampZ_1 = pyHELAS.IOVXXX(Tbar, Top, JZ_in, COUP['ZUU'])
                        # ampZ_2 = pyHELAS.VVVXXX(Wm_in, Wp_in, JZ, COUP['WWZ'])

                        # print("A-only rel diff =", abs(ampA_1-ampA_2)/max(1e-12,abs(ampA_1)))
                        # print("Z-only rel diff =", abs(ampZ_1-ampZ_2)/max(1e-12,abs(ampZ_1)))


                        # A-only / Z-only（すでに一致しているはず）

                        JA_in = pyHELAS.JVVXXX(Wm_in, Wp_in, COUP['WWA'], 0, 0)
                        JZ_in = pyHELAS.JVVXXX(Wm_in, Wp_in, COUP['WWZ'], mZ, Zwid)                     
                        ampA_1 = pyHELAS.IOVXXX(Tbar, Top, JA_in, COUP['AUU'])
                        ampZ_1 = pyHELAS.IOVXXX(Tbar, Top, JZ_in, COUP['ZUU'])
                        amp_1 = ampA_1 + ampZ_1

                        JA = pyHELAS.JIOXXX(Tbar, Top, COUP['AUU'], 0, 0)
                        JZ = pyHELAS.JIOXXX(Tbar, Top, COUP['ZUU'], mZ, Zwid)
                        ampA_2 = pyHELAS.VVVXXX(Wm_in, Wp_in, JA, COUP['WWA'])
                        ampZ_2 = pyHELAS.VVVXXX(Wm_in, Wp_in, JZ, COUP['WWZ'])
                        amp_2 = ampA_2 + ampZ_2

                        # J3 全体
                        J3 = pyHELAS.J3XXXX(Tbar, Top, COUP['AUU'], COUP['ZUU'], mZ, Zwid, Info)
                        amp_J3_VVV  = pyHELAS.VVVXXX(Wm_in, Wp_in, J3, Info['const']['gW'])

                        print("A-only rel diff =", abs(ampA_1-ampA_2)/max(1e-12,abs(ampA_1)))
                        print("Z-only rel diff =", abs(ampZ_1-ampZ_2)/max(1e-12,abs(ampZ_1)))
                        print('amp_J3_VVV', amp_J3_VVV)
                        print('ampA_1+ampZ_1', amp_1)                        
                        print("J3  vs sum  diff =", abs(amp_J3_VVV-amp_1)/max(1e-12,abs(amp_1)))

                        # print('J3.p + Wp_in.p + Wm_in.p =', J3.p + Wp_in.p + Wm_in.p)
                        # print('J3.p - Top.p + Tbar.p =', J3.p - Top.p + Tbar.p)

                        # print('Wp_in.p', Wp_in.p)
                        # print('Wm_in.p', Wm_in.p)
                        # print('B.p', B.p)
                        # print('Bbar.p', Bbar.p)
                        # print('Wp_out.p', Wp_out.p)
                        # print('Wm_out.p', Wm_out.p)

                        # print('J3.p', J3.p)
                        # print('Top.p', Top.p)
                        # print('Tbar.p', Tbar.p)

                        # print('amp_JVV_IOV =', amp_JVV_IOV)
                        # print('amp_J3_VVV =', amp_J3_VVV)

                        #print(lW1, lW2, lb1, lb2, lWp, lWm, amp0, amp1, amp2, amp3)


exit()

