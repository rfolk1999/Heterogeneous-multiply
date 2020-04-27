from django.shortcuts import render
from django.http import JsonResponse
from .models import Post, Calculation

from plotly.offline import plot
from plotly.graph_objects import Scatter3d, Surface

import math
import pandas as pd
import numpy as np

def mainpage(request):
    x_data = [i for i in range(20)]


    mP = 7
    mR = 2
    mb1 = 20.0
    mb2 = 2.0
    mSp = 30
    mN = 10
    mT = mP + mR + 1


    msc_scb = scb_sc(mb1, mP, mR, mT, mb2, mN, mSp)
    mbr_scb = scb_br(mb1, mP, mR, mT, mb2, mN, mSp)
    mlr_scb = scb_lr(mb1, mP, mR, mT, mb2, mN, mSp)
    msr_scb = scb_sr(mb1, mP, mR, mT, mb2, mN, mSp)
    mrc_scb = scb_rc(mb1, mP, mR, mT, mb2, mN, mSp)
    scz = np.arange(20)

    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    xGrid, yGrid = np.meshgrid(y, x)
    sc_z = scb_sc(xGrid, mP, mR, mT, yGrid, mN, mSp)
    br_z = scb_br(xGrid, mP, mR, mT, yGrid, mN, mSp)
    lr_z = scb_lr(xGrid, mP, mR, mT, yGrid, mN, mSp)
    sr_z = scb_sr(xGrid, mP, mR, mT, yGrid, mN, mSp)
    rc_z = scb_rc(xGrid, mP, mR, mT, yGrid, mN, mSp)
    
    gg = br_z[2,20]
    mscb_alg_min = min_alg(msc_scb, mbr_scb, mlr_scb, msr_scb, mrc_scb)
    mplot_scb = graph(x, y, sc_z, br_z, lr_z, sr_z, rc_z)
    
    #sc_z = scb_sc(xGrid, mP, mR, mT, yGrid, mN, mSp)
    #gg = sc_z
    #br_z = scb_br(xGrid, mP, mR, mT, yGrid, mN, mSp)
    #lr_z = scb_lr(xGrid, mP, mR, mT, yGrid, mN, mSp)
    #sr_z = scb_sr(xGrid, mP, mR, mT, yGrid, mN, mSp)
    #rc_z = scb_rc(xGrid, mP, mR, mT, yGrid, mN, mSp)
    #mscb_alg_min = min_alg(msc_scb, mbr_scb, mlr_scb, msr_scb, mrc_scb)
    #mplot_scb = graph(x, y, sc_z, br_z, lr_z, sr_z, rc_z)
    posts = Calculation.objects.all()
    response_data = {}
    if request.method=='POST': 
        P = int(request.POST.get('P'))
        R = int(request.POST.get('R'))
        b1 = float(request.POST.get('b1'))
        b2 = float(request.POST.get('b2'))
        Sp = int(request.POST.get('Sp'))
        N = int(request.POST.get('N'))
        T = P + R + 1

        sc_scb = scb_sc(b1, P, R, T, b2, N, Sp)
        br_scb = scb_br(b1, P, R, T, b2, N, Sp)
        lr_scb = scb_lr(b1, P, R, T, b2, N, Sp)
        sr_scb = scb_sr(b1, P, R, T, b2, N, Sp)
        rc_scb = scb_rc(b1, P, R, T, b2, N, Sp)

        sc_z = scb_sc(xGrid, P, R, T, yGrid, N, Sp)
        
        br_z = scb_br(xGrid, P, R, T, yGrid, N, Sp)
        lr_z = scb_lr(xGrid, P, R, T, yGrid, N, Sp)
        sr_z = scb_sr(xGrid, P, R, T, yGrid, N, Sp)
        rc_z = scb_rc(xGrid, P, R, T, yGrid, N, Sp)

        

        scb_alg_min = min_alg(sc_scb, br_scb, lr_scb, sr_scb, rc_scb)
        plot_scb = graph(x, y, sc_z, br_z, lr_z, sr_z, rc_z)

       
        #sc_pcb = pcb_sc(b1, P, R, T, b2, N)
        
        sc_pcb = pcb_sc(b1, P, R, T, b2, N, Sp)
        br_pcb = pcb_br(b1, P, R, T, b2, N, Sp)
        lr_pcb = pcb_lr(b1, P, R, T, b2, N, Sp)
        sr_pcb = pcb_sr(b1, P, R, T, b2, N, Sp)
        rc_pcb = pcb_rc(b1, P, R, T, b2, N, Sp)

        for i in range(21):
            for j in range(21):
                sc_z[i][j] = pcb_sc(xGrid[i][j], mP, mR, mT, yGrid[i][j], mN, mSp)
        
        for i in range(21):
            for j in range(21):
                br_z[i][j] = pcb_br(xGrid[i][j], mP, mR, mT, yGrid[i][j], mN, mSp)

        for i in range(21):
            for j in range(21):
                lr_z[i][j] = pcb_lr(xGrid[i][j], mP, mR, mT, yGrid[i][j], mN, mSp)

        for i in range(21):
            for j in range(21):
                sr_z[i][j] = pcb_sr(xGrid[i][j], mP, mR, mT, yGrid[i][j], mN, mSp)

        for i in range(21):
            for j in range(21):
                rc_z[i][j] = pcb_rc(xGrid[i][j], mP, mR, mT, yGrid[i][j], mN, mSp)


        pcb_alg_min = min_alg(sc_pcb, br_pcb, lr_pcb, sr_pcb, rc_pcb)
        plot_pcb = graph(x, y, sc_z, br_z, lr_z, sr_z, rc_z)

        sc_sco = sco_sc(b1, P, R, T, b2, N, Sp)
        br_sco = sco_br(b1, P, R, T, b2, N, Sp)
        lr_sco = sco_lr(b1, P, R, T, b2, N, Sp)
        sr_sco = sco_sr(b1, P, R, T, b2, N, Sp)
        rc_sco = sco_rc(b1, P, R, T, b2, N, Sp)


        for i in range(21):
            for j in range(21):
                sc_z[i][j] = sco_sc(xGrid[i][j], P, R, T, yGrid[i][j], N, Sp)
        #sc_z = sco_sc(xGrid, P, R, T, yGrid, N)
        br_z = sco_br(xGrid, P, R, T, yGrid, N, Sp)
        lr_z = sco_lr(xGrid, P, R, T, yGrid, N, Sp)
        sr_z = sco_sr(xGrid, P, R, T, yGrid, N, Sp)
        rc_z = sco_rc(xGrid, P, R, T, yGrid, N, Sp)

        sco_alg_min = min_alg(sc_sco, br_sco, lr_sco, sr_sco, rc_sco)
        plot_sco = graph(x, y, sc_z, br_z, lr_z, sr_z, rc_z)

        sc_pco = pco_sc(b1, P, R, T, b2, N, Sp)
        br_pco = pco_br(b1, P, R, T, b2, N, Sp)
        lr_pco = pco_lr(b1, P, R, T, b2, N, Sp)
        sr_pco = pco_sr(b1, P, R, T, b2, N, Sp)
        rc_pco = pco_rc(b1, P, R, T, b2, N, Sp)

        #sc_z = pco_sc(xGrid, P, R, T, yGrid, N)
        #br_z = pco_br(xGrid, P, R, T, yGrid, N)
        #lr_z = pco_lr(xGrid, P, R, T, yGrid, N)
        #sr_z = pco_sr(xGrid, P, R, T, yGrid, N)
        #rc_z = pco_rc(xGrid, P, R, T, yGrid, N)

        for i in range(21):
            for j in range(21):
                sc_z[i][j] = pco_sc(xGrid[i][j], mP, mR, mT, yGrid[i][j], mN, mSp)
        
        for i in range(21):
            for j in range(21):
                br_z[i][j] = pco_br(xGrid[i][j], mP, mR, mT, yGrid[i][j], mN, mSp)

        for i in range(21):
            for j in range(21):
                lr_z[i][j] = pco_lr(xGrid[i][j], mP, mR, mT, yGrid[i][j], mN, mSp)

        for i in range(21):
            for j in range(21):
                sr_z[i][j] = pco_sr(xGrid[i][j], mP, mR, mT, yGrid[i][j], mN, mSp)

        for i in range(21):
            for j in range(21):
                rc_z[i][j] = pco_rc(xGrid[i][j], mP, mR, mT, yGrid[i][j], mN, mSp)
        

        pco_alg_min = min_alg(sc_pco, br_pco, lr_pco, sr_pco, rc_pco)
        plot_pco = graph(x, y, sc_z, br_z, lr_z, sr_z, rc_z)

        sc_pio = pio_sc(b1, P, R, T, b2, N, Sp)
        br_pio = pio_br(b1, P, R, T, b2, N, Sp)
        lr_pio = pio_lr(b1, P, R, T, b2, N, Sp)
        sr_pio = pio_sr(b1, P, R, T, b2, N, Sp)
        rc_pio = pio_rc(b1, P, R, T, b2, N, Sp)

        #sc_z = pio_sc(xGrid, P, R, T, yGrid, N, Sp)
        #br_z = pio_br(xGrid, P, R, T, yGrid, N, Sp)
        #lr_z = pio_lr(xGrid, P, R, T, yGrid, N, Sp)
        #sr_z = pio_sr(xGrid, P, R, T, yGrid, N, Sp)
        #rc_z = pio_rc(xGrid, P, R, T, yGrid, N, Sp)

        for i in range(21):
            for j in range(21):
                sc_z[i][j] = pio_sc(xGrid[i][j], mP, mR, mT, yGrid[i][j], mN, mSp)
        
        for i in range(21):
            for j in range(21):
                br_z[i][j] = pio_br(xGrid[i][j], mP, mR, mT, yGrid[i][j], mN, mSp)

        for i in range(21):
            for j in range(21):
                lr_z[i][j] = pio_lr(xGrid[i][j], mP, mR, mT, yGrid[i][j], mN, mSp)

        for i in range(21):
            for j in range(21):
                sr_z[i][j] = pio_sr(xGrid[i][j], mP, mR, mT, yGrid[i][j], mN, mSp)

        for i in range(21):
            for j in range(21):
                rc_z[i][j] = pio_rc(xGrid[i][j], mP, mR, mT, yGrid[i][j], mN, mSp)

        pio_alg_min = min_alg(sc_pio, br_pio, lr_pio, sr_pio, rc_pio)
        plot_pio = graph(x, y, sc_z, br_z, lr_z, sr_z, rc_z)

        response_data = {'P': P, 'R': R, 'b1': b1, 'b2': b2, 'Sp': Sp, 'N': N, 
            'scb_alg_min': scb_alg_min, 'plot_scb': plot_scb, 'sc_scb': sc_scb, 'br_scb': br_scb, 'lr_scb': lr_scb, 'sr_scb': sr_scb, 'rc_scb': rc_scb,
            'pcb_alg_min': pcb_alg_min, 'plot_pcb': plot_pcb,  'sc_pcb': sc_pcb, 'br_pcb': br_pcb, 'lr_pcb': lr_pcb, 'sr_pcb': sr_pcb, 'rc_pcb': rc_pcb,
            'sco_alg_min': sco_alg_min, 'plot_sco': plot_sco, 'sc_sco': sc_sco, 'br_sco': br_sco, 'lr_sco': lr_sco, 'sr_sco': sr_sco, 'rc_sco': rc_sco,
            'pco_alg_min': pco_alg_min, 'plot_pco': plot_pco,  'sc_pco': sc_pco, 'br_pco': br_pco, 'lr_pco': lr_pco, 'sr_pco': sr_pco, 'rc_pco': rc_pco,
            'pio_alg_min': pio_alg_min, 'plot_pio': plot_pio,  'sc_pio': sc_pio, 'br_pio': br_pio, 'lr_pio': lr_pio, 'sr_pio': sr_pio, 'rc_pio': rc_pio,
            }
        return JsonResponse(response_data)
    else: 
        return render(request, 'mult/index.html', {'xGrid': gg, 'P': mP, 'R': mR, 'b1': mb1, 'b2': mb2, 'Sp': mSp, 'N': mN, 'plot_scb': mplot_scb, 
        'scb_alg_min': mscb_alg_min, 'sc_scb': msc_scb, 'br_scb': mbr_scb, 'lr_scb': mlr_scb, 'sr_scb': msr_scb, 'rc_scb': mrc_scb}) 
    return render(request, 'mult/index.html', {})  

def start_page():
    
    return response_data

def scb_sc(b1, P, R, T, b2, N, Sp):
    return 2*b1+2*math.sqrt(R)*b2

def scb_br(b1, P, R, T, b2, N, Sp):
    return (b1*(P-P*R/(T-P)+1)/math.sqrt(T))+(b2*(P-P/(T-P)+R)/math.sqrt(T))+(1+R)/math.sqrt(T)

def scb_lr(b1, P, R, T, b2, N, Sp):
    return b1/math.sqrt(T)*(P+1)+b2/math.sqrt(T)*(P+R*(1-1/(T-R)))+(1+R/(T-R))/math.sqrt(T)

def scb_sr(b1, P, R, T, b2, N, Sp):
    return b1*(2-R/T)+b2*math.sqrt(T)*(1-1/T)+(R/math.sqrt(T)+1)/math.sqrt(T)
        
def scb_rc(b1, P, R, T, b2, N, Sp):
    return (b1*(P+2+1/(0.5*N-1)-T/N)+b2*(P+2*R+R/(0.5*N)-T/N)+(N-1)*(1/(0.5*N-1)+R/(0.5*N)-T/pow(N,2)))/math.sqrt(T)


def pcb_sc(b1, P, R, T, b2, N, Sp):
    return max(2*math.sqrt(T)*(b1*(1-1/math.sqrt(T))+b2*math.sqrt(R)*(1-math.sqrt(R/T))),2*R*b2,2*b1)

def pcb_br(b1, P, R, T, b2, N, Sp):
    return max(P*(b1*(1-R/(T-P))+b2*(1-1/(T-P))),b1+1,R*(b2+1))

def pcb_lr(b1, P, R, T, b2, N, Sp):
    return max(P*(b1+b2),R*(b2*(1-1/(T-R))+1/(T-R)),b1+1)

def pcb_sr(b1, P, R, T, b2, N, Sp):
    return max(math.sqrt(T)*(2-2/math.sqrt(T)-R/T)*b1+P*b2,R*b2+R/math.sqrt(T),2*b1+1)
        
def pcb_rc(b1, P, R, T, b2, N, Sp):
    return max(2*b1+(T/2+T/N-P-2*R/N),2*R*b2+(N/(N-2)+R-0.5*T),b2*(T-T/N-R+2*R/N+(1+0.5*N)/(0.5*N-1))+b1*(T-R-T/N-1+1/(0.5*N-1)))

def sco_sc(b1, P, R, T, b2, N, Sp):
    return max(2*math.sqrt(T)*b1+2*math.sqrt(R*T)*b2+2*N*math.sqrt(T)*(math.sqrt(R)+1-(R-math.sqrt(R)-1)/math.sqrt(T))/Sp,N*T*(1-(1-R)/T)/Sp,2*math.sqrt(T)*b1+2*math.sqrt(R*T)*b2+N*P/Sp)

def sco_br(b1, P, R, T, b2, N, Sp):
    return b1*(P-P*R/(T-P)+1)+b2*(P-P/(T-P)+R)+1+R+P*N/Sp

def sco_lr(b1, P, R, T, b2, N, Sp):
    return b1*(P+1)+b2*(P+R*(1-1/(T-R)))+1+R/(T-R)+P*N/Sp

def sco_sr(b1, P, R, T, b2, N, Sp):
    return math.sqrt(T)*b1*(2-R/T)+b2*(T-1)+1+R/math.sqrt(T)+P*N/Sp
        
def sco_rc(b1, P, R, T, b2, N, Sp):
    return b1*(P+2+1/(0.5*N-1)-T/N)+b2*(P+2*R+R/(0.5*N)-T/N)+(N-1)*(1/(0.5*N-1)+R/(0.5*N)-T/pow(N,2))+P*N/Sp


def pco_sc(b1, P, R, T, b2, N, Sp):
    return max(max(2*math.sqrt(T)*(b1*(1-1/math.sqrt(T))+b2*math.sqrt(R)*(1-math.sqrt(R/T))),2*R*b2,2*b1)+(2*N*math.sqrt(T)*(math.sqrt(R)+1-(R-math.sqrt(R)-1)/math.sqrt(T)))/Sp,N*T*(1-(R-1)/T)/Sp,max(2*math.sqrt(T)*(b1*(1-1/math.sqrt(T))+b2*math.sqrt(R)*(1-math.sqrt(R/T))),2*R*b2,2*b1)+N*P/Sp)

def pco_br(b1, P, R, T, b2, N, Sp):
    return max(P*b1*(1-R/(T-P))+P*b2*(1-1/(T-P)),b1+1,R*(b2+1))+P*N/Sp

def pco_lr(b1, P, R, T, b2, N, Sp):
    return max(P*(b1+b2),R*(b2*(1-1/(T-R))+1/(T-R)),b1+1)+P*N/Sp

def pco_sr(b1, P, R, T, b2, N, Sp):
    return max(b2*P+b1*(2*math.sqrt(T)-2-R/math.sqrt(T)),R*(b2+1/math.sqrt(T)),2*b1+1)+P*N/Sp
        
def pco_rc(b1, P, R, T, b2, N, Sp):
    return max(2*b1+(T/2+T/N-P-2*R/N),2*R*b2+(N/(N-2)+R-0.5*T),b2*(T-T/N-R+2*R/N+(1+0.5*N)/(0.5*N-1))+b1*(T-R-T/N-1+1/(0.5*N-1)))+P*N/Sp


def pio_sc(b1, P, R, T, b2, N, Sp):
    return 2*math.sqrt(T)*(b1+b2*math.sqrt(R))+(N-1)*max(2*math.sqrt(T)*(b1+b2*math.sqrt(R)), P*N/Sp)+P*N/Sp

def pio_br(b1, P, R, T, b2, N, Sp):
    return b1*(1+P*(1-R/(T-P)))+b2*(R+P*(1-1/(T-P)))+R+1+(N-1)*max(b1*(1+P*(1-R/(T-P)))+b2*(R+P*(1-1/(T-P)))+R+1,P*N/Sp)+P*N/Sp

def pio_lr(b1, P, R, T, b2, N, Sp):
    return b1*(P+1)+b2*(P+R-R/(T-R))+(1+R/(T-R))+(N-1)*max(b1*(P+1)+b2*(P+R-R/(T-R))+(1+R/(T-R)),P*N/Sp)+P*N/Sp

def pio_sr(b1, P, R, T, b2, N, Sp):
    return b1*math.sqrt(T)*(2-R/T)+b2*(T-1)+R/math.sqrt(T)+1+(N-1)*max(b1*math.sqrt(T)*(2-R/T)+b2*(T-1)+R/math.sqrt(T)+1,P*N/Sp)+P*N/Sp
        
def pio_rc(b1, P, R, T, b2, N, Sp):
    return b1*(1-R+T+1/(0.5*N-1)+T/N)+b2*(R+T+N/(N-2)+(1-N)/(0.5*N-1)+(2*R-T)/N)+(1+2*R-T+(T-2*R)/N+N/(N-2))+(N-1)*max((b1*(1-R+T+1/(0.5*N-1)+T/N)+b2*(R+T+N/(N-2)+(1-N)/(0.5*N-1)+(2*R-T)/N)+(1+2*R-T+(T-2*R)/N+N/(N-2))),P*N/Sp)+P*N/Sp

def min_alg(sc, br, lr, sr, rc):
    min_val = min(sc, br, lr, sr, rc)
    if (min_val == sc):
        alg_min = 'SC'
    elif(min_val == br):
        alg_min = 'BR'
    elif(min_val == lr):
        alg_min = 'LR'
    elif(min_val == sr):
        alg_min = 'SR'
    else:
        alg_min = 'RC'
    return alg_min


def graph(x, y, sc_z, br_z, lr_z, sr_z, rc_z):
   
    #pl = plot([Scatter3d(x=x_data, y=x_data, z=sc_z_data,
    #                mode='lines', name='SC',
    #                opacity=1, marker_color='fuchsia'),
    #                Scatter3d(x=x_data, y=x_data, z=br_z_data,
    #                mode='lines', name='BR',
    #                opacity=1, marker_color='blue'),
    #                Scatter3d(x=x_data, y=x_data, z=lr_z_data,
    #                mode='lines', name='LR',
    #                opacity=1, marker_color='red'),
    #                Scatter3d(x=x_data, y=x_data, z=sr_z_data,
    #                mode='lines', name='SR',
    #                opacity=1, marker_color='green'),
    #                Scatter3d(x=x_data, y=x_data, z=rc_z_data,
    #                mode='lines', name='RC',
    #                opacity=1, marker_color='black')
    # ],
    #           output_type='div')

    pl = plot([Surface(x=x, y=y, z=sc_z, name='SC', colorscale="solar", showscale=False, opacity=1),
    Surface(x=x, y=y, z=br_z, name='BR', showscale=False, colorscale="Blues", opacity=1),
    Surface(x=x, y=y, z=lr_z, name='LR', showscale=False, colorscale="Reds", opacity=1),
    Surface(x=x, y=y, z=sr_z, name='SR', showscale=False, colorscale="Greens", opacity=1),
    Surface(x=x, y=y, z=rc_z, name='RC', showscale=False, colorscale="fall", opacity=1),
    ],
               output_type='div')
    
    return pl