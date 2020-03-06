from django.shortcuts import render
from django.http import JsonResponse
from .models import Post, Calculation

from plotly.offline import plot
from plotly.graph_objects import Scatter3d

import math

def mainpage(request):
    mP = 3
    mR = 2
    mb1 = 0.1
    mb2 = 0.1
    mSp = 30
    mN = 10
    mT = mP + mR + 1
    msc_scb = 2*mb1+2*math.sqrt(mR)*mb2
    mbr_scb = (mb1*(mP-mP*mR/(mT-mP)+1)/math.sqrt(mT))+(mb2*(mP-mP/(mT-mP)+mR)/math.sqrt(mT))+(1+mR)/math.sqrt(mT)
    mlr_scb = mb1/math.sqrt(mT)*(mP+1)+mb2/math.sqrt(mT)*(mP+mR*(1-1/(mT-mR)))+(1+mR/(mT-mR))/math.sqrt(mT)
    msr_scb = mb1*(2-mR/mT)+mb2*math.sqrt(mT)*(1-1/mT)+(mR/math.sqrt(mT)+1)/math.sqrt(mT)
    mrc_scb = (mb1*(mP+2+1/(0.5*mN-1)-mT/mN)+mb2*(mP+2*mR+mR/(0.5*mN)-mT/mN)+(mN-1)*(1/(0.5*mN-1)+mR/(0.5*mN)-mT/pow(mN,2)))/math.sqrt(mT)

    mscb_alg_min = min_alg(msc_scb, mbr_scb, mlr_scb, msr_scb, mrc_scb)
    mplot_scb = graph(msc_scb, mbr_scb, mlr_scb, msr_scb, mrc_scb)
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

        sc_scb = 2*b1+2*math.sqrt(R)*b2
        br_scb = (b1*(P-P*R/(T-P)+1)/math.sqrt(T))+(b2*(P-P/(T-P)+R)/math.sqrt(T))+(1+R)/math.sqrt(T)
        lr_scb = b1/math.sqrt(T)*(P+1)+b2/math.sqrt(T)*(P+R*(1-1/(T-R)))+(1+R/(T-R))/math.sqrt(T)
        sr_scb = b1*(2-R/T)+b2*math.sqrt(T)*(1-1/T)+(R/math.sqrt(T)+1)/math.sqrt(T)
        rc_scb = (b1*(P+2+1/(0.5*N-1)-T/N)+b2*(P+2*R+R/(0.5*N)-T/N)+(N-1)*(1/(0.5*N-1)+R/(0.5*N)-T/pow(N,2)))/math.sqrt(T)

        scb_alg_min = min_alg(sc_scb, br_scb, lr_scb, sr_scb, rc_scb)
        plot_scb = graph(sc_scb, br_scb, lr_scb, sr_scb, rc_scb)

        sc_pcb = max(2*math.sqrt(T)*(b1*(1-1/math.sqrt(T))+b2*math.sqrt(R)*(1-math.sqrt(R/T))),2*R*b2,2*b1)
        br_pcb = max(P*(b1*(1-R/(T-P))+b2*(1-1/(T-P))),b1+1,R*(b2+1))
        lr_pcb = max(P*(b1+b2),R*(b2*(1-1/(T-R))+1/(T-R)),b1+1)
        sr_pcb = max(math.sqrt(T)*(2-2/math.sqrt(T)-R/T)*b1+P*b2,R*b2+R/math.sqrt(T),2*b1+1)
        rc_pcb = max(2*b1+(T/2+T/N-P-2*R/N),2*R*b2+(N/(N-2)+R-0.5*T),b2*(T-T/N-R+2*R/N+(1+0.5*N)/(0.5*N-1))+b1*(T-R-T/N-1+1/(0.5*N-1)))

        pcb_alg_min = min_alg(sc_pcb, br_pcb, lr_pcb, sr_pcb, rc_pcb)
        plot_pcb = graph(sc_pcb, br_pcb, lr_pcb, sr_pcb, rc_pcb)

        response_data = {'P': P, 'R': R, 'b1': b1, 'b2': b2, 'Sp': Sp, 'N': N, 
            'scb_alg_min': scb_alg_min, 'plot_scb': plot_scb, 'sc_scb': sc_scb, 'br_scb': br_scb, 'lr_scb': lr_scb, 'sr_scb': sr_scb, 'rc_scb': rc_scb,
            'pcb_alg_min': pcb_alg_min, 'plot_pcb': plot_pcb,  'sc_pcb': sc_pcb, 'br_pcb': br_pcb, 'lr_pcb': lr_pcb, 'sr_pcb': sr_pcb, 'rc_pcb': rc_pcb,}
        return JsonResponse(response_data)
    else: 
        return render(request, 'mult/index.html', {'P': mP, 'R': mR, 'b1': mb1, 'b2': mb2, 'Sp': mSp, 'N': mN, 'plot_scb': mplot_scb, 
        'scb_alg_min': mscb_alg_min, 'sc_scb': msc_scb, 'br_scb': mbr_scb, 'lr_scb': mlr_scb, 'sr_scb': msr_scb, 'rc_scb': mrc_scb}) 
    return render(request, 'mult/index.html', {})  

def start_page():
    
    return response_data

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


def graph(sc, br, lr, sr, rc):
    x_data = [i for i in range(20)]
    sc_z_data = [i for i in range(20)]
    br_z_data = [i for i in range(20)]
    lr_z_data = [i for i in range(20)]
    sr_z_data = [i for i in range(20)]
    rc_z_data = [i for i in range(20)]
    count = 0
    while count<len(x_data):
        sc_z_data[count] = sc
        count = count + 1
    count = 0
    while count<len(x_data):
        br_z_data[count] = br
        count = count + 1
    count = 0
    while count<len(x_data):
        lr_z_data[count] = lr
        count = count + 1
    count = 0
    while count<len(x_data):
        sr_z_data[count] = sr
        count = count + 1
    count = 0
    while count<len(x_data):
        rc_z_data[count] = rc
        count = count + 1

    pl = plot([Scatter3d(x=x_data, y=x_data, z=sc_z_data,
                    mode='lines', name='SC',
                    opacity=1, marker_color='fuchsia'),
                    Scatter3d(x=x_data, y=x_data, z=br_z_data,
                    mode='lines', name='BR',
                    opacity=1, marker_color='blue'),
                    Scatter3d(x=x_data, y=x_data, z=lr_z_data,
                    mode='lines', name='LR',
                    opacity=1, marker_color='red'),
                    Scatter3d(x=x_data, y=x_data, z=sr_z_data,
                    mode='lines', name='SR',
                    opacity=1, marker_color='green'),
                    Scatter3d(x=x_data, y=x_data, z=rc_z_data,
                    mode='lines', name='RC',
                    opacity=1, marker_color='black')],
               output_type='div')
    return pl