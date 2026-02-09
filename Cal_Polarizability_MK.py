import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
from scipy.interpolate import make_interp_spline, BSpline 
from scipy.constants import epsilon_0, mu_0, c
import os

def path():
    return r"C:\RESEARCH\RESEARCH MOL\Surface Lattice\All-Dielectric\Silicon nanodisk\SemiAnalytic Calculation\COMSOL_Calculation"


def main(): 
    Idx = [2,3] 
    for idx in Idx:
        if idx==2:
            label = "EDx"
        elif idx==3:
            label = "MDy"
            
        epsd = 1.45
        filename = path()+'\Single_Scat_D200H100_EDxMDy_medAir'
        filein= f"{filename}.txt"
        fileout = f"{filename}_{label}.txt"
        fileclean = f"{filename}_{label}_clean.txt"
        Separate_file(filein,fileout,idx)
        ChangeComplex(fileout,fileclean)
        os.remove(fileout)
    #fig = plt.figure()
    fig = plt.figure(figsize=(8,6))
    eps0 = epsilon_0
    zeta0 = mu_0*c
    E0 = 1 #V/m
    
    fnameED = np.loadtxt(filename+"_EDx_clean.txt",dtype=complex) 
    fnameMD = np.loadtxt(filename+"_MDy_clean.txt",dtype=complex)
    
    x = fnameED[:,0] 
    yED = fnameED[:,1]/(eps0*epsd*E0)
    yMD = fnameMD[:,1]*zeta0/E0*c
    Xi=np.argmax(yMD.imag)
    
    yED_mod = modify_pol(yED, H=100e-9, n_d=1.45, n_s=1)
    yMD_mod = modify_pol(yMD, H=100e-9, n_d=1.45, n_s=1)
    
    plt.plot(x,yED.imag
            #,color="blue",label="Re $\u03B1^{MD}$",linewidth=3)
            ,color="black",label="Re $\u03B1^{ED}$",linewidth=3)
    plt.plot(x,yED_mod.imag
    #plt.plot(x,yMD.imag*1e20
            #,"--",color="red",label="Im $\u03B1^{ED}$",linewidth=3) 
            ,"--",color="blue",label="Re $\u03B1^{ED Modified}$",linewidth=3) 
    """ 
    plt.plo(x
            ,yMD.real*1e20
            #,np.degrees(np.angle(yED))
            ,color="Green",label="Re $\u03B1^{MD}$",linewidth=3) 
            #,"--",color="black",label="Phase-$p_x$",linewidth=3) 
    plt.plot(x,yMD.imag*1e20
            ,"--",color="Green",label="Im $\u03B1^{MD}$",linewidth=3)
            #,color="red",label="Amp-$m_y$/c",linewidth=3)
    """ 
    EdgeSizeGraph(left=.13, right=.88, top=.92, bottom=.17) 
    plt.xlabel("Wavelength (nm)", fontsize = 20) 
    plt.ylabel("$\u03B1$ ($nm^{3}$)", fontsize = 20)
    #plt.ylabel("$\u03B1$ ($\u00B5m^{3}$)", fontsize = 20)
    #ax1.set_ylabel("Real $\u03B1$ ($m^{3}$)", fontsize = 20, labelpad=10)
    plt.legend(loc="best",fontsize=15, frameon=False)
    plt.tick_params(axis="y", direction="out", left=True, labelsize=20)
    plt.tick_params(axis="x", direction="out",pad=10, bottom=True, top=False)
    plt.xticks(fontsize=20) 
    plt.yticks(fontsize=20) 
    plt.xlim(600,1000) 
    #plt.ylim(-1,2)
    yy = np.arange(-1,2.1,2)
    format_ylabels = [f'{tick:.0f}' if tick!=0 and tick!=1 else '0' if tick==0 else "1" for tick in yy]
    #plt.yticks(yy, format_ylabels, fontsize=20)
    plt.xticks(np.arange(600,1100,100))
    plt.text(0, 1.07, r"$\times 10^{7}$", transform=plt.gca().transAxes, fontsize=20, verticalalignment='top', horizontalalignment='left')
    #plt.tight_layout()
    plt.show()

def Separate_file(filein,fileout,idx):
    fileOut = open(fileout,"w") 
    fileOut.write("#nm C*m\n") 
    with open(filein,"r") as file:
        lines = file.readlines()[5:]
        for line in lines:
            line = line.split()
            res = "{} {}\n".format(line[0],line[idx])
            fileOut.write(res) 
        fileOut.close()

def ChangeComplex(fileout,fileout_clean):
    fileout = fileout
    fileout_rl = fileout_clean
    fname0 = open(fileout,"r")
    fname_rl = open(fileout_rl,"w")
    fname = fname0.readlines()
    for rl in fname: 
        if "i" in rl:
            rl = rl.replace("i","j")
            fname_rl.write(rl)
        else:
            fname_rl.write(rl)
    fname_rl.close()
    fname0.close()

def writing_file(filename,file_Out,idx):
    with open(filename,"r") as file:
        lines = file.readlines()[5:]
        with open(file_Out,"w") as file:
            for line in lines:
                line=line.split()
                data = "{} {} {}\n".format(line[0],line[1],line[idx])
                file.write(data)
def EdgeSizeGraph(left=.25, right=.92, top=.98, bottom=.14):
    return plt.subplots_adjust(left=left, right=right, top=top, bottom=bottom)

def modify_pol(alphaP_xx, H, n_d=1.45, n_s=1):
    eps0 = epsilon_0
    epsd = (n_d)**2
    epss = (n_s)**2
    zp = H/2
    res = ((epsd-epss)/(epsd+epss)) / ((4*np.pi)*(2*zp)**3)
    return 1/((1/alphaP_xx)+res)


main()








