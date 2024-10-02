# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 11:30:37 2019

@author: othmane.mounjid
"""

### Import path 


### Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

################################################################################################################
################################################################################################################
##################################### Basic functions ##########################################################
################################################################################################################
################################################################################################################
#### Regeneration function
def Regen_func_basic(evt, Lob_state, Agent_state = None): 
    ''' 
    Lob_state  : list = [qb (int),qa(int),s(int),pb(float),pa(float)] 
    evt : int in {0,...,7}          
    '''    
    if (evt == 1) or (evt == 2) : ## Buy cancel/ Sell market order consumed best bid
       Lob_state[0] = 1
       Lob_state[2] = min(Lob_state[2] + 1,size_s)
       Lob_state[3] -= tick
       Lob_state[4] = min(Lob_state[4], Lob_state[3] + size_s*0.1)
    elif (evt == 5) or (evt == 6) : ## Sell cancel/ Buy market order order consumed best ask
       Lob_state[1] = 1
       Lob_state[2] = min(Lob_state[2] + 1,size_s)
       Lob_state[3] = max(Lob_state[3], Lob_state[4] - size_s*0.1)
       Lob_state[4] += tick
    elif (evt == 3) : ## Buy limit order within the spread
       Lob_state[0] = size_lmid
       Lob_state[2] = max(Lob_state[2] - 1,1)
       Lob_state[3] += tick
       Lob_state[4] = min(Lob_state[4], Lob_state[3] + size_s*0.1)
    elif (evt == 7) : ## Sell limit order within the spread
       Lob_state[1] = size_lmid
       Lob_state[2] = max(Lob_state[2] - 1,1) 
       Lob_state[3] = max(Lob_state[3], Lob_state[4] - size_s*0.1)
       Lob_state[4] -= tick
       
### Market simulation
def Lob_simu(Lob_state_0,Intens_val,nb_iter, write_option = True, Regen_func = Regen_func_basic):
    cols_interest = ['Limit', 'Cancel', 'Market', 'MidLimit']
    Lob_state = list(Lob_state_0) #
    if write_option:
        df_res = pd.DataFrame(np.zeros((nb_iter+1,6)), columns = ['BB size','BA size','Spread','BB price','BA price','Event'])
        df_res.loc[0,:] = Lob_state + [None]
    intensities_values = np.zeros(8)
    
    
    for n in range(nb_iter):# n = 0
        #### Get market decision
        index_row_b = (Lob_state[0]-1) + (Lob_state[1]-1)*size_q + (Lob_state[2] - 1)*size_q*size_q
        index_row_a = (Lob_state[1]-1) + (Lob_state[0]-1)*size_q + (Lob_state[2] - 1)*size_q*size_q    
        intensities_values[:4] = Intens_val.loc[index_row_b, cols_interest]
        intensities_values[4:] = Intens_val.loc[index_row_a, cols_interest]
        times = np.random.exponential(1/intensities_values)
        index_min = times.argmin()
        
        #### Apply market decision
        if index_min == 0: ## Buy Limit order
            Lob_state[0] += 1
        elif index_min == 1: ## Buy Cancel order
            if Lob_state[0] <= 1 : # index_min = 1
                ### Regeneration 
                Regen_func(index_min, Lob_state)
            else:
                Lob_state[0] -= 1 
        elif index_min == 2: ## Sell Market order
            if Lob_state[0] <= 1 : 
                ### Regeneration 
                Regen_func(index_min, Lob_state)
            else:
                Lob_state[0] -= 1     
        elif index_min == 3: ## Buy limit order with the spread
            ### Regeneration
            Regen_func(index_min, Lob_state)
        elif index_min == 4: ## Sell  Limit order
            Lob_state[1] += 1    
        elif index_min == 5: ## Sell Cancel order
            if Lob_state[1] <= 1 : 
                ### Regeneration 
                Regen_func(index_min, Lob_state)
            else:
                Lob_state[1] -= 1    
        elif index_min == 6: ## Buy Market order
            if Lob_state[1] <= 1 : 
                ### Regeneration 
                Regen_func(index_min, Lob_state)
            else:
                Lob_state[1] -= 1    
        else: ## Sell limit order with the spread
            ### Regeneration
            Regen_func(index_min, Lob_state)

        ### Write result
        if write_option:
            df_res.loc[n+1,:] = Lob_state + [index_min]
            
    if write_option:
        return {'lob':Lob_state, 'History':df_res}
    else:
        return {'lob':Lob_state, 'History':pd.DataFrame()}

################################################################################################################
################################################################################################################
##################################### Matrix generator and some manipulations #################################################
################################################################################################################
################################################################################################################

### Computation generator matrix with constraints when spread is constant
def Build_Q_simple(Intens_val, size_q):
    
    size_Q_tilde = (size_q)*(size_q)
    Q_tilde = np.zeros((size_Q_tilde,size_Q_tilde))
    z_1 = np.zeros((size_Q_tilde,2*size_q))
    
    ## Build Q transition matrix 
    for qsame in range(size_q) : # QSame Loop // qsame = 1
        for qopp in range(size_q) : # QOpp Loop // qopp = 1
            CumIntens = 0.
            ## Cancellation order bid side : 
            if (qsame > 0) : ## the limit is not totally consumed  // No regeneration
                 CumIntens +=  Intens_val.loc[qsame*size_q+qopp ,['Cancel','Market']].sum()  
                 Q_tilde[qsame*size_q+qopp][(qsame-1)*size_q+qopp] += Intens_val.loc[qsame*size_q+qopp,['Cancel','Market']].sum()   
            else:
                 CumIntens +=  Intens_val.loc[qsame*size_q+qopp ,['Cancel','Market']].sum()  
                 z_1[qsame*size_q+qopp][qopp] += Intens_val.loc[qsame*size_q+qopp,['Cancel','Market']].sum()   
                
            ## Cancellation order ask side :
            if (qopp > 0) : ## the limit is not totally consumed // no regeneration
                 CumIntens +=  Intens_val.loc[qopp*size_q+qsame,['Cancel','Market']].sum() # IntensVal['lambdaCancel'][qopp*Qmax0+qsame]  
                 Q_tilde[qsame*size_q+qopp][qsame*size_q+(qopp-1)] += Intens_val.loc[qopp*size_q+qsame,['Cancel','Market']].sum()
            else:
                 CumIntens +=  Intens_val.loc[qopp*size_q+qsame ,['Cancel','Market']].sum()  
                 z_1[qsame*size_q+qopp][size_q + qsame] += Intens_val.loc[qopp*size_q+qsame,['Cancel','Market']].sum()   
    
                 
            ## Insertion order bid side :
            if (qsame < size_q-1) : ## when qsame = Qmax -1  no more order can be added to the bid limit
                 CumIntens +=  Intens_val.loc[qsame*size_q+qopp,'Limit'] # IntensVal['lambdaIns'][qsame*Qmax0+qopp]  
                 Q_tilde[qsame*size_q+qopp][(qsame+1)*size_q+qopp] += Intens_val.loc[qsame*size_q+qopp,'Limit']            
            
            ## Insertion oder ask side
            if (qopp < size_q-1) : ## when qopp = Qmax -1  no more order can be added to the ask limit
                 CumIntens +=  Intens_val.loc[qopp*size_q+qsame,'Limit'] # IntensVal['Limit'][qopp*Qmax0+qsame]  
                 Q_tilde[qsame*size_q+qopp][qsame*size_q+qopp+1] += Intens_val.loc[qopp*size_q+qsame,'Limit'] 
            
            ## Nothing happen 
            Q_tilde[qsame*size_q+qopp][qsame*size_q+qopp] += - CumIntens  
    
           
    return [Q_tilde,z_1]

### Computation of the generator matrix when spread is constant
def Build_Q_no_regen(Intens_val,size_q):

    size_Q_tilde = (size_q)*(size_q)
    Q_tilde = np.zeros((size_Q_tilde,size_Q_tilde))
    
    ## Build Q transition matrix 
    for qsame in range(size_q) : # QSame Loop // qsame = 1
        for qopp in range(size_q) : # QOpp Loop // qopp = 1
            CumIntens = 0.
            ## Cancellation order bid side : 
            if (qsame > 0) : ## the limit is not totally consumed  // No regeneration
                 CumIntens +=  Intens_val.loc[qsame*size_q+qopp ,['Cancel','Market']].sum()  
                 Q_tilde[qsame*size_q+qopp][(qsame-1)*size_q+qopp] += Intens_val.loc[qsame*size_q+qopp,['Cancel','Market']].sum()   

            ## Cancellation order ask side :
            if (qopp > 0) : ## the limit is not totally consumed // no regeneration
                 CumIntens +=  Intens_val.loc[qopp*size_q+qsame,['Cancel','Market']].sum() # IntensVal['lambdaCancel'][qopp*Qmax0+qsame]  
                 Q_tilde[qsame*size_q+qopp][qsame*size_q+(qopp-1)] += Intens_val.loc[qopp*size_q+qsame,['Cancel','Market']].sum()

                 
            ## Insertion order bid side :
            if (qsame < size_q-1) : ## when qsame = Qmax -1  no more order can be added to the bid limit
                 CumIntens +=  Intens_val.loc[qsame*size_q+qopp,'Limit'] # IntensVal['lambdaIns'][qsame*Qmax0+qopp]  
                 Q_tilde[qsame*size_q+qopp][(qsame+1)*size_q+qopp] += Intens_val.loc[qsame*size_q+qopp,'Limit']            
            
            ## Insertion oder ask side
            if (qopp < size_q-1) : ## when qopp = Qmax -1  no more order can be added to the ask limit
                 CumIntens +=  Intens_val.loc[qopp*size_q+qsame,'Limit'] # IntensVal['Limit'][qopp*Qmax0+qsame]  
                 Q_tilde[qsame*size_q+qopp][qsame*size_q+qopp+1] += Intens_val.loc[qopp*size_q+qsame,'Limit'] 
            
            ## Nothing happen 
            Q_tilde[qsame*size_q+qopp][qsame*size_q+qopp] += - CumIntens  
            
    return Q_tilde

### Compute the probabilities of execution
def Compute_prob_exec(Q_tilde,z_1):
    ##### Compue proba values
    Proba_exec = np.linalg.solve(Q_tilde.transpose(),-z_1)
    Proba_exec = Proba_exec/Proba_exec.sum(axis = 1)[:,None]
    return Proba_exec

def Compute_indexes(size_q):
    size_sym = (size_q+1)*size_q//2
    cumulate = 0
    indexes_ = np.zeros(size_sym,dtype = int)
    indexes_sym = np.zeros(size_sym,dtype = int)
    for q1 in range(size_q):
        for q2 in range(q1+1): 
            indexes_[cumulate + q2] = q1*size_q+q2
            indexes_sym[cumulate + q2] = q2*size_q+q1
        cumulate += (q1+1)
    return [indexes_, indexes_sym]

def Build_A(Average_proba_regen, Average_price_move_q, size_q):
    size_sym = (size_q+1)*size_q//2
    Average_price_move_q_sym = np.zeros(size_sym)
    A = np.zeros((size_sym,size_sym))
    cumulate = 0
    for qsame in range(size_q): # qsame = 0
        for qopp in range(qsame+1): # qopp = 0
            ratio =  (1 - Average_proba_regen[qsame*size_q + qopp,qsame*size_q + qopp] + Average_proba_regen[qsame*size_q + qopp, qopp*size_q + qsame])
            A[cumulate + qopp,:] = (Average_proba_regen[qsame*size_q + qopp,indexes_] - Average_proba_regen[qsame*size_q + qopp,indexes_sym])/ratio
            A[cumulate + qopp,cumulate + qopp] = 0
            Average_price_move_q_sym[cumulate+ qopp] = Average_price_move_q[qsame*size_q + qopp]/ratio
        cumulate += (qsame +1)
    return [A, Average_price_move_q_sym]        

### Compute stationary distribution queues
def Proba_stat(Tilde_Q,size_q): 
    size = size_q*size_q
    Tilde_Q_inv = np.array(Tilde_Q[:-1,:-1]) 
    for j in range(size-1):
        Tilde_Q_inv[:,j] -= Tilde_Q[-1,j]  
    F_inv = -Tilde_Q[size-1,:-1]

    ## Compute the stat proba
    Proba2 = np.zeros((size))
    Proba2[:-1] = np.linalg.solve(Tilde_Q_inv.transpose(),F_inv.transpose());Proba2[-1]  = 1-sum(Proba2)
    
    return Proba2


### Compute Embedded Markov chain
def P_prob(Tilde_Q,size_q):
    size = size_q*size_q
    P = np.zeros((size,size))
    ## Zero diagonal terms
    diagonal_indexes = np.arange(size)
    idx_true = Tilde_Q[diagonal_indexes,diagonal_indexes] == 0
    idx_zero = diagonal_indexes[idx_true]
    P[idx_zero,idx_zero] = 1
    
    ## Non zero diagonal terms
    idx_nzero = diagonal_indexes[~idx_true] 
    P[idx_nzero] = (-Tilde_Q[idx_nzero]/Tilde_Q[idx_nzero,idx_nzero][:,None])
    P[idx_nzero,idx_nzero] = 0

    return P

### Compute auto correl vector
def Auto_correl_vec(P,proba,move,k_max,size_q):
    size = size_q*size_q
    res = np.zeros(k_max)
    P_bis = np.eye(size)
    for k in range(k_max): # k = 0
        res[k] = sum((proba*move)*np.dot(P_bis,move))
        P_bis = np.dot(P,P_bis)
    return res

### Compute price volatility
def Compute_vol(vec_auto_correl):
    return vec_auto_correl[0] + 2*vec_auto_correl[1:].sum()
        
################################################################################################################
################################################################################################################
##################################### MC simu ##########################################################
################################################################################################################
################################################################################################################

def MC_simu_pi(NbSimu,Lob_state_0,Intens_val,nb_iter,write_option = True):
    pmid_0 = (Lob_state_0[3] + Lob_state_0[4])/2
    Val = np.zeros(nb_iter+1)
    Error = np.zeros(nb_iter+1)
    np.random.seed(0) ## fix random seed
    for n in range(NbSimu):#  n = 0
        dict_res_bis = Lob_simu(Lob_state_0,Intens_val,nb_iter, write_option = write_option)
        Df_lob_state = dict_res_bis['History']
        pmid_f = (Df_lob_state["BB price"] + Df_lob_state["BA price"])/2
        p_impact = (pmid_f - pmid_0)/tick
        Val += p_impact
        Error += (p_impact*p_impact) #  debug: print("value is : "+str(value)); print("Error value is : " +str(Error))
        if (n % 50) == 0:
            print(" n is :"  + str(n))
    Mean = Val/(NbSimu)
    n = (NbSimu-1) if (NbSimu>1) else NbSimu
    Var = ((Error)/n)-(NbSimu/n)*Mean*Mean
    return [Mean,Var]

################################################################################################################
################################################################################################################
##################################### Sym of the values ##########################################################
################################################################################################################
################################################################################################################

##### Auxiliary function : for the symmetrization of the intensities
def Sym_intens_val(Intens_val,size_q,size_s):
    index_sym = np.tile(np.arange(size_q),size_q)*size_q + np.repeat(np.arange(size_q),size_q)
    index_sym = np.tile(index_sym, size_s) + np.repeat(np.arange(size_s),size_q*size_q)*(size_q*size_q)
    nb_col = size_q*size_q*size_s
    nb_row = 7
    Intens_val_bis = pd.DataFrame(np.zeros((nb_col,nb_row)), columns = ['BB size','BA size','Spread','Limit','Cancel','Market','MidLimit'])
    cols_interests = ['Limit', 'Cancel', 'Market', 'MidLimit']
    Intens_val_bis.loc[:,cols_interests] = Intens_val.loc[index_sym,cols_interests].values
    Intens_val_bis['BB size'] = np.tile(np.repeat(np.arange(1,size_q+1),size_q),size_s)
    Intens_val_bis['BA size'] = np.tile(np.arange(1,size_q+1),size_q*size_s)
    Intens_val_bis['Spread'] = np.repeat(np.arange(1,size_s+1),size_q*size_q)
    return Intens_val_bis

################################################################################################################
################################################################################################################
##################################### Plot the values ##########################################################
################################################################################################################
################################################################################################################


def Plot_sns(df,option=False,path ="",ImageName="",xtitle="", xlabel ="", ylabel ="", annot =True , fig = False, a = 0, b = 0, subplot0 = 0, cbar = True, cmap="PuBu", mask = None, fmt = '.2g'):
    if not fig:
        ax = plt.axes()
    else:
        ax = fig.add_subplot(a,b,subplot0)
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    
    ax = sns.heatmap(df,cmap=cmap, ax = ax, annot = annot,cbar = cbar,mask=mask, fmt= fmt, annot_kws={"size": 20})
    ax.set_title(xtitle,fontsize = 18)
    ax.set_xlabel(xlabel,fontsize = 18)
    ax.set_ylabel(ylabel,fontsize = 18)
    ax.invert_yaxis()
    if option == "save" :
        plt.savefig(path+ImageName+".pdf", bbox_inches='tight') 
        
################################################################################################################
################################################################################################################
##################################### End ##########################################################
################################################################################################################
################################################################################################################

       
### Initialization of the parameters
path = "/users/eleves-a/2022/janis.aiad/3A/EAP1/HFT_QR_RL/HFT_QR_RL/models/QR/modified/Data/"
filename = "/users/eleves-a/2022/janis.aiad/3A/EAP1/HFT_QR_RL/HFT_QR_RL/models/QR/modified/Data/Intens_val_qr.csv"
Intens_val = pd.read_csv(filename, index_col = 0)

### Show the database
print(Intens_val.head(10))

### Simulation of the order book
##### Initialization of the parameters
size_q = Intens_val['BB size'].max()
size_n = 2
size_s = 2
size_lmid = 2
tick = 0.01
nb_iter = 1000
qb_0 = 1# 4
qa_0 = 1 # 4
s_0 = 1
pb_0 = 10
pa_0 = pb_0 + tick
Lob_state_0 = [qb_0,qa_0,s_0,pb_0,pa_0] 
write_option = True
np.random.seed(5)

##### Simulate the lob
dict_res = Lob_simu(Lob_state_0,Intens_val,nb_iter, write_option = write_option)
df_res = dict_res['History']
Lob_state = dict_res['lob']

##### Plot the result
plt.plot(df_res['BB price'],'b--')
plt.plot(df_res['BA price'],'r--')
plt.scatter(x = df_res.index, y =df_res['BB price'], s = df_res['BB size']*20)
plt.scatter(x = df_res.index, y =df_res['BA price'], s = df_res['BA size']*20)
plt.grid() 


### Computation of the price impact
##### MC simulation
####### Initialization of the parameters
NbSimu = 100 
nb_iter = 100
qb_0 = 2
qa_0 = 20
s_0 = 1
pb_0 = 10
pa_0 = pb_0 + tick
Lob_state_0 = [qb_0,qa_0,s_0,pb_0,pa_0] 

####### MC simu
Mean,Var = MC_simu_pi(NbSimu,Lob_state_0,Intens_val,nb_iter,write_option = True)

####### Plot values
plt.plot(Mean, 'b--')
plt.plot(Mean + 1.96*np.sqrt(Var/NbSimu), 'r--')
plt.plot(Mean - 1.96*np.sqrt(Var/NbSimu), 'r--')
plt.grid()
plt.show()

##### Numerical method
####### Initialization of the parameters
price_move = np.ones(2*size_q)*tick
price_move[:size_q] = -tick
d = np.zeros((2*size_q,size_q* size_q))
for q in range(size_q) : 
    d[q, 10*size_q + q] = 1
    d[q + size_q, q*size_q + 10] = 1
indexes_, indexes_sym = Compute_indexes(size_q)
size_sym = (size_q+1)*size_q//2

####### Build generator
Intens_val_bis  = Sym_intens_val(Intens_val,size_q,size_s)
Q_tilde, z_1 = Build_Q_simple(Intens_val_bis, size_q)

####### Compute proba execution
Proba_exec = Compute_prob_exec(Q_tilde,z_1)

####### Compute the price impact
Average_price_move_q = np.dot(Proba_exec,price_move)
Average_proba_regen = np.dot(Proba_exec,d)
A, Average_price_move_q_sym = Build_A(Average_proba_regen, Average_price_move_q, size_q)
P_impact_sym = np.linalg.solve((np.eye(size_sym) - A),Average_price_move_q_sym)

####### Plot the values
######### Proba execution
Proba_exec_bid = Proba_exec[:,:size_q].sum(axis = 1)
xpos1 =  np.repeat(np.arange(1,size_q+1),size_q)
ypos1 =  np.tile(np.arange(1,size_q+1),size_q)
data_frame = pd.DataFrame(np.zeros((size_q*size_q,3)),columns=['x','y','Prob'])
data_frame['x'] = xpos1
data_frame['y'] = ypos1
data_frame['Prob'] = Proba_exec_bid
option_save= False # "save"
path ="Image"; ImageName="\\Probexec_"; xtitle="" 
Plot_sns(data_frame.pivot("y", "x", "Prob"),option_save,path,ImageName, cbar = True, annot = False)
plt.show()

######### Price impact
P_impact = np.zeros(size_q*size_q)
P_impact[indexes_] = P_impact_sym #Average_price_move_q_sym
P_impact[indexes_sym] = -P_impact_sym
xpos1 =  np.repeat(np.arange(1,size_q+1),size_q)
ypos1 =  np.tile(np.arange(1,size_q+1),size_q)
data_frame = pd.DataFrame(np.zeros((size_q*size_q,3)),columns=['x','y','Prob'])
data_frame['x'] = xpos1
data_frame['y'] = ypos1
data_frame['Prob'] = P_impact
option_save= False # "save"
path ="Image"; ImageName="\\P_impact_"; xtitle="" 
Plot_sns(data_frame.pivot("y", "x", "Prob"),option_save,path,ImageName, cbar = True, annot = False)
plt.show()

### Computation of the stationary distribution
Q_no_regen = Build_Q_no_regen(Intens_val_bis,size_q)
proba = Proba_stat(Q_no_regen,size_q)

##### Plot the values
xpos1 = np.repeat(np.arange(1,size_q+1),size_q)
ypos1 = np.tile(np.arange(1,size_q+1),size_q)
data_frame = pd.DataFrame(np.zeros((size_q*size_q,3)),columns=['x','y','Prob'])
data_frame['x'] = xpos1
data_frame['y'] = ypos1
data_frame['Prob'] = proba
option_save= False # "save"
path ="Image"; ImageName="\\Proba_stat_"; xtitle="" 
Plot_sns(data_frame.pivot("y", "x", "Prob"),option_save,path,ImageName, cbar = True, annot = False)
plt.show()

### Computation of the price volatility
##### Initialize parameters 
P = P_prob(Q_no_regen,size_q) ## Proba transition matrix
######## Start Create price move vector
price_move = np.zeros(size_q*size_q)
price_move[np.arange(1,size_q)] = -1
price_move[np.arange(1,size_q)*size_q] = 1
######## End Create price move vector
k_max = 10
vec_auto_correl = Auto_correl_vec(P,proba,price_move,k_max,size_q) ## Auto correl vector

##### Compute vol meca
vol_mec = Compute_vol(vec_auto_correl)

##### Print result
print(" vol meca is : " + str(vol_mec))