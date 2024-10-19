# -*- coding: utf-8 -*-


### Import libraries
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns

### pandas printing options
pd.set_option('display.max_columns', 50)


################################################################################################################
################################################################################################################
##################################### Basic functions ##########################################################
################################################################################################################
################################################################################################################

#### Regeneration function
def Regen_func_agent(evt, Lob_state, Agent_state): 
    ''' 
    Lob_state  : list = [qb (int),qa(int),s(int),pb(float),pa(float)] 
    evt : int in {0,...,7}          
    '''    
    if (evt == 1) or (evt == 2) : ## Buy cancel/ Sell market order consumed best bid
       Lob_state[0] = size_n
       Lob_state[2] = min(Lob_state[2] + 1,size_s)
       Lob_state[3] -= tick
       Lob_state[4] = min(Lob_state[4], Lob_state[3] + size_s*tick)
    elif (evt == 5) or (evt == 6) : ## Sell cancel/ Buy market order order consumed best ask
       Lob_state[1] = size_n
       Lob_state[2] = min(Lob_state[2] + 1,size_s)
       Lob_state[4] += tick
       Lob_state[3] = max(Lob_state[3], Lob_state[4] - size_s*tick)
    elif (evt == 3) : ## Buy limit order within the spread
       Lob_state[0] = size_lmid
       Lob_state[2] = max(Lob_state[2] - 1,1)
       Lob_state[3] += tick
       Lob_state[4] = min(Lob_state[4], Lob_state[3] + size_s*tick)
       Agent_state['BB_position'] = Lob_state[0]-1
       Agent_state['BB_size'] = 0
    elif (evt == 7) : ## Sell limit order within the spread
       Lob_state[1] = size_lmid
       Lob_state[2] = max(Lob_state[2] - 1,1) 
       Lob_state[4] -= tick
       Lob_state[3] = max(Lob_state[3], Lob_state[4] - size_s*tick)

       

#### Check if a market decision is consitent
def Decision_isnot_consistent(decision,Lob_state, Agent_state):
    check1 = (Lob_state[0] == Agent_state['BB_size']) and (decision['evt'] == 1)
    return check1

#### Market decision function    
def Market_decision(Lob_state_before, Lob_state, Agent_state, Intens_val, dict_argument_Mkt_decision):
    global size_0
    cols_interest = dict_argument_Mkt_decision['cols_interest']
    index_row_b = (Lob_state[0] - 1) + (Lob_state[1]-1)*size_q + (Lob_state[2] - 1)*size_q*size_q
    index_row_a = (Lob_state[1]-1) + (Lob_state[0]-1)*size_q + (Lob_state[2] - 1)*size_q*size_q    
    try:
        dict_argument_Mkt_decision['intensities_values'][:4] = Intens_val.loc[index_row_b, cols_interest]
        dict_argument_Mkt_decision['intensities_values'][4:] = Intens_val.loc[index_row_a, cols_interest]
    except:
        print(" index_row_b is : " + str(index_row_b))
        print(" index_row_a is : " + str(index_row_a))
        print(" Lob_state is : " + str(Lob_state))
        raise ValueError(" Market decision function is not properly working")
    with np.errstate(divide='ignore'):
        times = np.random.exponential(1/dict_argument_Mkt_decision['intensities_values'])
    index_min = times.argmin()
    decision = {'evt' : index_min, 'size' : size_0, 'diff_time' : times.min()}
    while Decision_isnot_consistent(decision,Lob_state, Agent_state):
        decision = Market_decision(Lob_state_before, Lob_state, Agent_state, Intens_val, dict_argument_Mkt_decision)
    return decision

#### Apply market decision function  
def Apply_market_decision(decision,Lob_state, Agent_state, Regen_func = Regen_func_agent):
    if decision['evt'] == 0: ## Buy Limit order
        Lob_state[0] += decision['size']
        
    elif decision['evt'] == 1: ## Buy Cancel order
        if Lob_state[0] <= decision['size'] : ## Le checker au départ
            ### Regeneration 
            Regen_func(decision['evt'], Lob_state, Agent_state)
        else:
            Lob_state[0] -= decision['size']
            rest = Agent_state['BB_position'] - decision['size']
            Agent_state['BB_position'] = max(rest, 0) 
    elif decision['evt'] == 2: ## Sell Market order
        if Lob_state[0] <= decision['size'] : 
            Agent_state['P_exec'] += Lob_state[3]*Agent_state['BB_size'] ## Position is going to be updated in the regen func
            Agent_state['BB_size'] = 0
            Agent_state['remaining_qty'] -= Agent_state['BB_size']
            Agent_state['remaining_qty_period'] -= Agent_state['BB_size']
            ### vwap
            pen = 0
            Agent_state['VWAP'] += decision['size']*(Lob_state[3] + pen)
            Agent_state['VWAP_start_period'] += decision['size']*(Lob_state[3] + pen)
            Agent_state['Volume'] += decision['size']
            Agent_state['Volume_start_period'] += decision['size']
            ### Regeneration 
            Regen_func(decision['evt'], Lob_state, Agent_state)
        else:
            Lob_state[0] -= decision['size']
            rest = Agent_state['BB_position'] - decision['size']
            Agent_state['BB_position'] = max(rest, 0) 
            exec_qty = min(-min(rest,0),Agent_state['BB_size'])
            Agent_state['BB_size'] = Agent_state['BB_size'] - exec_qty
            Agent_state['P_exec'] += Lob_state[3]*exec_qty
            Agent_state['remaining_qty'] -= exec_qty
            Agent_state['remaining_qty_period'] -= exec_qty
            ### vwap
            pen = 0
            Agent_state['VWAP'] += decision['size']*(Lob_state[3] + pen)
            Agent_state['VWAP_start_period'] += decision['size']*(Lob_state[3] + pen)
            Agent_state['Volume'] += decision['size']
            Agent_state['Volume_start_period'] += decision['size']
            
    elif decision['evt'] == 3: ## Buy limit order within the spread
        ### Regeneration
        Regen_func(decision['evt'], Lob_state, Agent_state)
    elif decision['evt'] == 4: ## Sell  Limit order
        Lob_state[1] += decision['size']    
    elif decision['evt'] == 5: ## Sell Cancel order
        if Lob_state[1] <= decision['size'] : 
            ### Regeneration 
            Regen_func(decision['evt'], Lob_state, Agent_state)
        else:
            Lob_state[1] -= decision['size']    
    elif decision['evt'] == 6: ## Buy Market order
        if Lob_state[1] <= decision['size'] :
            ### vwap
            pen = 0
            Agent_state['VWAP'] += decision['size']*(Lob_state[4] + pen)
            Agent_state['VWAP_start_period'] += decision['size']*(Lob_state[4] + pen)
            Agent_state['Volume'] += decision['size']
            Agent_state['Volume_start_period'] += decision['size']
            ### Regeneration 
            Regen_func(decision['evt'], Lob_state, Agent_state)
        else:
            ### vwap
            pen = 0
            Agent_state['VWAP'] += decision['size']*(Lob_state[4] + pen)
            Agent_state['VWAP_start_period'] += decision['size']*(Lob_state[4] + pen)
            Agent_state['Volume'] += decision['size']
            Agent_state['Volume_start_period'] += decision['size']
            ### Regeneration 
            Lob_state[1] -= decision['size'] 
    elif decision['evt'] == 7: ## Sell limit order within the spread
        ### Regeneration
        Regen_func(decision['evt'], Lob_state, Agent_state)


################################################################################################################
##################################### Agents decision function #################################################
################################################################################################################

### Apply agents decision
def Apply_agent_decision_v1(decision,Lob_state, Agent_state, Regen_func = Regen_func_agent):
    if decision['evt'] == 0: ## Buy Limit order
        Lob_state[0] += (decision['size'] - Agent_state['BB_size'])
        Agent_state['BB_position'] = Lob_state[0] -1
        Agent_state['BB_size'] = decision['size']
        
    elif decision['evt'] == 8: ## Buy Market order order followed by buy limit order bbid
        ### Cancel the already inserted limit orders
        Lob_state[0] -= Agent_state['BB_size']
        Agent_state['BB_position'] = Lob_state[0] -1
        Agent_state['BB_size'] = 0     
        ### Buy Market order
        if Lob_state[1] <= decision['remaing_qty']: 
            pen = 0
            Agent_state['P_exec'] += Lob_state[4] * (decision['remaing_qty'] + pen)
            Agent_state['remaining_qty'] -= decision['remaing_qty']
            ### vwap
            pen = 0
            Agent_state['VWAP'] += decision['remaing_qty']*(Lob_state[3] + pen)
            Agent_state['VWAP_start_period'] += decision['remaing_qty']*(Lob_state[3] + pen)
            Agent_state['Volume'] += decision['remaing_qty']
            Agent_state['Volume_start_period'] += decision['remaing_qty']
            ### Regeneration 
            Regen_func(6, Lob_state, Agent_state)
        else:
            Lob_state[1] -= decision['remaing_qty']
            Agent_state['P_exec'] += Lob_state[4] * decision['remaing_qty']
            Agent_state['remaining_qty'] -= decision['remaing_qty']
            ### vwap
            pen = 0
            Agent_state['VWAP'] += decision['remaing_qty']*(Lob_state[3] + pen)
            Agent_state['VWAP_start_period'] += decision['remaing_qty']*(Lob_state[3] + pen)
            Agent_state['Volume'] += decision['remaing_qty']
            Agent_state['Volume_start_period'] += decision['remaing_qty']
        
        ### Regeneration if if bid queue 0 after cancellation
        if Lob_state[0] == 0: 
            Regen_func(1, Lob_state, Agent_state)
            
        ### Buy limit order 
        Lob_state[0] += decision['size'] 
        Agent_state['BB_position'] = Lob_state[0] -1
        Agent_state['BB_size'] = decision['size']
        
    elif decision['evt'] == 9: ## Buy Market order order 
        ### Cancel the already inserted limit orders
        Lob_state[0] -= Agent_state['BB_size']
        Agent_state['BB_position'] = Lob_state[0] -1
        Agent_state['BB_size'] = 0   
        
        ### Buy Market order
        if Lob_state[1] <= decision['remaing_qty']: 
            pen = 0
            Agent_state['P_exec'] += Lob_state[4] * (decision['remaing_qty'] + pen)
            Agent_state['remaining_qty'] -= decision['remaing_qty']
            Agent_state['remaining_qty_period'] -= decision['remaing_qty']
            ### vwap
            pen = 0
            Agent_state['VWAP'] += decision['remaing_qty']*(Lob_state[3] + pen)
            Agent_state['VWAP_start_period'] += decision['remaing_qty']*(Lob_state[3] + pen)
            Agent_state['Volume'] += decision['remaing_qty']
            Agent_state['Volume_start_period'] += decision['remaing_qty']
            ### Regeneration 
            Regen_func(6, Lob_state, Agent_state)
        else:
            Lob_state[1] -= decision['remaing_qty']
            Agent_state['P_exec'] += Lob_state[4] * decision['remaing_qty']
            Agent_state['remaining_qty'] -= decision['remaing_qty']
            Agent_state['remaining_qty_period'] -= decision['remaing_qty']
            ### vwap
            pen = 0
            Agent_state['VWAP'] += decision['remaing_qty']*(Lob_state[3] + pen)
            Agent_state['VWAP_start_period'] += decision['remaing_qty']*(Lob_state[3] + pen)
            Agent_state['Volume'] += decision['remaing_qty']
            Agent_state['Volume_start_period'] += decision['remaing_qty']
        
        ### Regeneration if bid queue 0 after cancellation
        if Lob_state[0] == 0: 
            Regen_func(1, Lob_state, Agent_state)

        ### Buy limit order 
        Lob_state[0] += decision['size'] 
        Agent_state['BB_position'] = Lob_state[0] -1
        Agent_state['BB_size'] = decision['size']
            
    elif decision['evt'] == 10: ## Buy Cancellation followed  by buy limit order 
        ### Cancel the already inserted limit orders
        Lob_state[0] -= Agent_state['BB_size']
        Agent_state['BB_position'] = Lob_state[0] -1
        Agent_state['BB_size'] = 0
        
        ### Regeneration if bid queue 0 after cancellation
        Regen_func(1, Lob_state, Agent_state)   
        
        ### Buy limit order 
        Lob_state[0] += decision['size'] 
        Agent_state['BB_position'] = Lob_state[0] -1
        Agent_state['BB_size'] = decision['size']

### Agents scheduling options
##### Linear scheduling
def Initialize_period_linear(Agent_state):
    Agent_state['end_period'] += Agent_state['period']
    if Agent_state['Volume_start_period']> 0 :
        Agent_state['VWAP_price_period'] += Agent_state['qty_start_period'] * (Agent_state['VWAP_start_period'])/Agent_state['Volume_start_period']
    else:
        Agent_state['VWAP_price_period'] = 0
    Agent_state['qty_start_period'] = max(min(Agent_state['qty_start']//Agent_state['nb_period'],Agent_state['remaining_qty'] - Agent_state['remaining_qty_period']),Agent_state['size'])
    Agent_state['remaining_qty_period'] = Agent_state['qty_start_period']
    Agent_state['VWAP_start_period'] = 0
    Agent_state['Volume_start_period'] = 0


##### Exponential scheduling
def Initialize_period_expon(Agent_state):
    Agent_state['end_period'] += Agent_state['period']
    if Agent_state['Volume_start_period']> 0 :
        Agent_state['VWAP_price_period'] += Agent_state['qty_start_period'] * (Agent_state['VWAP_start_period'])/Agent_state['Volume_start_period']
    else:
        Agent_state['VWAP_price_period'] = 0
    index_period_end = int(Agent_state['end_period']/Agent_state['period'])
    index_period_start = index_period_end - 1
    if index_period_end == Agent_state['nb_period']:
        qty_start = int(Agent_state['qty_start']*np.exp(-index_period_start/4))
    else:
        qty_start = int(Agent_state['qty_start']*(np.exp(-index_period_start/4) - np.exp(-index_period_end/4)))
        
    Agent_state['qty_start_period'] = max(min(qty_start,Agent_state['remaining_qty'] - Agent_state['remaining_qty_period']),Agent_state['size'])
    Agent_state['remaining_qty_period'] = Agent_state['qty_start_period']
    Agent_state['VWAP_start_period'] = 0
    Agent_state['Volume_start_period'] = 0

##################################### Basic agent #################################################
    
### Basic
def Agent_decision_basic(Lob_state_before,Lob_state, Agent_state, Initialize_period_linear_func = Initialize_period_linear):
    if (Agent_state['time'] >= Agent_state['end_period']) and (Agent_state['remaining_qty'] > 0): ## 
        ### Initialize agent start period
        Initialize_period_linear_func(Agent_state)
        
    agent_decision = {'evt' : None, 'size' : 0}
    return agent_decision

##################################### Agent strategy T1 ############################################

### T1 strategy
def T1_agent_decision(Lob_state_before,Lob_state, Agent_state, Initialize_period_linear_func = Initialize_period_linear):
    
    ### End of period
    if (Agent_state['time'] >= Agent_state['end_period']) and (Agent_state['remaining_qty'] > 0): ## 
        ### Send a market order
        agent_decision = {'evt' : 8, 'size' : min(Agent_state['size'],Agent_state['remaining_qty'] - Agent_state['remaining_qty_period']), 'remaing_qty' : Agent_state['remaining_qty_period']}
        ### Initialize agent start period
        Initialize_period_linear_func(Agent_state)
    
    ### Mid price moved  
    elif ((Lob_state_before[3] != Lob_state[3]) | (Lob_state_before[4] != Lob_state[4])) and (Agent_state['remaining_qty'] > 0):
        agent_decision = {'evt' : 9, 'size' : min(Agent_state['size'],Agent_state['remaining_qty_period'] - Agent_state['BB_size']), 'remaing_qty' : Agent_state['BB_size']}
    
    ### There is no qty inserted at bbid    
    elif (Agent_state['BB_size'] == 0) and (Agent_state['remaining_qty'] > 0):## send a buy limit order
        agent_decision = {'evt' : 0, 'size' : min(Agent_state['size'],Agent_state['remaining_qty_period'])}
     
    ### Do nothing    
    else:
        agent_decision = {'evt' : None, 'size' : 0}
        
    return agent_decision

##################################### Agent strategy T2 ############################################

### T2 strategy
def T2_agent_decision(Lob_state_before,Lob_state, Agent_state, Initialize_period_linear_func = Initialize_period_linear):
    
    ### End of period
    if (Agent_state['time'] >= Agent_state['end_period']) and (Agent_state['remaining_qty'] > 0): ## 
        ### Send a market order
        agent_decision = {'evt' : 8, 'size' : min(Agent_state['size'],Agent_state['remaining_qty'] - Agent_state['remaining_qty_period']), 'remaing_qty' : Agent_state['remaining_qty_period']}
        ### Initialize agent start period
        Initialize_period_linear_func(Agent_state)
    
    ### BBid price moved 
    elif (Lob_state_before[3] != Lob_state[3]) and (Agent_state['remaining_qty'] > 0):
        agent_decision = {'evt' : 8, 'size' : min(Agent_state['size'],Agent_state['remaining_qty_period']),'remaing_qty' : 0}
    
    ### Last in the queue
    elif (Lob_state[0] == Agent_state['BB_size']) and (Agent_state['remaining_qty'] > 0):
        agent_decision = {'evt' : 10, 'size' : Agent_state['size']}
    
    ### There is no qty inserted at bbid  
    elif (Agent_state['BB_size'] == 0) and (Agent_state['remaining_qty'] > 0): ## send a buy limit order
        agent_decision = {'evt' : 0, 'size' : min(Agent_state['size'],Agent_state['remaining_qty_period'])}
    
    ### Do nothing
    else:
        agent_decision = {'evt' : None, 'size' : 0}
        
    return agent_decision


################################################################################################################
##################################### Market simulation #################################################
################################################################################################################

### Market simulation
def Lob_simu_agent(Lob_state_0,Intens_val,nb_iter, Tf = 100, write_option = True, Agent_decision_func = Agent_decision_basic, Agent_state = {},
                   Apply_agent_decision = Apply_agent_decision_v1, write_price =  False):


    ### Initialization
    cols_interest = ['Limit', 'Cancel', 'Market', 'MidLimit']
    Lob_state = list(Lob_state_0) #
    Lob_state_before_agent = list(Lob_state_0)
    Lob_state_before_market = list(Lob_state_0)
    time = 0
    n = 0

    ### Write in df:
    if write_option:
        df_res = pd.DataFrame(np.zeros((3*nb_iter+1,14)), columns = ['BB size','BA size','Spread','BB price','BA price','Event','time','remaing_qty', 'remaing_qty_prd','state','BB_size', 'BB_position',
                              'qty_start_period', 'P_exec'])
        df_res.loc[0,:] = Lob_state + [None,time,Agent_state['remaining_qty'],Agent_state['remaining_qty_period'],'init', Agent_state['BB_size'], Agent_state['BB_position'], Agent_state['qty_start_period'], Agent_state['P_exec']]
    intensities_values = np.zeros(8)
    dict_argument_Mkt_decision = {'cols_interest' : cols_interest,
                                  'intensities_values' : intensities_values}

    ### Write only price    
    if write_price:
        df_price = pd.DataFrame(np.zeros((nb_iter+1,3)), columns = ['BB price','BA price','time'])
        df_price.loc[0,:] = Lob_state[3:] + [time]

    
    ### Main loop
    while (n < nb_iter) and (time <= Tf ):# n = 0 # Tf 
        ### Get agent decision 
        agent_decision = Agent_decision_func(Lob_state_before_market,Lob_state,Agent_state)
                
        ### Apply agent decision ## change market state
        Lob_state_before_agent = copy.deepcopy(Lob_state)
        Agent_state_before_agent = copy.deepcopy(Agent_state)
        Apply_agent_decision(agent_decision,Lob_state,Agent_state)
        

        #### Get market decision
        market_decision = Market_decision(Lob_state_before_agent,Lob_state,Agent_state, Intens_val, dict_argument_Mkt_decision)
        time += market_decision['diff_time']
        Agent_state['time'] = time
        n += 1
        
        #### Apply market decision ## change agent state
        Lob_state_before_market = copy.deepcopy(Lob_state)
        Agent_state_before_market = copy.deepcopy(Agent_state)
        Apply_market_decision(market_decision, Lob_state, Agent_state)
        
        
        ### Write result
        if write_option:                    
            df_res.loc[3*(n-1)+1,:] = Lob_state_before_agent + [agent_decision['evt'],time, Agent_state_before_agent['remaining_qty'], Agent_state_before_agent['remaining_qty_period'],'bef_a', Agent_state_before_agent['BB_size'], Agent_state_before_agent['BB_position'], Agent_state_before_agent['qty_start_period'], Agent_state_before_agent['P_exec']] 
            df_res.loc[3*(n-1)+2,:] = Lob_state_before_market + [market_decision['evt'],time,Agent_state_before_market['remaining_qty'],Agent_state_before_market['remaining_qty_period'],'bef_m',Agent_state_before_market['BB_size'],Agent_state_before_market['BB_position'], Agent_state_before_market['qty_start_period'], Agent_state_before_market['P_exec']] 
            df_res.loc[3*(n-1)+3,:] = Lob_state + [market_decision['evt'],time,Agent_state['remaining_qty'],Agent_state['remaining_qty_period'],'aft_m',Agent_state['BB_size'],Agent_state['BB_position'], Agent_state['qty_start_period'], Agent_state['P_exec']] 
        
        if write_price:
            df_price.loc[n,:] = Lob_state[3:] + [time]

            
    res = {'lob':Lob_state, 'Agent_state': Agent_state}
    if write_option:
        res['History'] = df_res.loc[:3*n,:]
    else:
        res['History'] = pd.DataFrame()
    
    if write_price:
        res['price'] = df_price.loc[:n,:]
    else:
        res['price'] = pd.DataFrame()
        
    return res

################################################################################################################
################################################################################################################
##################################### MC estimation ############################################################
################################################################################################################
################################################################################################################

### Computation price impact
def MC_simu_pi_1(NbSimu,Lob_state_0,Intens_val,nb_iter,t_step_index = 1,Tf = 100,write_option = False,
                 Agent_decision_func = Agent_decision_basic, Agent_state = {},
                 Apply_agent_decision = Apply_agent_decision_v1, write_price =  True):
    
    global period, size_agent, nb_period, inventory, time_step_0
    ### Initialization of the parameters
    t_end_index = Tf
    time_index = np.arange(0,t_end_index,t_step_index)
    df_final = pd.DataFrame(np.empty((time_index.shape[0],2)), columns = ['BB price','BA price'], index = time_index)
    df_final[:] = np.nan

    Val = np.zeros(time_index.shape[0]+1)
    Error = np.zeros(time_index.shape[0]+1)
    pmid_0 = (Lob_state_0[3] + Lob_state_0[4])/2

    for n in range(NbSimu):#  n = 0
        ### Initialization parameters before loop
        Agent_state = {'time' : 0, 'end_period' : 0, 'period' : period, 'size' : size_agent, 'nb_period' : nb_period,
                       'remaining_qty' : inventory, 'qty_start' : inventory, 'time_step' : time_step_0,
                       'remaining_qty_period' : 0, 'qty_start_period' : 0, 
                       'BB_size' : 0, 'BB_position' : 0, 
                       'BA_size' : 0, 'BA_position' : 0,
                       'P_exec' : 0, 'VWAP' : 0, 'VWAP_start_period' : 0,  'VWAP_price_period' : 0,
                       'Volume' : 0, 'Volume_start_period' : 0}     
     
        ### Lob simu
        dict_res_bis = Lob_simu_agent(Lob_state_0,Intens_val,nb_iter, Tf = Tf, write_option = write_option, Agent_decision_func = Agent_decision_func, Agent_state = Agent_state,
                       Apply_agent_decision = Apply_agent_decision,write_price = write_price)

        ### Process result : same timestamp
        Df_lob_state = dict_res_bis['price'].set_index("time")
        df_temp = Df_lob_state._append(df_final).sort_index().fillna(method = 'ffill').loc[time_index,:]
        pmid_f = (df_temp["BB price"] + df_temp["BA price"])/2
        p_impact = (pmid_f - pmid_0)/pmid_0

        ### Update estimation
        Val += p_impact
        Error += (p_impact*p_impact) #  debug: print("value is : "+str(value)); print("Error value is : " +str(Error))
        if (n % 5) == 0:
            print(" n is :"  + str(n))

    Mean = Val/(NbSimu)
    n = (NbSimu-1) if (NbSimu>1) else NbSimu
    Var = ((Error)/n)-(NbSimu/n)*Mean*Mean
    return [Mean, Var]

### Computation price impact
def MC_simu_pi_2(NbSimu,Lob_state_0,Intens_val,nb_iter,t_step_index = 1,Tf = 100,write_option = True,
                 Agent_decision_func = Agent_decision_basic, Agent_state = {},
                 Apply_agent_decision = Apply_agent_decision_v1):
    
    global period, size_agent, nb_period, inventory, time_step_0
    ### Initialization of the parameters
    t_end_index = Tf
    time_index = np.arange(0,t_end_index,t_step_index)
    df_final = pd.DataFrame(np.empty((time_index.shape[0],2)), columns = ['BB price','BA price'], index = time_index)
    df_final[:] = np.nan

    Val = np.zeros(time_index.shape[0]+1)
    Error = np.zeros(time_index.shape[0]+1)

    for n in range(NbSimu):#  n = 0
        ### Initialization parameters before loop
        Agent_state = {'time' : 0, 'end_period' : period, 'period' : period, 'size' : size_agent, 'nb_period' : nb_period,
                   'remaining_qty' : inventory, 'qty_start' : inventory,  'time_step' : time_step_0,
                   'remaining_qty_period' : 0, 'qty_start_period' : 0, 
                   'BB_size' : 0, 'BB_position' : 0, 
                   'BA_size' : 0, 'BA_position' : 0,
                   'P_exec' : 0, 'VWAP' : 0, 'VWAP_start_period' : 0, 'VWAP_price_period' : 0,
                   'Volume' : 0, 'Volume_start_period' : 0}
        
        ### Lob simu
        dict_res_bis = Lob_simu_agent(Lob_state_0,Intens_val,nb_iter, Tf = Tf, write_option = write_option, Agent_decision_func = Agent_decision_func, Agent_state = Agent_state,
                       Apply_agent_decision = Apply_agent_decision)
        
        ### Process result : same timestamp
        Df_lob_state = dict_res_bis['History'].loc[::3,:].set_index("time")[["BB price","BA price"]]
        df_temp = Df_lob_state._append(df_final).sort_index().fillna(method = 'ffill').loc[time_index,:]
        pmid_f = (df_temp["BB price"] + df_temp["BA price"])/2
        pvwap = Agent_state['VWAP'] / Agent_state['Volume']
        p_impact = (pmid_f - pvwap)/tick
        
        ### Update estimation
        Val += p_impact
        Error += (p_impact*p_impact) #  debug: print("value is : "+str(value)); print("Error value is : " +str(Error))
        if (n % 5) == 0:
            print(" n is :"  + str(n))

    Mean = Val/(NbSimu)
    n = (NbSimu-1) if (NbSimu>1) else NbSimu
    Var = ((Error)/n)-(NbSimu/n)*Mean*Mean
    return [Mean, Var]


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

###### Implementation of strategies

################################################################################################################
########### Strategy 0 : Basic
################################################################################################################

########## Initialization of the parameters
size_q = Intens_val['BB size'].max()
size_n = 2
size_s = 2
size_0 = 1
size_lmid = 2
tick = 0.1
nb_iter = 100
qb_0 = 4
qa_0 = 4
s_0 = 1
pb_0 = 10
pa_0 = pb_0 + tick
Lob_state_0 = [qb_0,qa_0,s_0,pb_0,pa_0] 
period = 10
nb_period = 10
Tf = period*nb_period
size_agent = 1
inventory =  10
Agent_state = {'time' : 0, 'end_period' : 0, 'period' : period, 'size' : size_agent, 'nb_period' : nb_period,
               'remaining_qty' : inventory, 'qty_start' : inventory,
               'remaining_qty_period' : inventory//nb_period, 'qty_start_period' : inventory//nb_period, 
               'BB_size' : 0, 'BB_position' : 0, 
               'BA_size' : 0, 'BA_position' : 0,
               'P_exec' : 0, 'VWAP' : 0, 'VWAP_start_period' : 0,  'VWAP_price_period' : 0,
               'Volume' : 0, 'Volume_start_period' : 0}  
write_option = True
Agent_decision_func = Agent_decision_basic
Apply_agent_decision = Apply_agent_decision_v1


##### Simulate the lob
dict_res = Lob_simu_agent(Lob_state_0,Intens_val,nb_iter, write_option = write_option,
                          Agent_decision_func= Agent_decision_func,
                          Agent_state = Agent_state)
df_res = dict_res['History']
Lob_state = dict_res['lob']


##### Plot the result
plt.plot(df_res['BB price'],'b--')
plt.plot(df_res['BA price'],'r--')
plt.scatter(x = df_res.index, y =df_res['BB price'], s = df_res['BB size']*20)
plt.scatter(x = df_res.index, y =df_res['BA price'], s = df_res['BA size']*20)
plt.grid()
plt.show()


################################################################################################################
######## Strategy 1 : Fire and forget : within a period
################################################################################################################

########## Initialization of the parameters
size_q = Intens_val['BB size'].max()
size_n = 1
size_s = 2
size_0 = 1
size_lmid = 1
tick = 0.01
nb_iter = 2000
qb_0 = 1 # 4
qa_0 = 1 # 4
s_0 = 1
pb_0 = 10
pa_0 = pb_0 + tick
size_0 = 1
period = 600
nb_period = 1
Tf = period*nb_period
size_agent = 1
size_min = size_0
size_period = 20*size_0
time_step_0 = 0.1
inventory =  int(size_period*nb_period)
Agent_state = {'time' : 0, 'end_period' : 0, 'period' : period, 'size' : size_agent, 'nb_period' : nb_period,
               'remaining_qty' : inventory, 'qty_start' : inventory, 'time_step' : time_step_0,
               'remaining_qty_period' : 0, 'qty_start_period' : 0, 
               'BB_size' : 0, 'BB_position' : 0, 
               'BA_size' : 0, 'BA_position' : 0,
               'P_exec' : 0, 'VWAP' : 0, 'VWAP_start_period' : 0,  'VWAP_price_period' : 0,
               'Volume' : 0, 'Volume_start_period' : 0}  

Lob_state_0 = [qb_0,qa_0,s_0,pb_0,pa_0] 
write_option = True # True # False
write_price = False # True # False
Agent_decision_func = T1_agent_decision
Apply_agent_decision = Apply_agent_decision_v1


##### Simulate the lob
dict_res = Lob_simu_agent(Lob_state_0,Intens_val,nb_iter, Tf = Tf, write_option = write_option, Agent_decision_func = Agent_decision_func, Agent_state = Agent_state,
                   Apply_agent_decision = Apply_agent_decision, write_price = write_price)
df_res = dict_res['History']
Lob_state = dict_res['lob']
Agent_state = dict_res['Agent_state']

##### Plot the result
plt.plot(df_res.loc[::3,'time'], df_res.loc[::3,'BB price'],'b--')
plt.plot(df_res.loc[::3,'time'], df_res.loc[::3,'BA price'],'r--')
plt.scatter(x = df_res.loc[::3,'time'], y =df_res.loc[::3,'BB price'], s = df_res.loc[::3,'BB size']*20)
plt.scatter(x = df_res.loc[::3,'time'], y =df_res.loc[::3,'BA price'], s = df_res.loc[::3,'BA size']*20)
plt.grid() 
plt.show()

###### Compute the price impact
###### Computation of the price impact through time simulation
write_option = False # True # False
NbSimu = 40 
t_step_index = 1
Mean, Var = MC_simu_pi_1(NbSimu,Lob_state_0,Intens_val,nb_iter,t_step_index = t_step_index,Tf = Tf,write_option = write_option,
                 Agent_decision_func = Agent_decision_func, Agent_state = dict(),
                 Apply_agent_decision = Apply_agent_decision)

##### Plot the result
plt.plot(Mean * 1e4, 'b--')
plt.plot((Mean  + 1.96*np.sqrt(Var/NbSimu))* 1e4, 'r--')
plt.plot((Mean - 1.96*np.sqrt(Var/NbSimu))* 1e4, 'r--')
plt.grid()
plt.show()

################################################################################################################
######## Strategy 2 : Pegging to the best : within a period
################################################################################################################

########## Initialization of the parameters
size_q = Intens_val['BB size'].max()
size_n = 1
size_s = 2
size_0 = 1
size_lmid = 1
tick = 0.1
nb_iter = 2000
qb_0 = 1 # 4
qa_0 = 1 # 4
s_0 = 1
pb_0 = 10
pa_0 = pb_0 + tick
size_0 = 1
period = 600
nb_period = 1
Tf = period*nb_period
size_agent = 1
size_min = size_0
size_period = 60*size_0
time_step_0 = 0.1
inventory =  int(size_period*nb_period)
Agent_state = {'time' : 0, 'end_period' : 0, 'period' : period, 'size' : size_agent, 'nb_period' : nb_period,
               'remaining_qty' : inventory, 'qty_start' : inventory,  'time_step' : time_step_0,
               'remaining_qty_period' : 0, 'qty_start_period' : 0, 
               'BB_size' : 0, 'BB_position' : 0, 
               'BA_size' : 0, 'BA_position' : 0,
               'P_exec' : 0, 'VWAP' : 0, 'VWAP_start_period' : 0, 'VWAP_price_period' : 0,
               'Volume' : 0, 'Volume_start_period' : 0}  

Lob_state_0 = [qb_0,qa_0,s_0,pb_0,pa_0] 
write_option = True # True # False
write_price = False # True # False
Agent_decision_func = T2_agent_decision
Apply_agent_decision = Apply_agent_decision_v1


##### Simulate the lob
dict_res = Lob_simu_agent(Lob_state_0,Intens_val,nb_iter, Tf = Tf, write_option = write_option, Agent_decision_func = Agent_decision_func, Agent_state = Agent_state,
                   Apply_agent_decision = Apply_agent_decision)
df_res = dict_res['History']
Lob_state = dict_res['lob']
Agent_state = dict_res['Agent_state']


##### Plot the result
plt.plot(df_res.loc[::3,'time'], df_res.loc[::3,'BB price'],'b--')
plt.plot(df_res.loc[::3,'time'], df_res.loc[::3,'BA price'],'r--')
plt.scatter(x = df_res.loc[::3,'time'], y =df_res.loc[::3,'BB price'], s = df_res.loc[::3,'BB size']*20)
plt.scatter(x = df_res.loc[::3,'time'], y =df_res.loc[::3,'BA price'], s = df_res.loc[::3,'BA size']*20)
plt.grid() 
plt.show()

###### Compute the price impact
###### Computation of the price impact through time simulation
write_option = False # True # False
NbSimu = 40
t_step_index = 1
Mean, Var = MC_simu_pi_1(NbSimu,Lob_state_0,Intens_val,nb_iter,t_step_index = t_step_index,Tf = Tf,write_option = write_option,
                 Agent_decision_func = Agent_decision_func, Agent_state = dict(),
                 Apply_agent_decision = Apply_agent_decision)

##### Plot the result
plt.plot(Mean * 1e4, 'b--')
plt.plot((Mean + 1.96*np.sqrt(Var/NbSimu))* 1e4, 'r--')
plt.plot((Mean - 1.96*np.sqrt(Var/NbSimu))* 1e4, 'r--')
plt.grid()
plt.show()


################################################################################################################
######## Strategy 1 : Fire and forget : linear scheduling
################################################################################################################

########## Initialization of the parameters
size_q = Intens_val['BB size'].max()
size_n = 1
size_s = 2
size_0 = 1
size_lmid = 1
tick = 0.01
nb_iter = 2000
qb_0 = 1 # 4
qa_0 = 1 # 4
s_0 = 1
pb_0 = 10
pa_0 = pb_0 + tick
size_0 = 1
period = 60
nb_period = 10
Tf = period*nb_period
size_agent = 1
size_min = size_0
size_period = 2*size_0
time_step_0 = 0.1
inventory =  int(size_period*nb_period)
Agent_state = {'time' : 0, 'end_period' : 0, 'period' : period, 'size' : size_agent, 'nb_period' : nb_period,
               'remaining_qty' : inventory, 'qty_start' : inventory, 'time_step' : time_step_0,
               'remaining_qty_period' : 0, 'qty_start_period' : 0, 
               'BB_size' : 0, 'BB_position' : 0, 
               'BA_size' : 0, 'BA_position' : 0,
               'P_exec' : 0, 'VWAP' : 0, 'VWAP_start_period' : 0,  'VWAP_price_period' : 0,
               'Volume' : 0, 'Volume_start_period' : 0}  

Lob_state_0 = [qb_0,qa_0,s_0,pb_0,pa_0] 
write_option = True # True # False
write_price = False # True # False
Agent_decision_func = T1_agent_decision
Apply_agent_decision = Apply_agent_decision_v1


##### Simulate the lob
dict_res = Lob_simu_agent(Lob_state_0,Intens_val,nb_iter, Tf = Tf, write_option = write_option, Agent_decision_func = Agent_decision_func, Agent_state = Agent_state,
                   Apply_agent_decision = Apply_agent_decision, write_price = write_price)
df_res = dict_res['History']
Lob_state = dict_res['lob']
Agent_state = dict_res['Agent_state']


##### Plot the result
plt.plot(df_res.loc[::3,'time'], df_res.loc[::3,'BB price'],'b--')
plt.plot(df_res.loc[::3,'time'], df_res.loc[::3,'BA price'],'r--')
plt.scatter(x = df_res.loc[::3,'time'], y =df_res.loc[::3,'BB price'], s = df_res.loc[::3,'BB size']*20)
plt.scatter(x = df_res.loc[::3,'time'], y =df_res.loc[::3,'BA price'], s = df_res.loc[::3,'BA size']*20)
plt.grid() 
plt.show()

###### Compute the price impact
###### Computation of the price impact through time simulation
write_option = False # True # False
NbSimu = 40 
t_step_index = 1
Mean, Var = MC_simu_pi_1(NbSimu,Lob_state_0,Intens_val,nb_iter,t_step_index = t_step_index,Tf = Tf,write_option = write_option,
                 Agent_decision_func = Agent_decision_func, Agent_state = dict(),
                 Apply_agent_decision = Apply_agent_decision)

##### Plot the result
plt.plot(Mean * 1e4, 'b--')
plt.plot((Mean  + 1.96*np.sqrt(Var/NbSimu))* 1e4, 'r--')
plt.plot((Mean - 1.96*np.sqrt(Var/NbSimu))* 1e4, 'r--')
plt.grid()
plt.show()


################################################################################################################
######## Strategy 1 : Fire and forget : exponential scheduling
################################################################################################################

########## Initialization of the parameters
size_q = Intens_val['BB size'].max()
size_n = 1 #2
size_s = 2 #2
size_0 = 1
size_lmid = 1
tick = 0.01
nb_iter = 2000
qb_0 = 1 # 4
qa_0 = 1 # 4
s_0 = 1
pb_0 = 10
pa_0 = pb_0 + tick
size_0 = 1
period = 60
nb_period = 10
Tf = period*nb_period
size_agent = 1
size_min = size_0
size_period = 2*size_0
time_step_0 = 0.1
inventory =  int(size_period*nb_period)
Agent_state = {'time' : 0, 'end_period' : 0, 'period' : period, 'size' : size_agent, 'nb_period' : nb_period,
               'remaining_qty' : inventory, 'qty_start' : inventory, 'time_step' : time_step_0,
               'remaining_qty_period' : inventory//nb_period, 'qty_start_period' : inventory//nb_period, 
               'BB_size' : 0, 'BB_position' : 0, 
               'BA_size' : 0, 'BA_position' : 0,
               'P_exec' : 0, 'VWAP' : 0, 'VWAP_start_period' : 0, 'VWAP_price_period' : 0,
               'Volume' : 0, 'Volume_start_period' : 0}  

Lob_state_0 = [qb_0,qa_0,s_0,pb_0,pa_0] 
write_option = True
Agent_decision_func = lambda x,y,z : T1_agent_decision(x,y,z, Initialize_period_linear_func = Initialize_period_expon)
Apply_agent_decision = Apply_agent_decision_v1


##### Simulate the lob
dict_res = Lob_simu_agent(Lob_state_0,Intens_val,nb_iter, Tf = Tf, write_option = write_option, Agent_decision_func = Agent_decision_func, Agent_state = Agent_state,
                   Apply_agent_decision = Apply_agent_decision)
df_res = dict_res['History']
Lob_state = dict_res['lob']
Agent_state = dict_res['Agent_state']


##### Plot the result
plt.plot(df_res.loc[::3,'time'], df_res.loc[::3,'BB price'],'b--')
plt.plot(df_res.loc[::3,'time'], df_res.loc[::3,'BA price'],'r--')
plt.scatter(x = df_res.loc[::3,'time'], y =df_res.loc[::3,'BB price'], s = df_res.loc[::3,'BB size']*20)
plt.scatter(x = df_res.loc[::3,'time'], y =df_res.loc[::3,'BA price'], s = df_res.loc[::3,'BA size']*20)
plt.grid() 
plt.show()

###### Compute the price impact
###### Computation of the price impact through time simulation
write_option = False # True # False
NbSimu = 40 
t_step_index = 1
Mean, Var = MC_simu_pi_1(NbSimu,Lob_state_0,Intens_val,nb_iter,t_step_index = t_step_index,Tf = Tf,write_option = write_option,
                 Agent_decision_func = Agent_decision_func, Agent_state = dict(),
                 Apply_agent_decision = Apply_agent_decision)

##### Plot the result
plt.plot(Mean * 1e4, 'b--')
plt.plot((Mean  + 1.96*np.sqrt(Var/NbSimu))* 1e4, 'r--')
plt.plot((Mean - 1.96*np.sqrt(Var/NbSimu))* 1e4, 'r--')
plt.grid()
plt.show()


################################################################################################################
######## Strategy 2 : Pegging to the best : linear scheduling
################################################################################################################

########## Initialization of the parameters
size_q = Intens_val['BB size'].max()
size_n = 1 #2
size_s = 2 #2
size_0 = 1
size_lmid = 1
tick = 0.01
nb_iter = 2000
qb_0 = 1 # 4
qa_0 = 1 # 4
s_0 = 1
pb_0 = 10
pa_0 = pb_0 + tick
size_0 = 1
period = 60
nb_period = 10
Tf = period*nb_period
size_agent = 1
size_min = size_0
size_period = 2*size_0
time_step_0 = 0.1
inventory =  int(size_period*nb_period)
Agent_state = {'time' : 0, 'end_period' : 0, 'period' : period, 'size' : size_agent, 'nb_period' : nb_period,
               'remaining_qty' : inventory, 'qty_start' : inventory,  'time_step' : time_step_0,
               'remaining_qty_period' : 0, 'qty_start_period' : 0, 
               'BB_size' : 0, 'BB_position' : 0, 
               'BA_size' : 0, 'BA_position' : 0,
               'P_exec' : 0, 'VWAP' : 0, 'VWAP_start_period' : 0, 'VWAP_price_period' : 0,
               'Volume' : 0, 'Volume_start_period' : 0}  

Lob_state_0 = [qb_0,qa_0,s_0,pb_0,pa_0] 
write_option = True # True # False
write_price = False # True # False
Agent_decision_func = T2_agent_decision
Apply_agent_decision = Apply_agent_decision_v1


##### Simulate the lob
dict_res = Lob_simu_agent(Lob_state_0,Intens_val,nb_iter, Tf = Tf, write_option = write_option, Agent_decision_func = Agent_decision_func, Agent_state = Agent_state,
                   Apply_agent_decision = Apply_agent_decision)
df_res = dict_res['History']
Lob_state = dict_res['lob']
Agent_state = dict_res['Agent_state']

##### Plot the result
plt.plot(df_res.loc[::3,'time'], df_res.loc[::3,'BB price'],'b--')
plt.plot(df_res.loc[::3,'time'], df_res.loc[::3,'BA price'],'r--')
plt.scatter(x = df_res.loc[::3,'time'], y =df_res.loc[::3,'BB price'], s = df_res.loc[::3,'BB size']*20)
plt.scatter(x = df_res.loc[::3,'time'], y =df_res.loc[::3,'BA price'], s = df_res.loc[::3,'BA size']*20)
plt.grid() 
plt.show()

###### Compute the price impact
###### Computation of the price impact through time simulation
write_option = False # True # False
NbSimu = 40
t_step_index = 1
Mean, Var = MC_simu_pi_1(NbSimu,Lob_state_0,Intens_val,nb_iter,t_step_index = t_step_index,Tf = Tf,write_option = write_option,
                 Agent_decision_func = Agent_decision_func, Agent_state = dict(),
                 Apply_agent_decision = Apply_agent_decision)

##### Plot the result
plt.plot(Mean * 1e4, 'b--')
plt.plot((Mean + 1.96*np.sqrt(Var/NbSimu))* 1e4, 'r--')
plt.plot((Mean - 1.96*np.sqrt(Var/NbSimu))* 1e4, 'r--')
plt.grid()
plt.show()


################################################################################################################                 
######## Strategy 2 : Pegging to the best : exponential scheduling
################################################################################################################

########## Initialization of the parameters
size_q = Intens_val['BB size'].max()
size_n = 1 #2
size_s = 2 #2
size_0 = 1
size_lmid = 1
tick = 0.01
nb_iter = 2000
qb_0 = 1 # 4
qa_0 = 1 # 4
s_0 = 1
pb_0 = 10
pa_0 = pb_0 + tick
size_0 = 1
period = 60
nb_period = 10
Tf = period*nb_period
size_agent = 1
size_min = size_0
size_period = 2*size_0
time_step_0 = 0.1
inventory =  int(size_period*nb_period)
Agent_state = {'time' : 0, 'end_period' : period, 'period' : period, 'size' : size_agent, 'nb_period' : nb_period,
               'remaining_qty' : inventory, 'qty_start' : inventory,
               'remaining_qty_period' : inventory//nb_period, 'qty_start_period' : inventory//nb_period, 
               'BB_size' : 0, 'BB_position' : 0, 
               'BA_size' : 0, 'BA_position' : 0,
               'P_exec' : 0, 'VWAP' : 0, 'VWAP_start_period' : 0, 'VWAP_price_period' : 0,
               'Volume' : 0, 'Volume_start_period' : 0}  

Lob_state_0 = [qb_0,qa_0,s_0,pb_0,pa_0] 
write_option = True
Agent_decision_func = lambda x,y,z : T2_agent_decision(x,y,z, Initialize_period_linear_func = Initialize_period_expon)
Apply_agent_decision = Apply_agent_decision_v1


##### Simulate the lob
dict_res = Lob_simu_agent(Lob_state_0,Intens_val,nb_iter, Tf = Tf, write_option = write_option, Agent_decision_func = Agent_decision_func, Agent_state = Agent_state,
                   Apply_agent_decision = Apply_agent_decision)
df_res = dict_res['History']
Lob_state = dict_res['lob']
Agent_state = dict_res['Agent_state']


##### Plot the result
plt.plot(df_res.loc[::3,'time'], df_res.loc[::3,'BB price'],'b--')
plt.plot(df_res.loc[::3,'time'], df_res.loc[::3,'BA price'],'r--')
plt.scatter(x = df_res.loc[::3,'time'], y =df_res.loc[::3,'BB price'], s = df_res.loc[::3,'BB size']*20)
plt.scatter(x = df_res.loc[::3,'time'], y =df_res.loc[::3,'BA price'], s = df_res.loc[::3,'BA size']*20)
plt.grid() 
plt.show()

###### Compute the price impact
###### Computation of the price impact through time simulation
write_option = False # True # False
NbSimu = 40
t_step_index = 1
Mean, Var = MC_simu_pi_1(NbSimu,Lob_state_0,Intens_val,nb_iter,t_step_index = t_step_index,Tf = Tf,write_option = write_option,
                 Agent_decision_func = Agent_decision_func, Agent_state = dict(),
                 Apply_agent_decision = Apply_agent_decision)

##### Plot the result
plt.plot(Mean * 1e4, 'b--')
plt.plot((Mean + 1.96*np.sqrt(Var/NbSimu))* 1e4, 'r--')
plt.plot((Mean - 1.96*np.sqrt(Var/NbSimu))* 1e4, 'r--')
plt.grid()
plt.show()