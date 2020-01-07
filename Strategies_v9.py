import pandas as pd 
import numpy as np 
import sys 
import logging 
import itertools as it
from datetime import datetime, timedelta
import time 
import os 
import sqlite3
import warnings

warnings.filterwarnings('ignore')



logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) 

formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
file_handler =logging.FileHandler('strategies.log')
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()

logger.addHandler(file_handler)
logger.addHandler(stream_handler)


class Strategies(object): 
    def __init__(self,  
                 leg=None, 
                 sdate=None, 
                 edate= None, 
                 dte=None,
                 user_cashflow=None,
                 user_range=None,
                 contract=None, 
                 delta=None, 
                 MID=None,
                 MID_STRIKE=None,
                 PC=None, 
                 strike=None, 
                 percent=None, 
                 distance=None,
                 PROFIT = None,
                 MAXLOSS = None,
                 P_DELTA_MIN =None,
                 P_DELTA_MAX = None,
                 INPUT_DAYS = None,
                 path = '/Users/jeremyjohnson/Documents/Backtest/Report.xlsx',
                 strategy=None
                ): 
        
        #Arguments....
        self.sdate= sdate
        self.leg = leg 
        self.dte = dte 
        self.PC = PC
        self.strategy = strategy
        self.edate = edate 
        self.contract = contract
        self.PROFIT = PROFIT 
        self.MAXLOSS = MAXLOSS
        self.P_DELTA_MIN = P_DELTA_MIN
        self.P_DELTA_MAX = P_DELTA_MAX
        self.INPUT_DAYS = INPUT_DAYS
        self.MID_STRIKE = MID_STRIKE
        self.path = path
        self.user_cashflow = user_cashflow
        self.user_range = user_range
        
        if strategy == 1:  #delta strategy 
            self.delta = delta
            
        elif strategy == 2: #apply distance strategy 
            self.distance = distance
            self.strike = strike
            
        elif strategy == 3: #apply percentage strategy
            self.percent = percent 
            
        elif strategy == 4: #apply absolute strategy (STRIKE & DTE)
            self.strike = strike
            
        elif strategy == 5: #apply mid_strategy(MID & DTE)
            self.MID = MID
            
        elif strategy == 6: #apply mid-strike
            self.MID_STRIKE = MID_STRIKE
 
    """
    This section are the codes for the different strategies: 
    Delta strategy: 
        - Filter on the call/put and date. 
        - Filter on the closest DTE value.  
        - Filter on the closest Delta value
        - Return the exp_dt and strike values which are passed to the get_rows function to get subsequent rows.  

    Percentage strategy: 
        - Filter on the call/put and date. 
        - Filter on the closest DTE value.  
        - Filter on the closest STRIKE value using the following function F(A, percent): (A-(percent*A)).                                   A=UNDLY_PRICE.
        - Return the exp_dt and strike values which are passed to the get_rows function to get subsequent rows.  

    Absolute strategy: 
        - Filter on the call/put and date. 
        - Filter on the closest DTE value.  
        - Filter on the closest STRIKE value using the following function F(A, percent): (A-(percent*A)).                                   A=UNDLY_PRICE.
        - Return the exp_dt and strike values which are passed to the get_rows function to get subsequent rows.  

    Mid strategy: 
        - Filter on the call/put and date. 
        - Filter on the closest DTE value.  
        - Filter on the closest MID value.
        - Return the exp_dt and strike values which are passed to the get_rows function to get subsequent rows.  

    Mid-Strike strategy: 
            
    Execute the different strategies by (conditions are checked before the run_condition function executes):
    Check if the constraints are satistified before proceeding. The constraints checked are: 
    - All parameters passed for the different strategies can be passed either as a constant or a list. In the case of a list, the       values passed must be of the same length as the input number of legs, else the system will throw an error and exit. In the         case where the value is passed as a constant, this value is repeated for the number of legs.
    
    Functions: 
        - get_rows: this function is used to lock an open position until the close condition is satisfied. 
        - cashflow_delta_condition: this function is used check if both the delta and cashflow conditions are satisfied before a             position is opened. 
        - run_conditions: this function is called when executing the positions.
        - get_parameter_delta: this function uses iter to iterate over the variables needed as input by the delta strategy.
        - get_parameter_percent: this function uses iter to iterate over the variables needed as input by the percent strategy.
        - get_parameter_distance: this function uses iter to iterate over the variables needed as input by the distance strategy.
        - get_parameter_abs: this function uses iter to iterate over the variables needed as input by the absolute strategy.
        - get_parameter_mid: this function uses iter to iterate over the variables needed as input by the mid strategy.
        - get_parameter_mid_strike: this function uses iter to iterate over the variables needed as input by the mid-strike                 strategy.
        - set_parameter_delta: this is used to get the input values from the get_parameter_delta. The same applies for the other             set_parameters for all the other strategies functions.
        
    REF: 
        #00: Manually close position if the end date is satisfied and there is an opened position.
        #01: Close position when the DTE condition is satisfied. 
        #02: Close position when the P_DELTA_MAX condition is satisfied. 
        #03: Close position when the P_DELTA_MIN condition is satisfied. 
        #04: Close position when the PROFIT condition is satisfied. 
        #05: Close position when the MAXLOSS condition is satisfied. 
        #06: Close position when the END-DATE condition is satisfied. 
        
    """
    @classmethod 
    def get_rows(cls, df, sdate, PC, contract, strikee, exp_dt):
        dt = df[(df.TRADE_DT==sdate) & (df.TYPE==PC) & (df.STRIKE == strikee) & (df.EXP_DT == exp_dt)]
        cols = dt.columns.tolist()
        cols = cols[8:9] + cols[-14:]
        dt.loc[:, cols] = dt.loc[:, cols] * (contract)
        dt.drop_duplicates(subset ="TRADE_DT", keep = 'first', inplace = True)
        return dt
    
    @classmethod
    def cashflow_delta_condition(cls, df, delta, user_range, contract):
        df['D'] = df.apply(lambda x: pd.eval((x.DELTA*contract) + sum(delta)), axis=1)
        df['C'] = df.apply(lambda x: True if (user_range[0] < x.D < user_range[1]) else False, axis=1)
        row = df[df['C']==True][:1].drop(['C', 'D'], axis=1)
        return row
    
    def run_conditions(self):
        t1 = time.perf_counter()
        if self.strategy == 1:
            if isinstance(self.delta, list)==True: 
                if len(self.delta) != self.leg:
                    sys.exit('self.delta must be a constant or have values equal the length of the self.leg specified...')
            else: 
                self.delta = [self.delta]
                
        elif self.strategy == 2:
            if isinstance(self.distance, list)==True: 
                if len(self.distance) != self.leg:
                    sys.exit('self.delta must be a constant or have values equal the length of the self.leg specified...')
            else: 
                self.distance = [self.distance]
            
            if isinstance(self.strike, list)==True: 
                if len(self.strike) != self.leg:
                    sys.exit('self.delta must be a constant or have values equal the length of the self.leg specified...')
            else: 
                self.strike = [self.strike]
                
        elif self.strategy == 3:
            if isinstance(self.percent, list)==True: 
                if len(self.percent) != self.leg:
                    sys.exit('self.delta must be a constant or have values equal the length of the self.leg specified...')
            else: 
                self.percent = [self.percent]
                
        elif self.strategy == 4:
            if isinstance(self.strike, list)==True: 
                if len(self.strike) != self.leg:
                    sys.exit('self.delta must be a constant or have values equal the length of the self.leg specified...')
            else: 
                self.strike = [self.strike]
                
        elif self.strategy == 5:
            if isinstance(self.MID, list)==True: 
                if len(self.MID) != self.leg:
                    sys.exit('self.delta must be a constant or have values equal the length of the self.leg specified...')
            else: 
                self.MID = [self.MID]
                
        elif self.strategy == 6:
            if isinstance(self.MID_STRIKE, list)==True: 
                if len(self.MID_STRIKE) != self.leg:
                    sys.exit('self.delta must be a constant or have values equal the length of the self.leg specified...')
            else: 
                self.MID_STRIKE = [self.MID_STRIKE]
                
        if isinstance(self.dte, list) == True: 
            if len(self.dte) != self.leg:
                sys.exit('self.dte must be a constant or have values equal the length of the self.leg specified...')
        else: 
            self.dte = [self.dte]

        if isinstance(self.contract, list)==True: 
            if len(self.contract) != self.leg:  
                sys.exit('self.delta must be a constant or have values equal the length of the self.leg specified...')
        else: 
            self.contract = [self.contract]
        
        """
         get_parameter_(delta, percent, distance, abs, mid) - used to iterate the contracts, dte, delta, distance etc. 
         set_parameter_(delta, percent, distance, abs, mid) - used to get the next contracts, dte etc. for the get_parameter.
        """
        
        def get_parameter_delta(): 
            dtee = it.cycle(self.dte)
            deltaa = it.cycle(self.delta)
            contractt = it.cycle(self.contract)
            return dtee, deltaa, contractt
        
        def get_parameter_percent(): 
            dtee = it.cycle(self.dte)
            percentt = it.cycle(self.percent)
            contractt = it.cycle(self.contract)
            return dtee, percentt, contractt
        
        def get_parameter_distance(): 
            dtee = it.cycle(self.dte)
            distancee = it.cycle(self.distance)
            contractt = it.cycle(self.contract)
            return dtee, distancee, contractt
        
        def get_parameter_abs(): 
            dtee = it.cycle(self.dte)
            contractt = it.cycle(self.contract)
            strikk = it.cycle(self.strike)
            return dtee, contractt, strikk
        
        def get_parameter_mid(): 
            dtee = it.cycle(self.dte)
            contractt = it.cycle(self.contract)
            midd = it.cycle(self.MID)
            return dtee, contractt, midd
        
        def get_parameter_mid_strike(): 
            dtee = it.cycle(self.dte)
            contractt = it.cycle(self.contract)
            midd_strike = it.cycle(self.MID_STRIKE)
            return dtee, contractt, midd_strike
        
        def set_parameter_delta(dtee, deltaa, contractt):
            return next(dtee), next(deltaa), next(contractt)
        
        def set_parameter_percent(dtee, percentt, contractt):
            return next(dtee), next(percentt), next(contractt)
        
        def set_parameter_distance(dtee, distancee, contractt):
            return next(dtee), next(distancee), next(contractt)
        
        def set_parameter_abs(dtee, contractt, strikk):
            return next(dtee), next(contractt), next(strikk)
        
        def set_parameter_mid(dtee, contractt, midd):
            return next(dtee), next(contractt), next(midd)
        
        def set_parameter_mid_strike(dtee, contractt, midd_strike):
            return next(dtee), next(contractt), next(midd_strike)
        
        #Insert the values for contract, dte etc. depending on the strategy selected. 
        if self.strategy == 1: 
            dtee, deltaa, contractt = get_parameter_delta()
            
        elif self.strategy == 2: 
            dtee, distancee, contractt = get_parameter_distance()
            
        elif self.strategy == 3: 
            dtee, percentt, contractt = get_parameter_percent()
            
        elif self.strategy == 4: 
            dtee, contractt, strikk = get_parameter_abs()
            
        elif self.strategy == 5: 
            dtee, contractt, midd = get_parameter_mid()
        
        elif self.strategy == 6: 
            dtee, contractt, midd_strike = get_parameter_mid_strike()
            
        
        all_trades, rlog= pd.DataFrame(), pd.DataFrame()
        open_close_position, get_params = {}, {}
        pnl_closee, cashfloww = [], []
             
        for i in range(self.leg):
            #Set the values for contract, dte etc. depending on the strategy selected. 
            if self.strategy == 1:
                dte, delta, contract = set_parameter_delta(dtee, deltaa, contractt)
                if i in get_params: 
                    get_params[i].append(dte, delta, contract) 
                else: 
                    get_params[i] = [dte, delta, contract]
            
            elif self.strategy == 2: 
                dte, distance, contract = set_parameter_distance(dtee, distancee, contractt)
                if i in get_params: 
                    get_params[i].append(dte, distance, contract) 
                else: 
                    get_params[i] = [dte, distance, contract]

            elif self.strategy == 3: 
                dte, percent, contract = set_parameter_percent(dtee, percentt, contractt)
                if i in get_params: 
                    get_params[i].append(dte, percent, contract) 
                else: 
                    get_params[i] = [dte, percent, contract]

            elif self.strategy == 4: 
                dte, contract, strikke = set_parameter_abs(dtee, contractt, strikk)
                if i in get_params: 
                    get_params[i].append(dte, contract, strikke) 
                else: 
                    get_params[i] = [dte, contract, strikke]
                    
            elif self.strategy == 5: 
                dte, contract, mid = set_parameter_mid(dtee, contractt, midd)
                if i in get_params: 
                    get_params[i].append(dte, contract, mid) 
                else: 
                    get_params[i] = [dte, contract, mid]
                    
            elif self.strategy == 6: 
                dte, contract, mid_strikee = set_parameter_mid(dtee, contractt, midd_strike)
                if i in get_params: 
                    get_params[i].append(dte, contract, mid_strikee) 
                else: 
                    get_params[i] = [dte, contract, mid_strikee]
        
        date_range = pd.date_range(self.sdate, self.edate, freq='D').tolist()
        date_range = [str(k).split(' ')[0] for k in date_range] + [self.edate]
        
        position_ind, counter, count_missing_days = 0, 0, 0
        for get_date in range(0, len(date_range), 730): 
            conn = sqlite3.connect('finall.db')
            check_date = date_range[get_date:get_date+730]
            sql_data = f"SELECT * FROM DATA where TRADE_DT between '{check_date[0]}' and '{check_date[-1]}'"
            logger.debug(f'Loading data for TRADE_DT between {check_date[0]} and {check_date[-1]} ...')
            df = pd.read_sql(sql_data, conn)
            print('Done...')
            
            sdate = datetime.strptime(check_date[0], '%Y-%m-%d')
            edate = datetime.strptime(check_date[-1], '%Y-%m-%d')
            deltas = edate - sdate
            edatee = sdate + timedelta(days=deltas.days)
            trade_days = df.TRADE_DT.unique()
            
            
            for j in range(deltas.days+1):
                trades = pd.DataFrame()
                day = str(sdate + timedelta(days=j)).split(' ')[0]
                
                if day not in trade_days: 
                    continue 

                #Loop is executed only when the position indicator is zero and a new position is to be opened. 
                if position_ind == 0: 
                    exp_dt_strikee = {}
                    pre_strike = None 
                    cal_delta = []

                    for k, v in get_params.items():
                        if self.strategy == 1:
                            dte, delta, contract = v[0], v[1], v[2] 

                        elif self.strategy == 2:
                            dte, distance, contract = v[0], v[1], v[2]

                        elif self.strategy == 3:
                            dte, percent, contract = v[0], v[1], v[2]

                        elif self.strategy == 4: 
                            dte, contract, st_strike = v[0], v[1], v[2]

                        elif self.strategy == 5: 
                            dte, contract, st_mid = v[0], v[1], v[2]

                        elif self.strategy == 6: 
                            dte, contract, st_mid = v[0], v[1], v[2]

                        #this process will be skipped in the case where the dataframe is empty. Example                                                   is during weekends where no trade values are recorded.
                        try:
                            dt = df[(df.TRADE_DT == day) & (df.TYPE==self.PC)]
                            dtee = dt.DTE.iloc[(dt.DTE-dte).abs().argsort()][:1].values[0]
                            dt = dt[(dt.DTE == dtee)] 
                        except Exception as e:
                            logger.error(e) 

                        if self.strategy == 1:
                            if k != self.leg-1: 
                                dt = dt.iloc[(dt.DELTA-delta).abs().argsort()][:1] 
                            else:
                                dt = dt.iloc[(dt.DELTA-delta).abs().argsort()]
                                dt = self.cashflow_delta_condition(dt, cal_delta, self.user_range, contract)
                            try:
                                exp_dt, strikee, mid_value = dt.EXP_DT.values[0], dt.STRIKE.values[0], dt.MID.values[0]
                                cal_delta.append(dt.DELTA.values[0]*contract)
                            except: 
                                break  

                        elif self.strategy == 2:
                            if k != self.leg-1:
                                val = (dt.UNDLY_PRICE + distance)[:1].values[0]
                                dt = dt.iloc[(dt.STRIKE - val).abs().argsort()][:1]
                            else:
                                val = (dt.UNDLY_PRICE + distance)[:1].values[0]
                                dt = dt.iloc[(dt.STRIKE - val).abs().argsort()]
                                dt = self.cashflow_delta_condition(dt, cal_delta, self.user_range)
                            try: 
                                exp_dt, strikee, mid_value = dt.EXP_DT.values[0], dt.STRIKE.values[0], dt.MID.values[0]
                                cal_delta.append(dt.DELTA.values[0]*contract)
                            except: 
                                break

                        elif self.strategy == 3:
                            if k != self.leg-1:
                                val = (dt.UNDLY_PRICE + (dt.UNDLY_PRICE * percent))[:1].values[0]
                                dt = dt.iloc[(dt.STRIKE - val).abs().argsort()][:1]
                            else: 
                                val = (dt.UNDLY_PRICE + (dt.UNDLY_PRICE * percent))[:1].values[0]
                                dt = dt.iloc[(dt.STRIKE - val).abs().argsort()]
                                dt = self.cashflow_delta_condition(dt, cal_delta, self.user_range, contract)
                            try:
                                exp_dt, strikee, mid_value = dt.EXP_DT.values[0], dt.STRIKE.values[0], dt.MID.values[0] 
                                cal_delta.append(dt.DELTA.values[0]*contract)
                            except: 
                                break
                            
                        elif self.strategy == 4:
                            if k != self.leg-1:
                                val = (dt.UNDLY_PRICE + st_strike)[:1].values[0]
                                dt = dt.iloc[(dt.STRIKE - val).abs().argsort()][:1] 
                            else: 
                                val = (dt.UNDLY_PRICE + st_strike)[:1].values[0]
                                dt = dt.iloc[(dt.STRIKE - val).abs().argsort()]
                                dt = self.cashflow_delta_condition(dt, cal_delta, self.user_range, contract)
                            try: 
                                exp_dt, strikee, mid_value = dt.EXP_DT.values[0], dt.STRIKE.values[0], dt.MID.values[0]
                                cal_delta.append(dt.DELTA.values[0]*contract)
                            except: 
                                break

                        elif self.strategy == 5:
                            if k != self.leg-1: 
                                dt = dt.iloc[(dt.MID-st_mid).abs().argsort()][:1]
                            else: 
                                dt = dt.iloc[(dt.MID-st_mid).abs().argsort()]
                                dt = self.cashflow_delta_condition(dt, cal_delta, self.user_range, contract)
                            try:
                                exp_dt, strikee, mid_value = dt.EXP_DT.values[0], dt.STRIKE.values[0], dt.MID.values[0]
                                cal_delta.append(dt.DELTA.values[0]*contract)
                            except: 
                                break

                        elif self.strategy == 6:
                            if k != self.leg-1:
                                dt = dt.iloc[(dt.MID-st_mid).abs().argsort()][:1] 
                            else: 
                                val = pre_strike + st_mid
                                dt = dt.iloc[(dt.STRIKE-val).abs().argsort()][:1]
                                dt = self.cashflow_delta_condition(dt, cal_delta, self.user_range, contract)
                            try: 
                                exp_dt, strikee, mid_value = dt.EXP_DT.values[0], dt.STRIKE.values[0], dt.MID.values[0]
                                cal_delta.append(dt.DELTA.values[0]*contract)
                                pre_strike = strikee
                            except: 
                                break
                        
                        if k in exp_dt_strikee: 
                            exp_dt_strikee[k].append(exp_dt, strikee, contract)
                        else: 
                            exp_dt_strikee[k] = [exp_dt, strikee, contract]
                            
                contratt = []
                for k, v in exp_dt_strikee.items():
                    exp_dt, strikee, contract  = v[0], v[1], v[2]
                    contratt.append(v[2])
                    #Get the required rows using the parameters exp_dt, strikee, contract etc. 
                    current_df = self.get_rows(df, day, self.PC, contract, strikee, exp_dt)
                    trades = trades.append(current_df)
                    
                if trades.shape[0] != self.leg: 
                    if count_missing_days != 3:
                        count_missing_days += 1
                        continue 
                    else:
                        print(f'Manually closing Trade at {day}...')
                        get_last_rows = rlog.iloc[-1].copy()
                        get_last_rows['O&P'] = 'Close Trade'
                        get_last_rows["P&L Close"] = pnl_closee[-1]
                        get_last_rows["P&L"] = pnl_closee[-1]
                        get_last_rows.CASHFLOW = cashfloww[-1]
                        get_last_rows.REF = "#00"
                        rlog=rlog.append(get_last_rows, ignore_index=True)

                        get_last_rows = all_trades.iloc[-self.leg:].copy()
                        get_last_rows.QUANTITY.iloc[-self.leg:] = -get_last_rows.QUANTITY.iloc[-self.leg:]
                        get_last_rows['O&P'] = 'Close Trade'
                        all_trades = all_trades.append(get_last_rows, ignore_index=True)
                        count_missing_days = 0
                        position_ind = 0
                else: 
                    count_missing_days = 0
                    cols = trades.columns.tolist()
                    cols = cols[8:9]+ cols[16:]
                    current_df = trades[cols]
                    trades.reset_index(inplace=True, drop=True) 
                    new_cashflow = current_df.MID.sum()
                    current_mid = current_df.MID.sum()

                    #Check the position conditions and set them to open or close based on the satisfied conditions. 
                    if position_ind == 0:
                        print(f'Opening New Trade at {day}...')
                        position_ind = 1
                        previous_df = trades[cols]
                        old_cashflow = -new_cashflow
                        mid_open = current_mid

                        cashflow = old_cashflow
                        contracts = [x for x in contratt]
                        pnl = 0
                        pnl_close = ""
                        open_close = "Open Trade"
                        start_day = datetime.strptime(trades.TRADE_DT.iloc[0], '%Y-%m-%d')
                        dit = (datetime.strptime(trades.TRADE_DT.iloc[0], '%Y-%m-%d') - start_day).days
                        REF = ''

                    elif ((position_ind==1) & (trades.DTE.iloc[0] <= self.INPUT_DAYS)):
                        position_ind = 0
                        cashflow = new_cashflow + old_cashflow
                        contracts = [-x for x in contratt]
                        pnl = current_mid - mid_open
                        pnl_close = current_mid - mid_open
                        open_close = "Close Trade"
                        dit = (datetime.strptime(trades.TRADE_DT.iloc[0], '%Y-%m-%d') - start_day).days
                        REF = '#01'
                        print(f'Closing Trade at {day}...')

                    elif ((position_ind == 1) & (current_df.DELTA.sum() - previous_df.DELTA.sum() >= self.P_DELTA_MAX)): 
                        position_ind = 0
                        cashflow = new_cashflow + old_cashflow
                        contracts = -contract
                        pnl = current_mid - mid_open
                        pnl_close = current_mid - mid_open
                        open_close = "Close Trade"
                        dit = (datetime.strptime(trades.TRADE_DT.iloc[0], '%Y-%m-%d') - start_day).days
                        REF = '#02'
                        print(f'Closing Trade at {day}...')

                    elif ((position_ind == 1) & (current_df.DELTA.sum() - previous_df.DELTA.sum() <= self.P_DELTA_MIN)):
                        position_ind = 0
                        cashflow = new_cashflow + old_cashflow
                        contracts = [ -x for x in contratt]
                        pnl = current_mid - mid_open
                        pnl_close = current_mid - mid_open
                        open_close = "Close Trade"
                        dit = (datetime.strptime(trades.TRADE_DT.iloc[0], '%Y-%m-%d') - start_day).days
                        REF = '#03'
                        print(f'Closing Trade at {day}...')

                    elif ((position_ind == 1) & (current_df.MID.sum() - previous_df.MID.sum() >= self.PROFIT)): 
                        position_ind = 0
                        cashflow = new_cashflow + old_cashflow
                        contracts = [-x for x in contratt]
                        pnl = current_mid - mid_open
                        pnl_close = current_mid - mid_open
                        open_close = "Close Trade"
                        dit = (datetime.strptime(trades.TRADE_DT.iloc[0], '%Y-%m-%d') - start_day).days
                        REF = '#04'
                        print(f'Closing Trade at {day}...')

                    elif ((position_ind == 1) & (current_df.MID.sum() - previous_df.MID.sum() <= self.MAXLOSS)): 
                        position_ind =  0
                        cashflow = new_cashflow + old_cashflow
                        contracts = [-x for x in contratt]
                        pnl = current_mid - mid_open
                        pnl_close = current_mid - mid_open
                        open_close = "Close Trade"
                        dit = (datetime.strptime(trades.TRADE_DT.iloc[0], '%Y-%m-%d') - start_day).days
                        REF = '#05'
                        print(f'Closing Trade at {day}...')

                    elif ((position_ind == 1) & (self.edate == day)):
                        position_ind = 0
                        cashflow = new_cashflow + old_cashflow
                        contracts = [ -x for x in contratt]
                        pnl = current_mid - mid_open
                        pnl_close = current_mid - mid_open
                        open_close = "Close Trade"
                        dit = (datetime.strptime(trades.TRADE_DT.iloc[0], '%Y-%m-%d') - start_day).days
                        REF = '#06'
                        print(f'Closing Trade at {day}...')

                    else:
                        pnl = current_mid - mid_open
                        pnl_close = ''
                        contracts = [x for x in contratt]
                        cashflow = new_cashflow + old_cashflow
                        open_close = " "
                        dit = (datetime.strptime(trades.TRADE_DT.iloc[0], '%Y-%m-%d') - start_day).days
                        REF = ''

                    pnl_closee.append(current_mid - mid_open)
                    cashfloww.append(new_cashflow + old_cashflow)
                    pos_df = pd.DataFrame(0, index=np.arange(len(trades)), columns=['POSITION', 'QUANTITY','DIT','CASHFLOW', 
                                                                                     "P&L", "P&L Close", 'O&P', 'REF'])
                    pos_df.POSITION = position_ind
                    pos_df.QUANTITY = contracts
                    pos_df.CASHFLOW = cashflow
                    pos_df["P&L"] = pnl
                    pos_df["P&L Close"] = pnl_close
                    pos_df['O&P'] = open_close
                    pos_df.DIT = dit 
                    pos_df.REF = REF

                    pcols = pos_df.columns.tolist()[2:] #position columns (DIT','CASHFLOW', "P&L", "P&L Close", 'O&P', 'REF')
                    td_udp = trades.columns.tolist()
                    td_udp = td_udp[:1] + td_udp[3:4] + td_udp[9:10] #Trade date, DTE and MID values 
                    cols = trades.columns.tolist()[16:]
                    result_log = pd.DataFrame(trades[cols].sum()).T #Sum all trades in a leg and transpose 
                    result_log = pd.concat([trades[td_udp][:1], pos_df[pcols][:1], result_log], axis=1) #append trade to pos_columns 
                    rlog = rlog.append(result_log) #used to append all trades needed to build results log.
                    trades = pd.concat([pos_df, trades], axis=1) #used to append all trades needed to build full trade log. 
                    all_trades = all_trades.append(trades)

        all_trades.reset_index(inplace=True, drop=True)

        #condition used to manually close an open position after the end of the whole cycle.
        if all_trades['O&P'].iloc[all_trades.index[-1]] == "Open Trade":
            print(f'Manually closing Trade...')
            get_last_rows = rlog.iloc[-1].copy()
            get_last_rows['O&P'] = 'Close Trade'
            get_last_rows["P&L Close"] = pnl_closee[-1]
            get_last_rows["P&L"] = pnl_closee[-1]
            get_last_rows.CASHFLOW = cashfloww[-1]
            get_last_rows.REF = "#00"
            rlog=rlog.append(get_last_rows, ignore_index=True)

            get_last_rows = all_trades.iloc[-self.leg:].copy()
            get_last_rows.QUANTITY.iloc[-self.leg:] = -get_last_rows.QUANTITY.iloc[-self.leg:]
            get_last_rows['O&P'] = 'Close Trade'
            all_trades = all_trades.append(get_last_rows, ignore_index=True)

        #this is used to select the columns need for building the result log. The columns are rearrage into the format required.
        coll =  ['PERIOD'] + rlog.columns.tolist() 
        Period = rlog.TRADE_DT.copy()
        Period = pd.to_datetime(Period, yearfirst=True).dt.strftime('%Y%m')
        rlog = pd.concat([Period, rlog], axis=1)
        rlog.columns = coll
        cols = rlog.columns.tolist()
        cols = cols[1:2] + cols[0:1] + cols[8:9] + cols[2:3] + cols[4:5] + cols[3:4] + cols[5:8] + cols[9:]
        rlog = rlog[cols]
        rlog.rename(columns={'O&P': 'RULE', 'TRADE_DT':'DATE', 'UNDLY_PRICE': 'INDEX'}, inplace=True)
        rlog.reset_index(inplace=True, drop=True)

        full_log_trade = all_trades[all_trades['O&P'] != " "] #this is used to select only rows with open and close conditions. 
        cols = all_trades.columns.tolist()
        cols = cols[8:9] + cols[15:16] + cols[1:2] + cols[16:17] + cols[6:7]
        full_log_trade = full_log_trade[cols]
        full_log_trade.rename(columns={'TRADE_DT':'Trade Date', 'OPTION_REF': 'Option Reference', 'MID': "Price", 'O&P': 
                                       "Comment"}, inplace=True)

        #Used to get the name of the strategy being executed and passed to the excel writer and written to excel. 
        if self.strategy == 1: 
            sheet1 = 'RESULT_LOG_DELTA'
            sheet2 = 'FULL_TRADE_LOG_DELTA'

        elif self.strategy == 2: 
            sheet1 = 'RESULT_LOG_DISTANCE'
            sheet2 = 'FULL_TRADE_LOG_DISTANCE'

        elif self.strategy == 3: 
            sheet1 = 'RESULT_LOG_PERCENTAGE'
            sheet2 = 'FULL_TRADE_LOG_PERCENTAGE'

        elif self.strategy == 4: 
            sheet1 = 'RESULT_LOG_ABSOLUTE'
            sheet2 = 'FULL_TRADE_LOG_ABSOLUTE'

        elif self.strategy == 5: 
            sheet1 = 'RESULT_LOG_MID'
            sheet2 = 'FULL_TRADE_LOG_MID'

        elif self.strategy == 6: 
            sheet1 = 'RESULT_LOG_MID_STRIKE'
            sheet2 = 'FULL_TRADE_LOG_MID_STRIKE'

        #add condition to check if file exists. (mode=a)
        if os.path.isfile(self.path) == True: 
            mode = 'a'
        else: 
            mode = 'w'
            
        with pd.ExcelWriter(self.path, mode=mode) as writer:  # doctest: +SKIP
            rlog.to_excel(writer, sheet_name=sheet1, index=False)
            full_log_trade.to_excel(writer, sheet_name=sheet2, index=False)

        t2 = time.perf_counter()
        logger.debug(f'Finished executing all trades {t2-t1} seconds...')


