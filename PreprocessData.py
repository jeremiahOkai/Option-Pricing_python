import numpy as np 
import pandas as pd 
import time 
import swifter
import logging
import itertools as it
import psutil
import os 
import concurrent.futures as cf 
import pickle 
from datetime import datetime 
from Strategies_v9 import Strategies
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

file_handler = logging.FileHandler('PreprocessData.log')
# file_handler.setLevel(logging.ERROR)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()

logger.addHandler(stream_handler)
logger.addHandler(file_handler)


class Preprocess(object):
    data = os.path.join('./' 'input_files/data.csv')
    DIVRATE = os.path.join('./' 'input_files/DIVRATE.csv')
    RFRATE = os.path.join('./' 'input_files/RFRATE.csv')
    
    def __init__(self,
                 chunksize = 20000,
                 output_file = './ProcessFiles/output.csv'): 
      
        self.output_file = output_file
        self.chunksize = chunksize
    
    @classmethod
    def data_to_pickle(cls):
        logger.debug('Loading data ...')
        t1 = time.perf_counter() 
        counter =0 
        for data in pd.read_csv(Preprocess.data, chunksize=self.chunksize, dtype=object):
            print(data.shape)
            data.dropna(inplace=True)
            data = data[data.MID != 0]
            data['TRADE_DT'] = pd.to_datetime(data.TRADE_DT.astype(str))
            data['EXP_DT'] = pd.to_datetime(data['EXP_DT'].astype(str))
            data['DTE'] = (data.EXP_DT - data.TRADE_DT).dt.days
            data = data.assign(OneM = np.nan, TwelveM = np.nan, Y_values = np.nan, POS_SIZE=1, Multiplier=100)
            data.to_pickle('./ProcessFiles/data_'+str(counter)+'.pkl')
            counter +=1
            
        t2 = time.perf_counter()
        logger.debug(f'Successfully Preprocessed datafiles at {t2-t1} secs...')
            
    @classmethod
    def process_files(cls, files, output_file):
        t1 = time.perf_counter()
        rate = pd.read_csv(Preprocess.RFRATE)
        div = pd.read_csv(Preprocess.DIVRATE)
        for file in files:
            with open(file, 'rb') as f: 
                data = pickle.load(f)
            rate['DR_Weekday'] = pd.to_datetime(rate.Date).dt.dayofweek.replace({0:'Mo', 1:'Tu', 2:'We', 3:'Th', 4:'Fr', 5:'Sa', 6:'Su'})
            rate['Date'] = pd.to_datetime(rate.Date)
            div['Date'] = pd.to_datetime(div.Date)
            div['Year'] = pd.Series(pd.to_datetime(div.Date)).dt.year
            div['Year'] = pd.to_numeric(div['Year'], errors='coerce')
            div = div.dropna(subset=['Year'])
            div['Year'] = div['Year'].astype(int)

            data = data.assign(OneM = np.nan, TwelveM = np.nan, Y_values = np.nan, POS_SIZE=1, Multiplier=100)

            for date in data.TRADE_DT.unique():
                try:
                    rate['1M'][rate['Date'] == date].values[0]
                    rate['12M'][rate['Date'] == date].values[0]
                except Exception as e:
                    pass
                else: 
                    data.loc[data.TRADE_DT == date, 'OneM'] = rate['1M'][rate['Date'] == date].values[0]
                    data.loc[data.TRADE_DT == date, 'TwelveM'] = rate['12M'][rate['Date'] == date].values[0]

            for year in data.TRADE_DT.dt.year.unique():
                try:
                    div['Value'][div['Year'] == year].values[0]
                except Exception as e: 
                    pass
                else: 
                    data.loc[data.TRADE_DT.dt.year == year, 'Y_values'] =div['Value'][div['Year'] == year].values[0]

            data.dropna(inplace=True)

            try:
                data[['OneM', 'DTE', 'TwelveM', 'Y_values', 'UNDLY_PRICE']] = data[['OneM', 'DTE', 'TwelveM', 'Y_values',                              'UNDLY_PRICE']].astype(float)
            except Exception as e: 
                logger.debug(e)

            try: 
                data['RF_RATE'] =  (data.OneM+(((data.DTE-30)*(data.TwelveM-data.OneM))/(360-30)))/100
                data['DIV_RATE'] = ((1 + (data.Y_values/data.UNDLY_PRICE))**(data.DTE/365)-1)
            except Exception as e: 
                    logger.debug(e)
            data.drop(['OneM', 'TwelveM', 'Y_values'], inplace=True, axis=1)
            dates = data.TRADE_DT.unique()
        
            for date in dates:
                dt = data.loc[data.TRADE_DT == date]
                
                for d in dt.DTE.unique():
                    c = dt[(dt.DTE == d) & (dt.TYPE =='C')]
                    if c.shape[0] > 40:
                        with open(output_file, 'a') as f:
                            c.to_csv(f, header=f.tell()==0, index=False)

                    p = dt[(dt.DTE == d) & (dt.TYPE =='P')]
                    if p.shape[0] > 40:
                        with open(output_file, 'a') as f:
                            p.to_csv(f, header=f.tell()==0, index=False)
        for file in files: 
            os.remove(file)
        t2 = time.perf_counter()
        logger.debug(f'Successfully Preprocessed datafiles at {t2-t1} secs...') 
    
    
    @classmethod    
    def run_calculators(cls, filename='./ProcessFiles/output.csv'):
        try:  
            # pl.initialize()
            logger.debug('Reading data...')
            logger.debug('Calculating Greeks...')
            t0 = time.perf_counter()
            
            db = Database()
            counter = 0
            df=pd.read_csv(filename)
            print(f'calculating sigma...')
            df['IV'] = df.parallel_apply(Calculators.newton_vol_call_put_div, axis=1)
            print(f'calculating FAIRVALUE...')
            df['FAIRVALUE'] = df.parallel_apply(Calculators.euro_vanilla_dividend, axis=1)
            print(f'calculating DELTA...')
            df['DELTA'] = df.parallel_apply(Calculators.delta_div, axis=1)
            print(f'calculating GAMMA...')
            df['GAMMA'] = df.parallel_apply(Calculators.gamma_div, axis=1)
            print(f'calculating THETA...')
            df['THETA'] = df.parallel_apply(Calculators.theta_div, axis=1)
            print(f'calculating VEGA...')
            df['VEGA'] = df.parallel_apply(Calculators.vega_div, axis=1)
            print(f'calculating WTVEGA...')
            df['WTVEGA'] = df.parallel_apply(Calculators.wtvega, axis=1)
            print(f'calculating ZOMMA...')
            df['ZOMMA'] = df.parallel_apply(Calculators.zomma, axis=1)
            print(f'calculating SPEED...')
            df['SPEED'] = df.parallel_apply(Calculators.speed, axis=1)
            print(f'calculating DUAL_DELTA...')
            df['DUAL_DELTA'] = df.parallel_apply(Calculators.dual_delta, axis=1)
            print(f'calculating DECAY_RATE...')
            df['DECAY_RATE'] =df.parallel_apply(Calculators.decay_rate, axis=1)
            print(f'calculating VOMMA...')
            df['VOMMA'] =df.parallel_apply(Calculators.vomma, axis=1)
            print(f'calculating VANNA...')
            df['VANNA'] = df.parallel_apply(Calculators.vanna, axis=1)
            print(f'calculating CHARM...')
            df['CHARM'] =df.parallel_apply(Calculators.charm, axis=1)
            print(f'calculating COLOR...')
            df['COLOR'] =df.parallel_apply(Calculators.color, axis= 1)
            db.create_tables(df)

            counter +=1
            t1 = time.perf_counter()
            logger.debug(f'Finished executing all tasks {(t1 - t0)/3600} hour(s)...')

        except Exception as e: 
            logger.exception(e)
 

    def execute_process(self): 
        print(
        """
        ########################################################################################################

            Welcome to the Option Pricing Code...
            Follow the instructions below to use the code successfully. 

            1. Enter 1 to run an Option Strategy.
            2. Enter 2 to execute Preprocess - add new option data, calculate greeks and write to the database. 

        ########################################################################################################
        """)
        try: 
            value = input('Enter a value, either 1 or 2 to continue... ')
            if not value.strip():
                raise ValueError('Field cannot be empty')
            try:
                value = int(float(value))
            except TypeError as e: 
                logger.debug('Integer or float values only')
                sys.exit('Exiting program...')
        except ValueError as e:
            logger.debug(e)
            sys.exit('Exiting program...')

        if value == 1: 
            print(
            """
            ############################################################################################
                                    1. Choose 1 for Delta Strategy. 
                                    2. Choose 2 for Distance Strategy. 
                                    3. Choose 3 for Percentage Strategy. 
                                    4. Choose 4 for Absolute Strategy. 
                                    5. Choose 5 for Mid Strategy 
                                    6. Choose 6 for Mid-Strike Strategy.
            ############################################################################################
            """
            )
            try: 
                strategy = input('Choose an option above to execute strategy... ')
                if not strategy.strip():
                    raise ValueError('Field cannot be empty')
                try: 
                    strategy = int(float(strategy))
                except TypeError as e: 
                    logger.debug('Integer or float values only')
                    sys.exit('Exiting program...')
            except ValueError as e:
                logger.debug(e)
                sys.exit('Exiting program...')

            print(
                '''
            #############################################################################################
                    Enter all values separated by commas with no spaces between them. Eg. 1,2,3
                    Starting and ending dates must be of format = 2010-10-01 (yyyy-mm-dd)
                    All values must be provided else the program will raise an error and exit. 
            #############################################################################################
                '''
                )

            if strategy == 1: 
                print(
                '''
            $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                                        Execting the Delta Strategy 
            $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                '''
                )

                try:
                    delta=input('Enter delta values: ')
                    if not delta.strip():
                        raise ValueError('Field cannot be empty')
                except ValueError as e:
                    logger.debug(e)
                    sys.exit('Exiting program...')
                else: 
                    delta = [int(i) for i in delta.split(',')] 
                    if len(delta) == 1: 
                        delta = delta[0]

                MID = None 
                MID_STRIKE = None 
                strike = None 
                distance = None 
                percent = None 

            elif strategy == 2: 
                print(
                '''
            $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                                        Execting the Distance Strategy 
            $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                ''')
            elif strategy == 3: 
                print(
                '''
            $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                                        Execting the Percentage Strategy 
            $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                ''') 

                try:
                    percent=input('Enter percent values: ')
                    if not percent.strip():
                        raise ValueError('Field cannot be empty')
                except ValueError as e:
                    logger.debug(e)
                    sys.exit('Exiting program...')
                else: 
                    percent = [int(i) for i in percent.split(',')] 
                    if len(percent) == 1: 
                        percent = percent[0]
                MID = None 
                MID_STRIKE = None 
                strike = None 
                distance = None 
                delta = None 


            elif strategy == 4: 
                print(
                '''
            $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                                        Execting the Absolute Strategy 
            $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                ''') 

                try:
                    strike=input('Enter strike values: ')
                    if not strike.strip():
                        raise ValueError('Field cannot be empty')
                except ValueError as e:
                    logger.debug(e)
                    sys.exit('Exiting program...')
                else: 
                    strike = [int(i) for i in strike.split(',')] 
                    if len(strike) == 1: 
                        strike = strike[0]

                MID = None 
                MID_STRIKE = None 
                distance = None 
                delta = None 
                percent = None 

            elif strategy == 5: 
                print(
                '''
            $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                                        Execting the MID Strategy 
            $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                ''') 

                try:
                    MID=input('Enter MID values: ')
                    if not MID.strip():
                        raise ValueError('Field cannot be empty')
                except ValueError as e:
                    logger.debug(e)
                    sys.exit('Exiting program...')
                else: 
                    MID = [int(i) for i in MID.split(',')] 
                    if len(MID) == 1: 
                        MID = MID[0]
                MID_STRIKE = None 
                strike = None 
                distance = None 
                delta = None 
                percent = None 

            elif strategy == 6: 
                print(
                '''
            $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                                        Execting the MID-STRIKE Strategy 
            $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                ''') 

                try:
                    MID_STRIKE=input('Enter MID_STRIKE values: ')
                    if not MID_STRIKE.strip():
                        raise ValueError('Field cannot be empty')
                except ValueError as e:
                    logger.debug(e)
                    sys.exit('Exiting program...')
                else: 
                    MID_STRIKE = [int(i) for i in MID_STRIKE.split(',')]
                    if len(MID_STRIKE) == 1: 
                        MID_STRIKE = MID_STRIKE[0]
                strike = None 
                distance = None 
                delta = None 
                MID = None 
                percent = None 

            try:
                leg = input('Enter number of legs: ')
                if not leg.strip():
                    raise ValueError('Field cannot be empty')
                try:
                    leg=int(float(leg))
                except ValueError as e: 
                    logger.debug('Value must be integer or float')
                    sys.exit('Exiting program...')
            except ValueError as e:
                print(e)
                sys.exit('Exiting program...')

            try:
                dte = input('Enter DTE: ')
                if not dte.strip():
                    raise ValueError('Field cannot be empty')
                try:
                    dte=int(float(dte))
                except ValueError as e: 
                    print('Value must be integer or float')
                    sys.exit('Exiting program...')
            except ValueError as e:
                logger.debug(e)
                sys.exit('Exiting program...')
                

            try:
                INPUT_DAYS = input('Enter INPUT_DAYS: ')
                if not INPUT_DAYS.strip():
                    raise ValueError('Field cannot be empty')
                try:
                    INPUT_DAYS=int(float(INPUT_DAYS))
                except ValueError as e: 
                    logger.debug('Value must be integer or float')
                    sys.exit('Exiting program...')
            except ValueError as e:
                logger.debug(e)
                sys.exit('Exiting program...')

            try:
                sdate= input('Enter start date: ')
                edate= input('Enter end date: ')
                if not sdate.strip() or not edate.strip():
                    raise ValueError('Field cannot be empty')
                try:
                    sd=datetime.strptime(sdate, '%Y-%m-%d')
                    ed=datetime.strptime(edate, '%Y-%m-%d')
                except ValueError as e: 
                    logger.debug("Incorrect data format, should be YYYY-MM-DD")
                    sys.exit('Exiting program...')
            except ValueError as e:
                logger.debug(e)
                sys.exit('Exiting program...')
                
                
            try:
                PC= input('Enter PC: ')
                if not PC.strip():
                    raise ValueError('Field cannot be empty')
            except ValueError as e: 
                logger.debug(e)
                sys.exit('Exiting program...')
            else: 
                PC=PC.upper()

            try:
                user_range=input('Enter delta range to check before opening position: ')
                contract=input('Enter contracts: ')
                if not user_range.strip() or not contract.strip():
                    raise ValueError('Field cannot be empty')
            except ValueError as e:
                logger.debug(e)
                sys.exit('Exiting program...')
            else: 
                user_range = [float(i) for i in user_range.split(',')] 
                contract = [int(i) for i in contract.split(',')] 
                if len(contract) == 1: 
                    contract = contract[0]

            try:
                P_DELTA_MAX=input('Enter P_DELTA_MAX: ')
                if not P_DELTA_MAX.strip():
                    raise ValueError('Field cannot be empty')
                P_DELTA_MAX=int(float(P_DELTA_MAX))
                if int(P_DELTA_MAX) < 0: 
                    raise ValueError ('Field must greater than zero. Eg. 9999')
            except ValueError as e:
                logger.debug(e)
                sys.exit('Exiting program...')

            try:
                P_DELTA_MIN=input('Enter P_DELTA_MIN: ')
                if not P_DELTA_MIN.strip():
                    raise ValueError('Field cannot be empty')
                P_DELTA_MIN=int(float(P_DELTA_MIN))
                if int(P_DELTA_MIN) > 0: 
                    raise ValueError ('Field must greater than zero. Eg. -9999')
            except ValueError as e:
                logger.debug(e)
                sys.exit('Exiting program...')

            try:
                PROFIT=input('Enter PROFIT: ')
                MAXLOSS=input('Enter MAXLOSS: ')
                if not MAXLOSS.strip() or not PROFIT.strip():
                    raise ValueError('Field cannot be empty')
                try:
                    PROFIT=int(float(PROFIT))
                    MAXLOSS=int(float(MAXLOSS))
                except ValueError as e: 
                    logger.debug('Value must be integer or float') 
                    sys.exit('Exiting program...')
            except ValueError as e:
                logger.debug(e)
                sys.exit('Exiting program...')

            try: 
                confirm = input('Enter 1 to proceed...\nEnter 2 to exit program... \n')
                if not confirm.strip(): 
                    raise ValueError('Field cannot be empty')
                try: 
                    confirm = int(float(confirm))
                except TypeError as e: 
                    logger.debug('Integer or float values only')
            except ValueError as e: 
                logger.debug(e)

            if confirm == 1:
                st = Strategies(
                leg=leg, 
                sdate=sdate, 
                edate=edate, 
                dte=dte,
                user_cashflow=None,
                user_range=user_range,
                contract=contract, 
                delta=delta, 
                MID=MID,
                MID_STRIKE=MID_STRIKE,
                PC=PC, 
                strike=strike, 
                percent=percent, 
                distance=distance,
                PROFIT = PROFIT,
                MAXLOSS = MAXLOSS,
                P_DELTA_MIN = P_DELTA_MIN,
                P_DELTA_MAX = P_DELTA_MAX,
                INPUT_DAYS = INPUT_DAYS,
                strategy=strategy
                )
                st.run_conditions()
            else:
                sys.exit('Program exited successfully...')

        elif value == 2: 
            self.data_to_pickle()
            files = glob.glob('./ProcessFiles/*.pkl')
            self.process_files(files, self.output_file)
            self.run_calculators(aa)
            
            