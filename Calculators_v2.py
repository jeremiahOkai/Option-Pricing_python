import numpy as np
import pandas as pd 
import scipy.stats as si
import time
import logging 
import warnings


warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__) 
logger.setLevel(logging.DEBUG) 

formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

file_handler = logging.FileHandler('Calculators.log') 
file_handler.setLevel(logging.ERROR) 
file_handler.setFormatter(formatter) 

# stream_handler = logging.StreamHandler()

logger.addHandler(file_handler)
# logger.addHandler(stream_handler)


class Calculators (object): 
    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: interest rate
    #sigma: volatility of underlying asset
    #q: continuous dividend rate
    
    def __init__(): 
        pass   
            
            
    """
    DELTA, THETA, VOMMA, VANNA, CHARM, RHO, VEGA

    The next function can be called with 'call' or 'put' 
    for the option parameter to calculate the desired BS option price for non-dividend

    """
    def delta_div(df):
        try:
            if df.TYPE == 'C':
                delta = (si.norm.cdf((((np.log(df.UNDLY_PRICE / df.STRIKE)) + (((df.RF_RATE -df.DIV_RATE) + (df.IV ** 2) / 2) * (df.DTE / 365))) / (df.IV * np.sqrt((df.DTE / 365)))),0.0,1.0)) *(df.POS_SIZE * df.Multiplier)

            if df.TYPE == 'P':
                delta = ((si.norm.cdf((((np.log(df.UNDLY_PRICE / df.STRIKE))+(((df.RF_RATE - df.DIV_RATE) + (df.IV ** 2) / 2) * (df.DTE / 365)))/(df.IV * np.sqrt((df.DTE / 365)))),0.0,1.0)) -1) *(df.POS_SIZE*df.Multiplier)
        except Exception as e: 
            logger.error(e)
        else:
            return delta
    
    def vega_div(df):
        try:
            vega = ((df.UNDLY_PRICE * (si.norm.pdf((((np.log(df.UNDLY_PRICE / df.STRIKE)) + (((df.RF_RATE-df.DIV_RATE) + (df.IV ** 2) / 2) *(df.DTE/365))) / (df.IV*np.sqrt((df.DTE/365)))),0.0,1.0)) * np.sqrt((df.DTE/365)))/100)*(df.POS_SIZE*df.Multiplier)
        except Exception as e: 
            logger.error(e)
        else:
            return vega

    def theta_div(df):
        try:
            if df.TYPE == 'C':
                theta = ((-((df.UNDLY_PRICE *(si.norm.pdf((((np.log(df.UNDLY_PRICE /df.STRIKE)) +(((df.RF_RATE-df.DIV_RATE)+(df.IV **2 )/2) *(df.DTE/365)))/(df.IV*np.sqrt((df.DTE/365)))),0.0,1.0))*df.IV) / (2*np.sqrt((df.DTE/365))))-((df.RF_RATE-df.DIV_RATE)*df.STRIKE*(1/(np.exp((df.DTE/365)*(df.RF_RATE -df.DIV_RATE)))) * (si.norm.cdf(((((np.log(df.UNDLY_PRICE/df.STRIKE))+(((df.RF_RATE -df.DIV_RATE)+(df.IV**2)/2) * (df.DTE/365)))/(df.IV*np.sqrt((df.DTE/365)))) -(df.IV*np.sqrt((df.DTE/365)))),0.0,1.0))))/365) * (df.Multiplier*df.POS_SIZE)

            if df.TYPE == 'P':
                theta = ((-((df.UNDLY_PRICE *(si.norm.pdf((((np.log(df.UNDLY_PRICE/df.STRIKE)) +(((df.RF_RATE-df.DIV_RATE)+(df.IV ** 2) / 2) * (df.DTE/365))) / (df.IV*np.sqrt((df.DTE/365)))),0.0,1.0))*df.IV)/(2*np.sqrt((df.DTE/365))))+((df.RF_RATE -df.DIV_RATE) * df.STRIKE * (1/(np.exp((df.DTE/365)*(df.RF_RATE -df.DIV_RATE))))*si.norm.cdf(-((((np.log(df.UNDLY_PRICE/df.STRIKE)) + (((df.RF_RATE-df.DIV_RATE) +(df.IV ** 2) / 2) * (df.DTE/365))) / (df.IV*np.sqrt((df.DTE/365)))) -(df.IV*np.sqrt((df.DTE/365)))),0.0,1.0)))/365)*(df.POS_SIZE*df.Multiplier)
        except Exception as e: 
            logger.error(e)
        else:
            return theta
    
    def gamma_div(df):
        try: 
            gamma = ((si.norm.pdf((((np.log(df.UNDLY_PRICE / df.STRIKE)) + (((df.RF_RATE-df.DIV_RATE) +(df.IV ** 2) / 2) * (df.DTE / 365))) / (df.IV*np.sqrt((df.DTE/365)))),0.0,1.0)) / (df.UNDLY_PRICE *(df.IV * np.sqrt((df.DTE/365))))) * (df.POS_SIZE * df.Multiplier)
        except Exception as e: 
            logger.error(e)
        else:
            return gamma
    
    def newton_vol_call_put_div(df, sigma=0.5, PRECISION=0.000001, MAX=100):
        for _ in range(MAX):
            try:
                d1 = (np.log(df.UNDLY_PRICE / df.STRIKE) + (df.RF_RATE - df.DIV_RATE + 0.5 * sigma ** 2) * (df.DTE/365)) / (sigma * np.sqrt(df.DTE/365))

                d2 = (np.log(df.UNDLY_PRICE / df.STRIKE) + (df.RF_RATE - df.DIV_RATE - 0.5 * sigma ** 2) * (df.DTE/365)) / (sigma * np.sqrt(df.DTE/365))

                vega = (1 / np.sqrt(2 * np.pi)) * df.UNDLY_PRICE * np.exp(-df.DIV_RATE * (df.DTE/365)) * np.sqrt(df.DTE/365) * np.exp((-si.norm.cdf(d1, 0.0, 1.0) ** 2) * 0.5)

                if df.TYPE == 'C':
                    fx = df.MID - (df.UNDLY_PRICE * np.exp(-df.DIV_RATE * (df.DTE/365)) * si.norm.cdf(d1, 0.0, 1.0) - df.STRIKE * np.exp(-df.RF_RATE * (df.DTE/365)) * si.norm.cdf(d2, 0.0, 1.0)) 

                if df.TYPE == 'P':
                    fx = df.MID - (df.STRIKE * np.exp(-df.RF_RATE * (df.DTE/365)) * si.norm.cdf(-d2, 0.0, 1.0) - df.UNDLY_PRICE * np.exp(-df.DIV_RATE * (df.DTE/365)) * si.norm.cdf(-d1, 0.0, 1.0))  
            except Exception as e: 
                logger.error(e)
            else:
                if abs(fx) < PRECISION: 
                    return sigma
                sigma = sigma + fx/vega
            return sigma
        
                
    def wtvega(df):
        try:
            (np.sqrt(22/df.DTE) * df.VEGA)
        except Exception as e: 
            logger.error(e)
        else:
            return (np.sqrt(22/df.DTE) * df.VEGA)
    
    def zomma(df): 
        try:
            zomma = df.GAMMA*(((((np.log(df.UNDLY_PRICE/df.STRIKE))+(((df.RF_RATE-df.DIV_RATE)+(df.IV**2)/2) *(df.DTE/365)))/(df.IV*np.sqrt((df.DTE/365))))*((((np.log(df.UNDLY_PRICE/df.STRIKE))+(((df.RF_RATE-df.DIV_RATE)+(df.IV**2)/2)*(df.DTE/365)))/(df.IV*np.sqrt((df.DTE/365))))-(df.IV*np.sqrt((df.DTE/365))))-1)/df.IV)
        except Exception as e: 
            logger.error(e)
        else:
            return zomma 
    
    def speed(df):
        try:
            speed = (-df.GAMMA / df.UNDLY_PRICE) * (1 + (((np.log( df.UNDLY_PRICE / df.STRIKE))+ (((df.RF_RATE - df.DIV_RATE) + (df.IV**2) / 2) * (df.DTE/365))) / (df.IV * np.sqrt(( df.DTE/365)))) / (df.IV * np.sqrt((df.DTE/365))))
        except Exception as e: 
            logger.error(e)
        else:
            return speed 
    
    def dual_delta(df):
        try:
            if df.TYPE == 'C':
                dual_delta =(-np.exp(-df.RF_RATE*(df.DTE/365)) * (si.norm.cdf(((((np.log(df.UNDLY_PRICE /df.STRIKE)) + (((df.RF_RATE-df.DIV_RATE)+(df.IV**2)/2) *(df.DTE/365))) / (df.IV*np.sqrt((df.DTE/365)))) - (df.IV*np.sqrt((df.DTE/365)))),0.0,1.0))) *(df.POS_SIZE*df.Multiplier)

            if df.TYPE == 'P':
                dual_delta = (np.exp(-df.RF_RATE*(df.DTE/365)) * si.norm.cdf(-((((np.log(df.UNDLY_PRICE /df.STRIKE)) + (((df.RF_RATE - df.DIV_RATE)+ (df.IV**2)/2) * (df.DTE/365))) /(df.IV*np.sqrt((df.DTE/365)))) - (df.IV*np.sqrt((df.DTE/365)))),0.0,1.0))*(df.POS_SIZE*df.Multiplier)
        except Exception as e: 
            logger.error(e)
        else:
            return dual_delta
    
    @staticmethod
    def decay_rate(df):
        try: 
            (df.THETA/100) / df.FAIRVALUE
        except Exception as e: 
            logger.error(e)
        else:
            return (df.THETA/100) / df.FAIRVALUE

    def euro_vanilla_dividend(df):
        try:
            d1 = (np.log(df.UNDLY_PRICE / df.STRIKE) + (df.RF_RATE - df.DIV_RATE + 0.5 * df.IV ** 2) * (df.DTE/365)) / (df.IV * np.sqrt(df.DTE/365))
            d2 = (np.log(df.UNDLY_PRICE / df.STRIKE) + (df.RF_RATE - df.DIV_RATE - 0.5 * df.IV ** 2) * (df.DTE/365)) / (df.IV * np.sqrt(df.DTE/365))
            if df.TYPE == 'C':
                fx = df.UNDLY_PRICE * np.exp(-df.DIV_RATE * (df.DTE/365)) * si.norm.cdf(d1, 0.0, 1.0) - df.STRIKE * np.exp(-df.RF_RATE * (df.DTE/365)) * si.norm.cdf(d2, 0.0, 1.0) 
            if df.TYPE == 'P':
                fx = df.STRIKE * np.exp(-df.RF_RATE * (df.DTE/365)) * si.norm.cdf(-d2, 0.0, 1.0) - df.UNDLY_PRICE * np.exp(-df.DIV_RATE * (df.DTE/365)) * si.norm.cdf(-d1, 0.0, 1.0)
        except Exception as e: 
            logger.error(e)
        else:
            return fx 
    
    def vomma(df):
        try:
            vomma = df.VEGA*((si.norm.cdf((((np.log(df.UNDLY_PRICE / df.STRIKE)) + (((df.RF_RATE -df.DIV_RATE) + (df.IV ** 2) / 2) * (df.DTE /365))) / (df.IV * np.sqrt((df.DTE / 365)))),0.0,1.0)
    *(si.norm.cdf(((((np.log(df.UNDLY_PRICE / df.STRIKE)) + (((df.RF_RATE - df.DIV_RATE) + (df.IV ** 2) / 2) * (df.DTE / 365))) / (df.IV*np.sqrt((df.DTE/365)))) - (df.IV*np.sqrt((df.DTE/365)))),0.0,1.0))) / df.IV)
        except Exception as e: 
            logger.error(e)
        else:
            return vomma 
    
    def vanna(df):
        try:
            vanna = (df.VEGA / df.UNDLY_PRICE) * (1-(((np.log(df.UNDLY_PRICE / df.STRIKE)) +(((df.RF_RATE -df.DIV_RATE) + (df.IV ** 2) / 2) *(df.DTE/365))) / (df.IV * np.sqrt((df.DTE / 365)))) / (df.IV*np.sqrt((df.DTE/365))))
        except Exception as e:
            logger.error(e)
        else:
            return vanna
    
    def charm(df): 
        try:
            charm = ((df.DIV_RATE * np.exp(-df.DIV_RATE *(df.DTE / 365)) * si.norm.cdf((((np.log(df.UNDLY_PRICE / df.STRIKE)) +(((df.RF_RATE - df.DIV_RATE) + (df.IV**2) / 2) *(df.DTE / 365))) / (df.IV*np.sqrt((df.DTE/365)))),0.0,1.0))-(np.exp(-df.DIV_RATE *(df.DTE/365)) * (si.norm.pdf((((np.log(df.UNDLY_PRICE / df.STRIKE)) +(((df.RF_RATE -df.DIV_RATE) + (df.IV ** 2) / 2) * (df.DTE / 365))) / (df.IV *np.sqrt((df.DTE / 365)))),0.0,1.0))) * ((2 *(df.RF_RATE - df.DIV_RATE) * (df.DTE/365) - ((((np.log(df.UNDLY_PRICE / df.STRIKE)) + (((df.RF_RATE - df.DIV_RATE) + (df.IV ** 2) / 2) *(df.DTE/365))) / (df.IV*np.sqrt((df.DTE/365)))) - (df.IV*np.sqrt((df.DTE/365)))) *(df.IV*np.sqrt((df.DTE / 365)))) / (2 * (df.DTE / 365) * (df.IV * np.sqrt((df.DTE /365)))))) *(df.POS_SIZE * df.Multiplier)
        except Exception as e: 
            logger.error(e)
        else:
            return charm 
    
    def color(df):
        try:
            color = ((-np.exp(-df.DIV_RATE*(df.DTE/365))*(si.norm.pdf((((np.log(df.UNDLY_PRICE/df.STRIKE))+(((df.RF_RATE-df.DIV_RATE)+(df.IV**2)/2)*(df.DTE/365)))/(df.IV*np.sqrt((df.DTE/365)))),0.0,1.0))/((df.UNDLY_PRICE**2)*(df.IV*np.sqrt((df.DTE/365)))))*((2*df.DIV_RATE*(df.DTE/365))+1+((((np.log(df.UNDLY_PRICE/df.STRIKE))+(((df.RF_RATE-df.DIV_RATE)+(df.IV**2)/2)*(df.DTE/365)))/(df.IV*np.sqrt((df.DTE/365))))*((2*(df.RF_RATE-df.DIV_RATE)*(df.DTE/365)-((((np.log(df.UNDLY_PRICE/df.STRIKE))+(((df.RF_RATE-df.DIV_RATE)+(df.IV**2)/2)*(df.DTE/365)))/(df.IV*np.sqrt((df.DTE/365)))) - (df.IV*np.sqrt((df.DTE/365))))*(df.IV*np.sqrt((df.DTE/365))))/(df.IV*np.sqrt((df.DTE/365)))))))*(df.Multiplier*df.POS_SIZE)
        except Exception as e: 
            logger.error(e)
        else:
            return color 
    
    
    
    
    
    
    
    





