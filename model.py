from numpy import exp, sqrt, log
from scipy.stats import norm

class BlackSholes:
    # Represents European options
    # Time to maturity in years
    def __init__(self, stock_price: float, strike: float, time: float, risk_interest: float, volatility: float):
        self.stock_price = stock_price
        self.strike = strike
        self.time = time
        self.risk_interest = risk_interest
        self.volatility = volatility
    
    def run(self):
        stock_price = self.stock_price
        strike = self.strike
        time = self.time
        risk_interest = self.risk_interest
        volatility = self.volatility

        d1 = (log(stock_price/strike) + (risk_interest + volatility**2 / 2) * time) / (volatility * sqrt(time))
        d2 = (log(stock_price/strike) + (risk_interest - volatility**2 / 2) * time) / (volatility * sqrt(time))

        self.call_price = stock_price * norm.cdf(d1) - strike * exp(-risk_interest * time) * norm.cdf(d2)
        self.put_price = strike * exp(-risk_interest * time) * norm.cdf(-d2) - stock_price * norm.cdf(-d1)

if __name__ == "__main__":
    stock_price = 200
    strike = 500
    time = 0.5
    risk_interest = 0.04
    volatility = 1

    BS = BlackSholes(stock_price, strike, time, risk_interest, volatility)

    BS.run()

    print(f"BS.call_price: {BS.call_price}, BS.put_price: {BS.put_price}")