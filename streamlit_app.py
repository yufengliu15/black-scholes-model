import streamlit as st
import numpy as np
from numpy import exp, sqrt, log
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Black-Scholes Option Pricing Model",
    layout="wide",
    initial_sidebar_state="expanded")

# Custom CSS to inject into Streamlit
st.markdown("""
<style>
/* Adjust the size and alignment of the CALL and PUT value containers */
.metric-container {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 8px;
    width: auto; 
    margin: 0 auto; 
}

/* Custom classes for CALL and PUT values */
.metric-call {
    background-color: #90ee90; 
    color: black;
    margin-right: 10px; 
    border-radius: 10px; 
}

.metric-put {
    background-color: #ffcccb;
    color: black;
    border-radius: 10px; 
}

/* Style for the value text */
.metric-value {
    font-size: 1.5rem;
    font-weight: bold;
    margin: 0;
}

/* Style for the label text */
.metric-label {
    font-size: 1rem;
    margin-bottom: 4px;
}

</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("Black-Scholes Model")
    st.info("View the source code below")
    st.markdown("<a href='https://github.com/yufengliu15/black-scholes-model'>Source Code</a>", unsafe_allow_html=True)
    stock_price = st.number_input("Current Asset Price", value=100.0)
    strike = st.number_input("Strike Price", value=100.0)
    time = st.number_input("Time to Maturity (Years)", value=1.0)
    volatility = st.number_input("Volatility (σ)", value=0.2)
    risk_interest = st.number_input("Risk-Free Interest Rate", value=0.05)

    st.markdown("---")
    calculate_btn = st.button('Heatmap Parameters')
    spot_min = st.number_input('Min Spot Price', min_value=0.01, value=stock_price*0.8, step=0.01)
    spot_max = st.number_input('Max Spot Price', min_value=0.01, value=stock_price*1.2, step=0.01)
    vol_min = st.slider('Min Volatility for Heatmap', min_value=0.01, max_value=1.0, value=volatility*0.5, step=0.01)
    vol_max = st.slider('Max Volatility for Heatmap', min_value=0.01, max_value=1.0, value=volatility*1.5, step=0.01)
    
    spot_range = np.linspace(spot_min, spot_max, 10)
    vol_range = np.linspace(vol_min, vol_max, 10)

class BlackScholes:
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

        call_price = stock_price * norm.cdf(d1) - strike * exp(-risk_interest * time) * norm.cdf(d2)
        put_price = strike * exp(-risk_interest * time) * norm.cdf(-d2) - stock_price * norm.cdf(-d1)

        self.call_price = call_price
        self.put_price = put_price

        return call_price, put_price

# Calculate Call and Put values
bs_model = BlackScholes(stock_price, strike, time, risk_interest, volatility)
call_price, put_price = bs_model.run()

st.title("Black-Scholes Pricing Model")
# Display Call and Put Values in colored tables
col1, col2 = st.columns([1,1], gap="small")

with col1:
    # Using the custom class for CALL value
    st.markdown(f"""
        <div class="metric-container metric-call">
            <div>
                <div class="metric-label">CALL Value</div>
                <div class="metric-value">${call_price:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    # Using the custom class for PUT value
    st.markdown(f"""
        <div class="metric-container metric-put">
            <div>
                <div class="metric-label">PUT Value</div>
                <div class="metric-value">${put_price:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# ======== Heatmap ========
def plot_heatmap(bs_model, spot_range, vol_range, strike):
    call_prices = np.zeros((len(vol_range), len(spot_range)))
    put_prices = np.zeros((len(vol_range), len(spot_range)))

    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            bs_cell = BlackScholes(
                stock_price=spot,
                time=bs_model.time,
                strike=strike,
                risk_interest=bs_model.risk_interest,
                volatility=vol
            )

            bs_cell.run()
            call_prices[i,j] = bs_cell.call_price
            put_prices[i,j] = bs_cell.put_price

    # Plotting Call Price Heatmap
    fig_call, ax_call = plt.subplots(figsize=(10, 8))
    sns.heatmap(call_prices, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), annot=True, fmt=".2f", cmap="coolwarm", ax=ax_call)
    ax_call.set_title('CALL')
    ax_call.set_xlabel('Spot Price')
    ax_call.set_ylabel('Volatility')
    
    # Plotting Put Price Heatmap
    fig_put, ax_put = plt.subplots(figsize=(10, 8))
    sns.heatmap(put_prices, xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), annot=True, fmt=".2f", cmap="coolwarm", ax=ax_put)
    ax_put.set_title('PUT')
    ax_put.set_xlabel('Spot Price')
    ax_put.set_ylabel('Volatility')
    
    return fig_call, fig_put

st.markdown("")
st.title("Options Price - Interactive Heatmap")
st.info("Explore how option prices fluctuate with varying 'Spot Prices and Volatility' levels using interactive heatmap parameters, all while maintaining a constant 'Strike Price'.")

# Interactive Sliders and Heatmaps for Call and Put Options
col1, col2 = st.columns([1,1], gap="small")

with col1:
    st.subheader("Call Price Heatmap")
    heatmap_fig_call, _ = plot_heatmap(bs_model, spot_range, vol_range, strike)
    st.pyplot(heatmap_fig_call)

with col2:
    st.subheader("Put Price Heatmap")
    _, heatmap_fig_put = plot_heatmap(bs_model, spot_range, vol_range, strike)
    st.pyplot(heatmap_fig_put)