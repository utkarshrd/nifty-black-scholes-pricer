import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set a professional color palette and style
plt.style.use('seaborn-v0_8-darkgrid')
FIN_COLORS = {'blue': '#0A4A8F', 'orange': '#F47920', 'green': '#00A451'} 
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 12


def bs_pricer(S, K, T, r, sigma):
    """
    Calculates the Black-Scholes price for a European call and put option.
    
    Parameters:
    S (float): Spot price of the underlying asset
    K (float): Strike price of the option
    T (float): Time to expiry in years
    r (float): Risk-free interest rate (annualized)
    sigma (float): Volatility of the underlying asset (annualized)
    
    Returns:
    (float, float): Tuple containing the Call Price and Put Price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S * st.norm.cdf(d1) - K * np.exp(-r * T) * st.norm.cdf(d2)
    put_price = K * np.exp(-r * T) * st.norm.cdf(-d2) - S * st.norm.cdf(-d1)
    
    return call_price, put_price, d1, d2

def calculate_greeks(S, K, T, r, sigma, d1, d2):
    """
    Calculates the Greeks for a European option.
    
    Returns:
    dict: A dictionary containing delta, gamma, theta, and vega.
    """
    # Delta
    call_delta = st.norm.cdf(d1)
    put_delta = call_delta - 1
    
    # Gamma
    gamma = st.norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    # Theta (per day)
    term1 = - (S * st.norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    call_theta_term2 = - r * K * np.exp(-r * T) * st.norm.cdf(d2)
    put_theta_term2 = + r * K * np.exp(-r * T) * st.norm.cdf(-d2)
    call_theta = (term1 + call_theta_term2) / 365
    put_theta = (term1 + put_theta_term2) / 365
    
    # Vega (per 1% change in vol)
    vega = S * st.norm.pdf(d1) * np.sqrt(T) * 0.01
    
    return {
        'Call Delta': call_delta,
        'Put Delta': put_delta,
        'Gamma': gamma,
        'Call Theta': call_theta,
        'Put Theta': put_theta,
        'Vega': vega
    }


# --- INPUTS ---
S0 = 24500
STRIKES = np.arange(24000, 25001, 100)
EXPIRIES_DAYS = [7, 14, 30, 60]
TTE = np.array(EXPIRIES_DAYS) / 365.0 # Time to expiry in years
R_INDIA = 0.07
SIGMA_BASE = 0.15 # 15% baseline volatility
SIGMA_RANGE = np.arange(0.10, 0.26, 0.01)

# --- CALCULATION ---
results = []
for K in STRIKES:
    for T_days, T_years in zip(EXPIRIES_DAYS, TTE):
        # Calculate price and greeks at the baseline volatility
        call_price, put_price, d1, d2 = bs_pricer(S0, K, T_years, R_INDIA, SIGMA_BASE)
        greeks = calculate_greeks(S0, K, T_years, R_INDIA, SIGMA_BASE, d1, d2)
        
        results.append({
            'Strike': K,
            'Expiry (Days)': T_days,
            'Volatility': SIGMA_BASE,
            'Call Price': call_price,
            'Put Price': put_price,
            **greeks
        })

df = pd.DataFrame(results)

print("Sample Output: NIFTY Option Prices and Greeks")
# In a non-Jupyter environment, 'display' is not available.
# You would typically print the DataFrame head directly.
print(df[df['Expiry (Days)'] == 30].head())


df_30d = df[df['Expiry (Days)'] == 30]

plt.figure()
plt.plot(df_30d['Strike'], df_30d['Call Price'], label='Call Prices', color=FIN_COLORS['blue'], linewidth=2.5)
plt.plot(df_30d['Strike'], df_30d['Put Price'], label='Put Prices', color=FIN_COLORS['orange'], linewidth=2.5)

plt.title(f'NIFTY Call & Put Prices vs. Strike\n(Expiry: 30 Days, Spot: ₹{S0}, Vol: {SIGMA_BASE*100:.0f}%)')
plt.xlabel('Strike Price (₹)')
plt.ylabel('Option Price (₹)')
plt.axvline(x=S0, color='grey', linestyle='--', label=f'Spot Price (₹{S0})')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Save the plot
plt.savefig('nifty_options_pricer/outputs/price_vs_strike.png', dpi=300, bbox_inches='tight')
# plt.show() # Uncomment to display plots if running in an interactive environment


plt.figure()
plt.plot(df_30d['Strike'], df_30d['Call Delta'], label='Call Delta', color=FIN_COLORS['blue'], linewidth=2.5)
plt.plot(df_30d['Strike'], df_30d['Put Delta'], label='Put Delta', color=FIN_COLORS['orange'], linewidth=2.5)

plt.title('NIFTY Option Delta vs. Strike (Expiry: 30 Days)')
plt.xlabel('Strike Price (₹)')
plt.ylabel('Delta')
plt.axvline(x=S0, color='grey', linestyle='--', label=f'Spot Price (₹{S0})')
plt.axhline(y=0.5, color='black', linestyle=':', linewidth=1, label='0.5 Delta')
plt.axhline(y=-0.5, color='black', linestyle=':', linewidth=1, label='-0.5 Delta')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Save the plot
plt.savefig('nifty_options_pricer/outputs/delta_vs_strike.png', dpi=300, bbox_inches='tight')
# plt.show()


T_mesh, SIGMA_mesh = np.meshgrid(TTE, SIGMA_RANGE)
# Using an ATM strike for the surface plot
K_atm = S0 

_c, _p, d1, _d2 = bs_pricer(S0, K_atm, T_mesh, R_INDIA, SIGMA_mesh)
greeks = calculate_greeks(S0, K_atm, T_mesh, R_INDIA, SIGMA_mesh, d1, _d2)
vega_surface = greeks['Vega'].reshape(len(SIGMA_RANGE), len(TTE))

fig = plt.figure(figsize=(14, 9))
ax = fig.add_subplot(111, projection='3d')

X, Y = np.meshgrid(EXPIRIES_DAYS, SIGMA_RANGE * 100)
ax.plot_surface(X, Y, vega_surface, cmap='viridis')

ax.set_title('Vega Surface for ATM NIFTY Option')
ax.set_xlabel('Time to Expiry (Days)')
ax.set_ylabel('Volatility (%)')
ax.set_zlabel('Vega (per 1% vol change)')
ax.view_init(30, -45) # Elevate and rotate the camera

# Save the plot
plt.savefig('nifty_options_pricer/outputs/vega_surface.png', dpi=300, bbox_inches='tight')
# plt.show()


df_compare = df_30d[df_30d['Strike'].isin([24300, 24500, 24700, 24900])].copy()

# Create hypothetical market prices (Model Price +/- some noise)
np.random.seed(42)
noise = np.random.uniform(-15, 15, size=len(df_compare))
df_compare['Market Price'] = df_compare['Call Price'] + noise

bar_width = 0.35
index = np.arange(len(df_compare))

plt.figure()
plt.bar(index, df_compare['Call Price'], bar_width, label='Model Price', color=FIN_COLORS['blue'])
plt.bar(index + bar_width, df_compare['Market Price'], bar_width, label='Market Price', color=FIN_COLORS['green'])

plt.title('Model Price vs. Market Price Comparison (30-Day Calls)')
plt.xlabel('Strike Price (₹)')
plt.ylabel('Option Price (₹)')
plt.xticks(index + bar_width / 2, df_compare['Strike'])
plt.legend()
plt.grid(axis='y', linestyle='--', linewidth=0.5)

# Save the plot
plt.savefig('nifty_options_pricer/outputs/model_vs_market.png', dpi=300, bbox_inches='tight')
# plt.show()


print("\n--- Statistical Summary of 30-Day Options ---")
summary_stats = df_30d[['Call Price', 'Put Price', 'Call Delta', 'Put Delta', 'Gamma', 'Call Theta', 'Put Theta', 'Vega']].describe()
print(summary_stats.to_string()) # Use print instead of display for non-Jupyter environment

# Export the full dataset to a CSV file
csv_path = 'nifty_options_pricer/outputs/option_prices_and_greeks.csv'
df.to_csv(csv_path, index=False)

print(f"\nFull results have been exported to: {csv_path}")