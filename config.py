# Contains all parameters

# Curates set of liquid, sector-grouped tickers 
# Pairs are chosen within same sector
UNIVERSE = {
    "beverages":    ["KO", "PEP", "MNST", "KDP", "STZ", "TAP", "BUD", "SAM"],
    "banks":        ["JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC", "TFC", "SCHW", "BK", "STT"],
    "oil":          ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "VLO", "PSX", "OXY", "HAL", "DVN", "PXD"],
    "tech":         ["MSFT", "GOOGL", "META", "AAPL", "AMZN", "NVDA", "AMD", "INTC", "QCOM", "TXN", "MU", "AMAT"],
    "retail":       ["WMT", "TGT", "COST", "DG", "DLTR", "KR", "ACI", "BJ", "FIVE", "OLLI"],
    "pharma":       ["JNJ", "PFE", "MRK", "ABBV", "BMY", "LLY", "AMGN", "GILD", "BIIB", "REGN", "VRTX", "ZTS"],
    "airlines":     ["DAL", "UAL", "AAL", "LUV", "ALK", "JBLU", "SAVE", "HA"],
    "insurance":    ["AIG", "MET", "PRU", "AFL", "TRV", "ALL", "HIG", "GL", "LNC", "UNM"],
    "utilities":    ["NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "PCG", "ED", "XEL"],
    "reits":        ["PLD", "AMT", "CCI", "EQIX", "PSA", "EQR", "AVB", "DRE", "VTR", "WELL"],
    "media":        ["DIS", "NFLX", "CMCSA", "WBD", "PARA", "FOX", "FOXA", "LYV"],
    "auto":         ["GM", "F", "TSLA", "TM", "HMC", "STLA", "AN", "KMX", "LAD"],
}

# --- Date range ---
# The model is trained on data from 2010 - 2019
# Analysis and predictions are gauged on accuracy based on real performance from 2020-2024
TRAIN_START = "2010-01-01"
TRAIN_END   = "2019-12-31"
TEST_START  = "2020-01-01"
TEST_END    = "2024-12-31"

# --- Pair selection ---
COINT_PVALUE_THRESHOLD = 0.05     
MIN_HALF_LIFE_DAYS     = 5         
MAX_HALF_LIFE_DAYS     = 60        

# --- Kalman filter (hedge ratio) ---
# Controls how quickly the Kalman filter allows the hedge ratio to shift
# Reasonable trusting of fluctuating market values; updates hedge ratio
KALMAN_DELTA           = 1e-4      
KALMAN_VE              = 0.01      

# --- Signal generation ---
ROLLING_WINDOW         = 60        
ENTRY_ZSCORE           = 2.0       
EXIT_ZSCORE            = 0.0       
STOP_LOSS_ZSCORE       = 3.5      

# --- Regime (HMM) ---
# Training calm and crisis regimes on one year of trading
N_REGIMES              = 2        
HMM_LOOKBACK           = 252       

# --- Backtester ---
# Basis points per transaction (x2 for both legs)
TRANSACTION_COST_BPS   = 5        
CAPITAL_PER_PAIR       = 10_000   