import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from hurst import compute_Hc
from PIL import Image

def safe_series_conversion(data):
    """Conversão segura para pandas Series"""
    if isinstance(data, pd.Series):
        return data.copy()
    try:
        if hasattr(data, 'values'):
            return pd.Series(data.values.flatten())
        elif isinstance(data, (list, tuple, np.ndarray)):
            return pd.Series(data)
        else:
            return pd.Series([data])
    except Exception as e:
        st.error(f"Falha na conversão para Series: {str(e)}")
        return pd.Series(dtype=float)

def calculate_hurst_series(price_data, window_size):
    """Calcula o índice de Hurst com tratamento robusto de erros"""
    prices = safe_series_conversion(price_data).dropna()
    
    if len(prices) < window_size or len(prices) < 2:
        return pd.Series(dtype=float)
    
    hurst_values = []
    valid_indices = []
    
    for i in range(window_size, len(prices)):
        window = prices.iloc[i-window_size:i]
        try:
            H, _, _ = compute_Hc(window, kind='random_walk', simplified=True)
            hurst_values.append(H)
            valid_indices.append(prices.index[i])
        except Exception:
            continue
    
    return pd.Series(hurst_values, index=valid_indices)

def main():
    st.set_page_config(layout="wide")
    
    # Interface do usuário
    try:
        logo = Image.open("Hurst_time_series.png")
        col1, col2 = st.columns([1, 6])
        with col1:
            st.image(logo, width=100)
        with col2:
            st.title("Análise de Regressão à Média")
    except:
        st.title("Análise de Regressão à Média")
    
    st.sidebar.header("Parâmetros")
    ticker = st.sidebar.text_input("Ticker", "btc-usd")
    start_date = st.sidebar.text_input("Data Inicial", "2023-01-01")
    end_date = st.sidebar.text_input("Data Final", pd.to_datetime("today").strftime("%Y-%m-%d"))
    window_size = st.sidebar.selectbox("Janela do Hurst", [30, 50, 100, 200], index=2)
    
    if st.sidebar.button("Analisar"):
        with st.spinner("Processando..."):
            try:
                # Download dos dados
                data = yf.download(ticker, start=start_date, end=end_date)
                
                if data.empty:
                    st.error("Dados não encontrados. Verifique o ticker e período.")
                    return
                
                # Pré-processamento
                close_prices = data['Close'].copy()
                
                # Cálculo do Hurst
                hurst = calculate_hurst_series(close_prices, window_size)
                
                if hurst.empty:
                    st.warning("Hurst não pôde ser calculado. Tente uma janela menor.")
                    return
                
                # Visualização
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                
                # Gráfico de preços
                ax1.plot(close_prices.index, close_prices, 'b-', label="Preço")
                ax1.set_title(f"Preço de Fechamento - {ticker}")
                ax1.grid(True, alpha=0.3)
                
                # Gráfico do Hurst
                ax2.plot(hurst.index, hurst, 'r-', label="Índice de Hurst")
                ax2.axhline(0.5, color='k', linestyle='--')
                ax2.set_title(f"Índice de Hurst (Janela={window_size})")
                ax2.set_ylim(0, 1)
                ax2.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                plt.close()
                
                # Estatísticas
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Último Valor do Hurst", f"{hurst.iloc[-1]:.4f}")
                with col2:
                    regime = "Tendência" if hurst.iloc[-1] > 0.5 else "Reversão"
                    st.metric("Regime Atual", regime)
                
            except Exception as e:
                st.error(f"Erro durante a execução: {str(e)}")

if __name__ == '__main__':
    main()
