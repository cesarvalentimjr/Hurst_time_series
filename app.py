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

def plot_combined_charts(data, hurst_series, ticker, window_size):
    """Cria os gráficos combinados conforme solicitado"""
    fig, axes = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'hspace': 0.3})
    
    # Gráfico de preços com médias móveis
    axes[0].plot(data.index, data['Close'], label="Preço", color='blue', linewidth=1.5)
    axes[0].plot(data.index, data['SMA_200'], label="SMA 200", color='orange', linestyle='--', linewidth=1.5)
    axes[0].plot(data.index, data['EMA_50'], label="EMA 50", color='purple', linestyle='--', linewidth=1.5)
    
    # Áreas de regime
    ymin, ymax = data['Close'].min(), data['Close'].max()
    axes[0].fill_between(
        hurst_series.index,
        ymin,
        ymax,
        where=(hurst_series > 0.5),
        color='green',
        alpha=0.2,
        label="Tendência (H>0.5)"
    )
    axes[0].fill_between(
        hurst_series.index,
        ymin,
        ymax,
        where=(hurst_series <= 0.5),
        color='red',
        alpha=0.2,
        label="Reversão (H≤0.5)"
    )
    
    axes[0].set_title(f"Preço e Médias Móveis - {ticker}", fontsize=14)
    axes[0].set_ylabel("Preço", fontsize=12)
    axes[0].legend(loc='upper left', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Gráfico do Hurst
    axes[1].plot(hurst_series.index, hurst_series, label="Índice de Hurst", color='saddlebrown', linewidth=1.5)
    axes[1].axhline(0.5, color='black', linestyle='--', linewidth=1.2, label="Limite H=0.5")
    
    axes[1].fill_between(
        hurst_series.index,
        0.5,
        hurst_series,
        where=(hurst_series > 0.5),
        color='green',
        alpha=0.2
    )
    axes[1].fill_between(
        hurst_series.index,
        0.5,
        hurst_series,
        where=(hurst_series <= 0.5),
        color='red',
        alpha=0.2
    )
    
    axes[1].set_title(f"Índice de Hurst (Janela={window_size} períodos)", fontsize=14)
    axes[1].set_xlabel("Data", fontsize=12)
    axes[1].set_ylabel("Índice de Hurst", fontsize=12)
    axes[1].set_ylim(0, 1)
    axes[1].legend(loc='upper left', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # Formatação comum dos eixos de data
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, len(data)//10))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    return fig

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
    window_size = st.sidebar.slider("Janela do Hurst", 30, 200, 100, 10)
    
    if st.sidebar.button("Analisar"):
        with st.spinner("Processando dados..."):
            try:
                # Download e preparação dos dados
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                
                if data.empty:
                    st.error("Dados não encontrados. Verifique o ticker e período.")
                    return
                
                # Cálculo das médias móveis
                data = data[['Close']].copy()
                data['SMA_200'] = data['Close'].rolling(window=200).mean()
                data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
                
                # Cálculo do Hurst
                hurst_series = calculate_hurst_series(data['Close'], window_size)
                
                if hurst_series.empty:
                    st.warning("Não foi possível calcular o Hurst. Tente uma janela menor.")
                    return
                
                # Alinhamento dos índices
                common_index = data.index.intersection(hurst_series.index)
                data = data.loc[common_index]
                hurst_series = hurst_series.loc[common_index]
                
                # Plotagem dos gráficos
                fig = plot_combined_charts(data, hurst_series, ticker, window_size)
                st.pyplot(fig)
                plt.close(fig)
                
                # Estatísticas
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Último Valor do Hurst", f"{hurst_series.iloc[-1]:.4f}")
                with col2:
                    regime = "Tendência" if hurst_series.iloc[-1] > 0.5 else "Reversão"
                    st.metric("Regime Atual", regime)
                
                # Dados tabulares
                if st.checkbox("Mostrar dados completos"):
                    display_data = data.copy()
                    display_data['Hurst'] = hurst_series
                    st.dataframe(display_data.tail(100).style.format({
                        'Close': '{:.2f}',
                        'SMA_200': '{:.2f}',
                        'EMA_50': '{:.2f}',
                        'Hurst': '{:.4f}'
                    })
                    
            except Exception as e:
                st.error(f"Erro durante a análise: {str(e)}")

if __name__ == '__main__':
    main()
