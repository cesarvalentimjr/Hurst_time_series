import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from hurst import compute_Hc
from PIL import Image

# Configuração inicial da página
st.set_page_config(layout="wide", page_title="Análise de Hurst")

def load_logo():
    try:
        logo = Image.open("Hurst_time_series.png")
        col1, col2 = st.columns([1, 6])
        with col1:
            st.image(logo, width=100)
        with col2:
            st.title("Análise de Regressão à Média ou Continuidade")
    except FileNotFoundError:
        st.title("Análise de Regressão à Média ou Continuidade")
        st.warning("Arquivo da logo não encontrado. Coloque 'Hurst_time_series.png' na mesma pasta do script.")

def calculate_hurst_series(price_series, window_size):
    if not isinstance(price_series, pd.Series):
        price_series = pd.Series(price_series.squeeze()) if hasattr(price_series, 'squeeze') else pd.Series(price_series)
    clean_prices = price_series.dropna()
    if len(clean_prices) < window_size or len(clean_prices) < 2:
        return pd.Series(dtype=float)

    hurst_values = []
    valid_indices = []

    for i in range(window_size, len(clean_prices)):
        window_data = clean_prices.iloc[i-window_size:i]
        try:
            H, _, _ = compute_Hc(window_data.values.flatten(), kind='random_walk', simplified=True)
            hurst_values.append(H)
            valid_indices.append(clean_prices.index[i])
        except Exception:
            continue

    hurst_series = pd.Series(index=price_series.index, dtype=float)
    hurst_series.loc[valid_indices] = hurst_values
    return hurst_series.dropna()

def classify_hurst(H):
    if H < 0.3:
        return "Forte Reversão"
    elif H < 0.4:
        return "Reversão"
    elif H < 0.5:
        return "Leve Reversão"
    elif H < 0.6:
        return "Leve Tendência"
    elif H < 0.7:
        return "Tendência"
    else:
        return "Forte Tendência"

def plot_analysis(data, hurst_series, ticker, window_size):
    fig, axes = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'hspace': 0.3})

    axes[0].plot(data.index, data['Close'], label="Preço", color='blue', linewidth=1.5)
    axes[0].plot(data.index, data['SMA_200'], label="SMA 200", color='orange', linestyle='--', linewidth=1.5)
    axes[0].plot(data.index, data['EMA_50'], label="EMA 50", color='purple', linestyle='--', linewidth=1.5)
    ymin, ymax = data['Close'].min(), data['Close'].max()

    colors = {
        "Forte Reversão": ('darkred', 0.0, 0.3),
        "Reversão": ('red', 0.3, 0.4),
        "Leve Reversão": ('lightcoral', 0.4, 0.5),
        "Leve Tendência": ('lightgreen', 0.5, 0.6),
        "Tendência": ('green', 0.6, 0.7),
        "Forte Tendência": ('darkgreen', 0.7, 1.0)
    }

    for label, (color, low, high) in colors.items():
        mask = hurst_series.between(low, high)
        axes[0].fill_between(hurst_series.index, ymin, ymax, where=mask, color=color, alpha=0.2, label=label)

    axes[0].set_title(f"Preço e Médias Móveis - {ticker}", fontsize=14)
    axes[0].set_ylabel("Preço", fontsize=12)
    axes[0].legend(loc='upper left', fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(hurst_series.index, hurst_series, label="Índice de Hurst", color='saddlebrown', linewidth=1.5)
    for label, (color, low, high) in colors.items():
        mask = hurst_series.between(low, high)
        axes[1].fill_between(hurst_series.index, low, high, where=mask, color=color, alpha=0.2, label=label)

    axes[1].axhline(0.5, color='black', linestyle='--', linewidth=1.2, label="H=0.5")
    axes[1].set_title(f"Índice de Hurst (Janela={window_size} períodos)", fontsize=14)
    axes[1].set_xlabel("Data", fontsize=12)
    axes[1].set_ylabel("Índice de Hurst", fontsize=12)
    axes[1].set_ylim(0, 1)
    axes[1].legend(loc='upper left', fontsize=10)
    axes[1].grid(True, alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, len(data)//10)))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    return fig

def main():
    load_logo()
    st.sidebar.header("Configurações")
    ticker = st.sidebar.text_input("Ticker", "btc-usd")
    start_date = st.sidebar.text_input("Data Inicial (YYYY-MM-DD)", "2023-01-01")
    end_date = st.sidebar.text_input("Data Final (YYYY-MM-DD)", pd.to_datetime("today").strftime("%Y-%m-%d"))
    window_size = st.sidebar.slider("Janela do Hurst", 30, 200, 100, 10)

    if st.sidebar.button("Analisar"):
        with st.spinner("Carregando e processando dados..."):
            try:
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if data.empty:
                    st.error("Nenhum dado encontrado. Verifique o ticker e as datas.")
                    return

                data = data[['Close']].copy()
                data['SMA_200'] = data['Close'].rolling(window=200, min_periods=1).mean()
                data['EMA_50'] = data['Close'].ewm(span=50, adjust=False, min_periods=1).mean()

                hurst_series = calculate_hurst_series(data['Close'].squeeze(), window_size)
                if hurst_series.empty:
                    st.warning(f"Não foi possível calcular o Hurst com janela de {window_size} períodos. Tente uma janela menor.")
                    fig, ax = plt.subplots(figsize=(16, 6))
                    ax.plot(data.index, data['Close'], label="Preço", color='blue', linewidth=1.5)
                    ax.plot(data.index, data['SMA_200'], label="SMA 200", color='orange', linestyle='--', linewidth=1.5)
                    ax.plot(data.index, data['EMA_50'], label="EMA 50", color='purple', linestyle='--', linewidth=1.5)
                    ax.set_title(f"Preço e Médias Móveis - {ticker}")
                    ax.legend()
                    st.pyplot(fig)
                    plt.close(fig)
                    return

                data = data.loc[hurst_series.index]
                fig = plot_analysis(data, hurst_series, ticker, window_size)
                st.pyplot(fig)
                plt.close(fig)

                st.subheader("Estatísticas Atuais")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Último Valor do Hurst", f"{hurst_series.iloc[-1]:.4f}")
                with col2:
                    regime = classify_hurst(hurst_series.iloc[-1])
                    st.metric("Regime Atual", regime)

                if st.checkbox("Mostrar dados completos"):
                    display_data = data.copy()
                    display_data['Hurst'] = hurst_series
                    formatted_data = display_data.tail(100).style.format({
                        'Close': '{:.2f}',
                        'SMA_200': '{:.2f}',
                        'EMA_50': '{:.2f}',
                        'Hurst': '{:.4f}'
                    })
                    st.dataframe(formatted_data)
            except Exception as e:
                st.error(f"Erro durante a análise: {str(e)}")
    else:
        st.info("Ajuste os parâmetros e clique em 'Analisar' para gerar os gráficos.")

if __name__ == '__main__':
    main()

