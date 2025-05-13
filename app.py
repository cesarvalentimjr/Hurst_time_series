import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from hurst import compute_Hc
from PIL import Image

# Função robusta para calcular o índice de Hurst
def calculate_hurst_series(price_series, window_size):
    """
    Calcula o índice de Hurst com janela deslizante
    Args:
        price_series: pd.Series com os preços
        window_size: tamanho da janela de cálculo
    Returns:
        pd.Series com os valores de Hurst
    """
    if not isinstance(price_series, pd.Series):
        try:
            price_series = pd.Series(price_series)
        except Exception as e:
            st.error(f"Erro ao converter dados para Series: {e}")
            return pd.Series(dtype=float)
    
    clean_series = price_series.dropna()
    
    if len(clean_series) < window_size or len(clean_series) < 2:
        return pd.Series(dtype=float)
    
    hurst_values = []
    valid_indices = []
    
    for i in range(window_size, len(clean_series)):
        window_data = clean_series.iloc[i-window_size:i]
        try:
            H, _, _ = compute_Hc(window_data, kind='random_walk', simplified=True)
            hurst_values.append(H)
            valid_indices.append(clean_series.index[i])
        except Exception as e:
            continue
    
    return pd.Series(hurst_values, index=valid_indices)

# Configuração principal da aplicação
def main():
    st.set_page_config(layout="wide", page_title="Análise de Hurst")
    
    # Cabeçalho com logo
    try:
        logo = Image.open("Hurst_time_series.png")
        col1, col2 = st.columns([1, 6])
        with col1:
            st.image(logo, width=100)
        with col2:
            st.title("Análise de Regressão à Média ou Continuidade")
    except FileNotFoundError:
        st.title("Análise de Regressão à Média ou Continuidade")
        st.warning("Logo não encontrada. O arquivo 'Hurst_time_series.png' deve estar na mesma pasta.")

    # Parâmetros de entrada
    st.sidebar.header("Configurações")
    ticker = st.sidebar.text_input("Ativo (ex: btc-usd, ^BVSP)", "btc-usd")
    start_date = st.sidebar.text_input("Data Inicial (YYYY-MM-DD)", "2023-01-01")
    end_date = st.sidebar.text_input("Data Final (YYYY-MM-DD)", pd.to_datetime("today").strftime("%Y-%m-%d"))
    window_size = st.sidebar.slider("Janela do Hurst", 10, 500, 100, 10)
    
    if st.sidebar.button("Analisar"):
        with st.spinner("Carregando dados..."):
            try:
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                
                if data.empty:
                    st.error("Nenhum dado encontrado. Verifique o ticker e as datas.")
                    return
                
                # Pré-processamento
                data_viz = data[['Close']].copy()
                data_viz['SMA_200'] = data_viz['Close'].rolling(200).mean()
                data_viz['EMA_50'] = data_viz['Close'].ewm(span=50, adjust=False).mean()
                
                # Cálculo do Hurst
                hurst_series = calculate_hurst_series(data_viz['Close'], window_size)
                
                if hurst_series.empty:
                    st.warning("Não foi possível calcular o Hurst. Reduza o tamanho da janela.")
                    return
                
                # Visualização
                st.success("Análise concluída!")
                plot_results(data_viz, hurst_series, ticker, window_size)
                
                # Dados tabulares
                if st.checkbox("Mostrar dados completos"):
                    display_data = data_viz.join(hurst_series.rename('Hurst'))
                    st.dataframe(display_data.tail(100).style.format({
                        'Close': '{:.2f}',
                        'SMA_200': '{:.2f}',
                        'EMA_50': '{:.2f}',
                        'Hurst': '{:.4f}'
                    }))
                    
            except Exception as e:
                st.error(f"Erro durante a análise: {str(e)}")

def plot_results(data, hurst, ticker, window_size):
    """Gera os gráficos de análise"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'hspace': 0.3})
    
    # Gráfico de preços
    ax1.plot(data.index, data['Close'], label='Preço', color='blue', lw=1.5)
    ax1.plot(data.index, data['SMA_200'], label='SMA 200', color='orange', ls='--', lw=1.5)
    ax1.plot(data.index, data['EMA_50'], label='EMA 50', color='purple', ls='--', lw=1.5)
    
    # Áreas de regime
    ymin, ymax = data['Close'].min(), data['Close'].max()
    ax1.fill_between(hurst.index, ymin, ymax, where=(hurst > 0.5), 
                    color='green', alpha=0.1, label='Tendência (H>0.5)')
    ax1.fill_between(hurst.index, ymin, ymax, where=(hurst <= 0.5), 
                    color='red', alpha=0.1, label='Reversão (H≤0.5)')
    
    ax1.set_title(f"Preço e Médias Móveis - {ticker}", fontsize=14)
    ax1.legend(loc='upper left')
    ax1.grid(alpha=0.3)
    
    # Gráfico do Hurst
    ax2.plot(hurst.index, hurst, label='Índice de Hurst', color='saddlebrown', lw=1.5)
    ax2.axhline(0.5, color='black', ls='--', lw=1, label='Limite H=0.5')
    
    ax2.fill_between(hurst.index, 0.5, hurst, where=(hurst>0.5), color='green', alpha=0.1)
    ax2.fill_between(hurst.index, 0.5, hurst, where=(hurst<=0.5), color='red', alpha=0.1)
    
    ax2.set_title(f"Índice de Hurst (Janela={window_size})", fontsize=14)
    ax2.set_ylim(0, 1)
    ax2.legend(loc='upper left')
    ax2.grid(alpha=0.3)
    
    # Formatação comum
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, len(data)//10)))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    st.pyplot(fig)
    plt.close()

if __name__ == '__main__':
    main()

