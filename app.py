import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from hurst import compute_Hc
from PIL import Image # Para a logo

# Função para calcular o índice de Hurst com uma janela deslizante
def calculate_hurst_series_func(series_data, window_size_hurst):
    hurst_values = []
    # Garante que series_data seja uma Series e remove NaNs
    series_data_clean = pd.Series(series_data).dropna()
    
    # Verifica se há dados suficientes após a limpeza
    if len(series_data_clean) < window_size_hurst or len(series_data_clean) < 2:
        return pd.Series(dtype=float) # Retorna uma série vazia se não houver dados suficientes

    for i in range(window_size_hurst, len(series_data_clean)):
        window_data = series_data_clean[i-window_size_hurst:i]
        # compute_Hc precisa de pelo menos 2 pontos e a janela deve ter o tamanho esperado
        if len(window_data) < 2 or len(window_data) < window_size_hurst:
            hurst_values.append(np.nan)
            continue
        try:
            H, _, _ = compute_Hc(window_data, kind='random_walk', simplified=True)
            hurst_values.append(H)
        except Exception: # Captura exceções mais genéricas do compute_Hc
            hurst_values.append(np.nan)
            
    if hurst_values:
        # Garante que o índice corresponda aos dados limpos e fatiados
        return pd.Series(hurst_values, index=series_data_clean.index[window_size_hurst:])
    else:
        return pd.Series(dtype=float)

# Função principal da aplicação
def main():
    st.set_page_config(layout="wide")

    # Título e Logo
    try:
        logo = Image.open("Hurst_time_series.png") 
        col1, col2 = st.columns([1, 6]) 
        with col1:
            st.image(logo, width=100) 
        with col2:
            st.title("Análise de regressão à média ou continuidade")
    except FileNotFoundError:
        st.title("Análise de regressão à média ou continuidade")
        st.warning("Arquivo da logo 'Hurst_time_series.png' não encontrado. Coloque o arquivo na mesma pasta do app.py.")

    # Parâmetros de entrada
    ticker_default = "btc-usd"
    start_date_default = "2023-01-01"
    end_date_default = pd.to_datetime("today").strftime("%Y-%m-%d")
    window_size_default = 100

    st.sidebar.header("Parâmetros de Análise")
    ticker = st.sidebar.text_input("Ticker (ex: btc-usd, ^BVSP)", ticker_default)
    start_date = st.sidebar.text_input("Data de Início (YYYY-MM-DD)", start_date_default)
    end_date = st.sidebar.text_input("Data Final (YYYY-MM-DD)", end_date_default)
    window_size = st.sidebar.number_input("Janela do Hurst (períodos)", min_value=10, max_value=500, value=window_size_default, step=1)

    if st.sidebar.button("Analisar"):
        data_load_state = st.text(f"Carregando dados para {ticker}...")
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if data.empty:
                st.error(f"Nenhum dado encontrado para o ticker {ticker} no período de {start_date} a {end_date}.")
                data_load_state.text("")
                return
            data_load_state.text(f"Dados para {ticker} carregados com sucesso!")
        except Exception as e:
            st.error(f"Erro ao baixar os dados para {ticker}: {e}")
            data_load_state.text("")
            return

        data_viz = data[['Close']].copy()
        data_viz['SMA_200'] = data_viz['Close'].rolling(window=200, min_periods=1).mean()
        data_viz['EMA_50'] = data_viz['Close'].ewm(span=50, adjust=False, min_periods=1).mean()

        # Calcular o índice de Hurst (Linha 54 no código original do usuário)
        hurst_series = calculate_hurst_series_func(data_viz['Close'], window_size)

        if hurst_series.empty or hurst_series.isna().all():
            st.warning(f"Não foi possível calcular o índice de Hurst para {ticker} com janela de {window_size} períodos. Verifique os dados ou o tamanho da janela.")
            fig, ax = plt.subplots(figsize=(16, 5))
            ax.plot(data_viz.index, data_viz['Close'], label="Preço de Fechamento", color='blue', linewidth=1.5)
            ax.plot(data_viz.index, data_viz['SMA_200'], label="SMA 200 Períodos", color='orange', linestyle='--', linewidth=1.5)
            ax.plot(data_viz.index, data_viz['EMA_50'], label="EMA 50 Períodos", color='purple', linestyle='--', linewidth=1.5)
            ax.set_title(f"Preço de Fechamento, SMA 200 e EMA 50 para {ticker}", fontsize=14)
            ax.set_xlabel("Data", fontsize=12)
            ax.set_ylabel("Preço", fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, len(data_viz.index)//10)))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            st.pyplot(fig)
            plt.close(fig)
            return

        common_index = data_viz.index.intersection(hurst_series.index)
        if common_index.empty:
            st.error("Erro ao alinhar os dados de preço e Hurst. Não há datas em comum.")
            return
            
        data_viz_aligned = data_viz.loc[common_index]
        hurst_series_final = hurst_series.loc[common_index]

        if data_viz_aligned.empty or hurst_series_final.empty:
            st.error("Após o alinhamento, as séries de dados de preço ou Hurst estão vazias.")
            return

        st.subheader("Gráficos de Análise")
        fig, axes = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'hspace': 0.45})

        axes[0].plot(data_viz_aligned.index, data_viz_aligned['Close'], label="Preço", color='blue', linewidth=1.5)
        axes[0].plot(data_viz_aligned.index, data_viz_aligned['SMA_200'], label="SMA 200", color='orange', linestyle='--', linewidth=1.5)
        axes[0].plot(data_viz_aligned.index, data_viz_aligned['EMA_50'], label="EMA 50", color='purple', linestyle='--', linewidth=1.5)

        min_price_plot = data_viz_aligned['Close'].min()
        max_price_plot = data_viz_aligned['Close'].max()

        axes[0].fill_between(hurst_series_final.index, min_price_plot, max_price_plot, where=(hurst_series_final > 0.5), color='green', alpha=0.2, label="Regime de Tendência (H > 0.5)")
        axes[0].fill_between(hurst_series_final.index, min_price_plot, max_price_plot, where=(hurst_series_final <= 0.5), color='red', alpha=0.2, label="Regime de Reversão à Média (H <= 0.5)")

        axes[0].set_title(f"Preço ({ticker}), Médias Móveis e Regimes de Hurst", fontsize=14)
        axes[0].set_xlabel("Data", fontsize=12)
        axes[0].set_ylabel("Preço", fontsize=12)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        axes[0].xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, len(data_viz_aligned.index)//10)))
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')

        axes[1].plot(hurst_series_final.index, hurst_series_final, label="Índice de Hurst", color='saddlebrown', linewidth=1.5)
        axes[1].axhline(0.5, color='black', linestyle='--', linewidth=1.2, label="Limite H = 0.5")

        axes[1].fill_between(hurst_series_final.index, 0, hurst_series_final, where=(hurst_series_final > 0.5), color='green', alpha=0.2)
        axes[1].fill_between(hurst_series_final.index, 0, hurst_series_final, where=(hurst_series_final <= 0.5), color='red', alpha=0.2)
        
        axes[1].set_title(f"Índice de Hurst ({ticker} - Janela de {window_size} Períodos)", fontsize=14)
        axes[1].set_xlabel("Data", fontsize=12)
        axes[1].set_ylabel("Índice de Hurst", fontsize=12)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, len(hurst_series_final.index)//10)))
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout(pad=2.0)
        st.pyplot(fig)
        plt.close(fig)

        if st.checkbox("Mostrar dados tabulares (preço, médias, Hurst)"):
            display_data = data_viz_aligned.copy()
            display_data['Hurst'] = hurst_series_final
            st.dataframe(display_data.tail(100))
    else:
        st.info("Ajuste os parâmetros na barra lateral e clique em 'Analisar' para gerar os gráficos.")

if __name__ == '__main__':
    main()

