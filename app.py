import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from hurst import compute_Hc
from PIL import Image # Para a logo

# Fun

o para calcular o 
dice de Hurst com uma janela deslizante
def calculate_hurst_series_func(series_data, window_size_hurst):
    hurst_values = []
    series_data_clean = pd.Series(series_data).dropna()
    if len(series_data_clean) < window_size_hurst:
        return pd.Series(dtype=float)

    for i in range(window_size_hurst, len(series_data_clean)):
        window_data = series_data_clean[i-window_size_hurst:i]
        if len(window_data) < 2: # compute_Hc precisa de pelo menos 2 pontos
            hurst_values.append(np.nan)
            continue
        try:
            H, _, _ = compute_Hc(window_data, kind='random_walk', simplified=True)
            hurst_values.append(H)
        except Exception: 
            hurst_values.append(np.nan)
            
    if hurst_values:
        return pd.Series(hurst_values, index=series_data_clean.index[window_size_hurst:])
    else:
        return pd.Series(dtype=float)

# Fun

o principal da aplica

o
def main():
    st.set_page_config(layout="wide")

    # T
tulo e Logo
    try:
        # O usu
rio precisa fornecer este arquivo na mesma pasta do app.py
        logo = Image.open("Hurst_time_series.png") 
        col1, col2 = st.columns([1, 6]) 
        with col1:
            st.image(logo, width=100) 
        with col2:
            st.title("An
lise de regress
o 
 m
dia ou continuidade")
    except FileNotFoundError:
        st.title("An
lise de regress
o 
 m
dia ou continuidade")
        st.warning("Arquivo da logo 'Hurst_time_series.png' n
o encontrado. Coloque o arquivo na mesma pasta do app.py.")

    # Par
metros de entrada (podem ser widgets do Streamlit no futuro)
    ticker_default = "btc-usd"
    start_date_default = "2023-01-01"
    end_date_default = pd.to_datetime("today").strftime("%Y-%m-%d") # Usar data de hoje como padr
o
    window_size_default = 100

    st.sidebar.header("Par
metros de An
lise")
    ticker = st.sidebar.text_input("Ticker (ex: btc-usd, ^BVSP)", ticker_default)
    start_date = st.sidebar.text_input("Data de In
cio (YYYY-MM-DD)", start_date_default)
    end_date = st.sidebar.text_input("Data Final (YYYY-MM-DD)", end_date_default)
    window_size = st.sidebar.number_input("Janela do Hurst (per
odos)", min_value=10, max_value=500, value=window_size_default)

    if st.sidebar.button("Analisar"):
        data_load_state = st.text(f"Carregando dados para {ticker}...")
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            if data.empty:
                st.error(f"Nenhum dado encontrado para o ticker {ticker} no per
odo especificado.")
                data_load_state.text("")
                return
            data_load_state.text(f"Dados para {ticker} carregados!")
        except Exception as e:
            st.error(f"Erro ao baixar os dados: {e}")
            data_load_state.text("")
            return

        # Trabalhar com uma c
pia para visualiza

o e c
lculos
        data_viz = data[['Close']].copy()
        data_viz['SMA_200'] = data_viz['Close'].rolling(window=200, min_periods=1).mean()
        data_viz['EMA_50'] = data_viz['Close'].ewm(span=50, adjust=False, min_periods=1).mean()

        # Calcular o 
dice de Hurst
        hurst_series = calculate_hurst_series_func(data['Close'], window_size)

        if hurst_series.empty:
            st.warning(f"N
o foi poss
vel calcular o 
dice de Hurst. Verifique os dados ou o tamanho da janela ({window_size}).")
            # Plotar apenas o gr
fico de pre
os se Hurst n
o puder ser calculado
            fig, ax = plt.subplots(figsize=(16, 5))
            ax.plot(data_viz.index, data_viz['Close'], label="Pre
o", color='blue', linewidth=1.5)
            ax.plot(data_viz.index, data_viz['SMA_200'], label="SMA 200", color='orange', linestyle='--', linewidth=1.5)
            ax.plot(data_viz.index, data_viz['EMA_50'], label="EMA 50", color='purple', linestyle='--', linewidth=1.5)
            ax.set_title("Pre
o com SMA e EMA", fontsize=14)
            ax.set_xlabel("Data", fontsize=12)
            ax.set_ylabel("Pre
o", fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(alpha=0.5)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            st.pyplot(fig)
            return

        # Alinhar 
dices para plotagem
        common_index = data_viz.index.intersection(hurst_series.index)
        if common_index.empty:
            st.error("N
o foi poss
vel alinhar os dados de pre
o e Hurst. Verifique as datas e os dados.")
            return
            
        data_viz_aligned = data_viz.loc[common_index]
        hurst_series_final = hurst_series.loc[common_index]

        if data_viz_aligned.empty or hurst_series_final.empty:
            st.error("Dados alinhados resultaram em s
ries vazias. N
o 
 poss
vel gerar os gr
ficos.")
            return

        st.subheader("Gr
ficos de An
lise")
        fig, axes = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'hspace': 0.4}, constrained_layout=False)

        # Gr
fico do pre
o e m
dias m
veis
        axes[0].plot(data_viz_aligned.index, data_viz_aligned['Close'], label="Pre
o", color='blue', linewidth=1.5)
        axes[0].plot(data_viz_aligned.index, data_viz_aligned['SMA_200'], label="SMA 200", color='orange', linestyle='--', linewidth=1.5)
        axes[0].plot(data_viz_aligned.index, data_viz_aligned['EMA_50'], label="EMA 50", color='purple', linestyle='--', linewidth=1.5)

        min_price_plot = data_viz_aligned['Close'].min()
        max_price_plot = data_viz_aligned['Close'].max()

        axes[0].fill_between(
            hurst_series_final.index,
            min_price_plot,
            max_price_plot,
            where=(hurst_series_final > 0.5),
            color='green',
            alpha=0.2,
            label="Trend Following (H > 0.5)"
        )
        axes[0].fill_between(
            hurst_series_final.index,
            min_price_plot,
            max_price_plot,
            where=(hurst_series_final < 0.5),
            color='red',
            alpha=0.2,
            label="Mean Reversion (H < 0.5)"
        )

        axes[0].set_title("Pre
o com SMA, EMA e Regi
es Estrat
gicas de Hurst", fontsize=14)
        axes[0].set_xlabel("Data", fontsize=12)
        axes[0].set_ylabel("Pre
o", fontsize=12)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        axes[0].xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, len(data_viz_aligned)//200))) # Ajuste din
mico do locator
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Gr
fico do 
dice de Hurst
        axes[1].plot(hurst_series_final.index, hurst_series_final, label="
dice de Hurst", color='saddlebrown', linewidth=1.5)
        axes[1].axhline(0.5, color='black', linestyle='--', linewidth=1.2, label="Limite H = 0.5")

        axes[1].fill_between(
            hurst_series_final.index,
            0,
            hurst_series_final,
            where=(hurst_series_final > 0.5),
            color='green',
            alpha=0.2
        )
        axes[1].fill_between(
            hurst_series_final.index,
            0,
            hurst_series_final,
            where=(hurst_series_final < 0.5),
            color='red',
            alpha=0.2
        )
        
        axes[1].set_title(f"
dice de Hurst (Janela de {window_size} Per
odos)", fontsize=14)
        axes[1].set_xlabel("Data", fontsize=12)
        axes[1].set_ylabel("
dice de Hurst", fontsize=12)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, len(hurst_series_final)//200))) # Ajuste din
mico
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout(pad=2.0)
        st.pyplot(fig)

        # Exibir dados tabulares (opcional)
        if st.checkbox("Mostrar dados tabulares (pre
o, m
dias, Hurst)"):
            display_data = data_viz_aligned.copy()
            display_data['Hurst'] = hurst_series_final
            st.dataframe(display_data.tail(100)) # Mostra as 
ltimas 100 linhas
    else:
        st.info("Ajuste os par
metros na barra lateral e clique em 'Analisar' para gerar os gr
ficos.")

if __name__ == '__main__':
    main()

