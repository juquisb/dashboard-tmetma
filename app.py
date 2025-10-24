import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime as dt

# --- Configurações e Lógica de Processamento de Dados (Adaptada da Classe Original) ---

FILAS_MONITORADAS = [
    'Acolhimento',
    'Tri-agora',
    'Tri-algumas_horas',
    'plantao',
    'Parcerias',
    'N2-Administrativo'
]

METAS = {
    'Acolhimento': {'TME': 3.0, 'TMA': 30.0},
    'Tri-agora': {'TME': 5.0, 'TMA': 60.0},
    'Tri-algumas_horas': {'TME': 60.0, 'TMA': 120.0},
    'plantao': {'TME': 5.0, 'TMA': 60.0},
    'Parcerias': {'TME': 3.0, 'TMA': 30.0},
    'N2-Administrativo': {'TME': 3.0, 'TMA': 30.0}
}

def converter_tempo_para_minutos(tempo_obj):
    """Converte tempo (HH:MM:SS, datetime.time, ou número) para minutos decimais."""
    try:
        if pd.isna(tempo_obj):
            return 0.0
        if isinstance(tempo_obj, (int, float)):
            return float(tempo_obj)
        if isinstance(tempo_obj, dt.time):
            return tempo_obj.hour * 60 + tempo_obj.minute + tempo_obj.second / 60
        if isinstance(tempo_obj, str) and ':' in tempo_obj:
            h, m, s = [int(x) for x in tempo_obj.split(':')]
            return h * 60 + m + s / 60
        # Tenta converter datetime.timedelta para minutos
        if isinstance(tempo_obj, dt.timedelta):
            return tempo_obj.total_seconds() / 60
        
        # Tenta converter string para timedelta se for o caso
        if isinstance(tempo_obj, str):
            # Tenta parsear como HH:MM:SS
            parts = tempo_obj.split(':')
            if len(parts) == 3:
                h, m, s = [int(x) for x in parts]
                return h * 60 + m + s / 60
            # Tenta parsear como número (minutos)
            return float(tempo_obj)

        return float(tempo_obj)
    except Exception:
        return 0.0

def formatar_tempo_hhmmss(minutos):
    """Converte minutos decimais para o formato hh:mm:ss."""
    if minutos <= 0 or pd.isna(minutos):
        return "00:00:00"
    total_segundos = int(round(minutos * 60))
    horas = total_segundos // 3600
    minutos_restantes = (total_segundos % 3600) // 60
    segundos = total_segundos % 60
    return f"{horas:02d}:{minutos_restantes:02d}:{segundos:02d}"

@st.cache_data
def ler_e_processar_dados(uploaded_file):
    """Lê o arquivo Excel e calcula as médias de TME/TMA por data e fila."""
    try:
        df = pd.read_excel(uploaded_file)
        
        # Limpeza e conversão de colunas
        if 'Inicio da ação' not in df.columns:
            st.error("Coluna 'Inicio da ação' não encontrada no arquivo.")
            return None
        if 'Fila' not in df.columns:
            st.error("Coluna 'Fila' não encontrada no arquivo.")
            return None
        if 'Tempo na Fila' not in df.columns:
            st.error("Coluna 'Tempo na Fila' não encontrada no arquivo.")
            return None
        if 'Tempo de atendimento' not in df.columns:
            st.error("Coluna 'Tempo de atendimento' não encontrada no arquivo.")
            return None

        df['Inicio da ação'] = pd.to_datetime(df['Inicio da ação'], errors='coerce')
        df = df.dropna(subset=['Inicio da ação'])
        df['Data'] = df['Inicio da ação'].dt.date

        # Aplicar a conversão para minutos
        df['TME_minutos'] = df['Tempo na Fila'].apply(converter_tempo_para_minutos)
        df['TMA_minutos'] = df['Tempo de atendimento'].apply(converter_tempo_para_minutos)

        # Agrupamento e cálculo das médias
        resultados_df = df.groupby(['Data', 'Fila']).agg(
            TME=('TME_minutos', 'mean'),
            TMA=('TMA_minutos', 'mean'),
            volume=('Fila', 'size')
        ).reset_index()
        
        return resultados_df

    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")
        return None

def gerar_grafico_tme_tma_plotly(df_fila, fila):
    """Gera o gráfico de TME e TMA usando Plotly com dois eixos Y."""
    
    # Cores
    cor_tme = '#1f77b4'  # Azul
    cor_tma = '#ff7f0e'  # Laranja
    
    meta_tme_valor = METAS[fila]['TME']
    meta_tma_valor = METAS[fila]['TMA']

    # Criar figura com subplots e dois eixos Y
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # --- Plotar TME (Eixo Y Primário - Esquerda) ---
    fig.add_trace(
        go.Scatter(
            x=df_fila['Data'], 
            y=df_fila['TME'], 
            mode='lines+markers+text', 
            name='TME Real (min)',
            marker=dict(color=cor_tme, size=10, symbol='circle'),
            line=dict(width=4),
            text=[formatar_tempo_hhmmss(tme) for tme in df_fila['TME']],
            textposition="top center",
            textfont=dict(color=cor_tme, size=11, weight='bold'),
        ),
        secondary_y=False,
    )

    # Linha da Meta TME
    fig.add_trace(
        go.Scatter(
            x=df_fila['Data'], 
            y=[meta_tme_valor] * len(df_fila), 
            mode='lines', 
            name=f'Meta TME ({formatar_tempo_hhmmss(meta_tme_valor)})',
            line=dict(color=cor_tme, dash='dash', width=2.5),
            hoverinfo='skip'
        ),
        secondary_y=False,
    )

    # --- Plotar TMA (Eixo Y Secundário - Direita) ---
    fig.add_trace(
        go.Scatter(
            x=df_fila['Data'], 
            y=df_fila['TMA'], 
            mode='lines+markers+text', 
            name='TMA Real (min)',
            marker=dict(color=cor_tma, size=10, symbol='square'),
            line=dict(width=4),
            text=[formatar_tempo_hhmmss(tma) for tma in df_fila['TMA']],
            textposition="top center",
            textfont=dict(color=cor_tma, size=11, weight='bold'),
        ),
        secondary_y=True,
    )

    # Linha da Meta TMA
    fig.add_trace(
        go.Scatter(
            x=df_fila['Data'], 
            y=[meta_tma_valor] * len(df_fila), 
            mode='lines', 
            name=f'Meta TMA ({formatar_tempo_hhmmss(meta_tma_valor)})',
            line=dict(color=cor_tma, dash='dash', width=2.5),
            hoverinfo='skip'
        ),
        secondary_y=True,
    )

    # Configurações de Layout ---
    fig.update_layout(
        title_text=f"<b>{fila.upper()}</b> - Desempenho TME e TMA",
        title_font_size=20,
        title_font_color='#1f1f1f',  # Cor escura para o título
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(t=100),
        template='plotly_white',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#1f1f1f')  # Cor padrão para todo o texto do gráfico
    )
    
    # Eixo X
    fig.update_xaxes(
        title_text="Data",
        title_font_color='#1f1f1f',  # Cor do título do eixo X
        tickfont_color='#1f1f1f',    # Cor dos ticks do eixo X
        tickformat="%d/%b",
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(220, 220, 220, 0.8)',
        linecolor='#e1e1e1',
        zerolinecolor='#e1e1e1'
    )

    # Eixo Y Primário (TME)
    fig.update_yaxes(
        title_text="TME (Tempo Médio de Espera) - Minutos",
        title_font_color='#1f1f1f',  # Cor do título do eixo Y
        tickfont_color='#1f1f1f',    # Cor dos ticks do eixo Y
        secondary_y=False, 
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(220, 220, 220, 0.8)',
        linecolor='#e1e1e1',
        zerolinecolor='#e1e1e1',
        range=[0, max(df_fila['TME'].max() * 1.15 if not df_fila['TME'].empty else 0, meta_tme_valor * 1.2)]
    )

    # Eixo Y Secundário (TMA)
    fig.update_yaxes(
        title_text="TMA (Tempo Médio de Atendimento) - Minutos",
        title_font_color='#1f1f1f',  # Cor do título do eixo Y secundário
        tickfont_color='#1f1f1f',    # Cor dos ticks do eixo Y secundário
        secondary_y=True, 
        showgrid=False,
        range=[0, max(df_fila['TMA'].max() * 1.15 if not df_fila['TMA'].empty else 0, meta_tma_valor * 1.2)]
    )
    
    return fig

# --- Aplicação Streamlit Principal ---

def main():
    st.set_page_config(
        page_title="Dashboard TME/TMA",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # CSS personalizado para harmonizar o tema
    st.markdown("""
        <style>
        .main {
            background-color: #ffffff;
        }
        .stApp {
            background-color: #f8f9fa;
        }
        .css-18e3th9 {
            padding-top: 2rem;
            padding-bottom: 2rem;
            background-color: #f8f9fa;
        }
        .stSidebar {
            background-color: #f0f2f6;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #1f1f1f;
        }
        .stMarkdown {
            color: #1f1f1f;
        }
        div[data-testid="stMetricValue"] {
            color: #1f1f1f;
        }
        </style>
        """, unsafe_allow_html=True)

    st.title("📊 Dashboard de Desempenho TME/TMA")
    st.markdown("---")

    # 1. Upload de Arquivo
    with st.sidebar:
        st.header("Upload de Dados")
        uploaded_file = st.file_uploader(
            "Selecione o arquivo Excel (com colunas 'Inicio da ação', 'Fila', 'Tempo na Fila', 'Tempo de atendimento')", 
            type=['xlsx']
        )
        
        if uploaded_file is None:
            st.info("Aguardando o upload do arquivo Excel.")
            return

    # 2. Processamento dos Dados
    with st.spinner('Processando dados...'):
        df_processado = ler_e_processar_dados(uploaded_file)
    
    if df_processado is None or df_processado.empty:
        st.warning("Nenhum dado válido encontrado após o processamento.")
        return

    # 3. Filtro de Filas
    filas_disponiveis = df_processado['Fila'].unique().tolist()
    filas_para_exibir = [f for f in FILAS_MONITORADAS if f in filas_disponiveis]
    
    if not filas_para_exibir:
        st.warning("Nenhuma das filas monitoradas foi encontrada nos dados.")
        st.dataframe(df_processado['Fila'].unique())
        return

    # 4. Exibição dos Gráficos
    st.header("Visualização por Fila")

    for fila in filas_para_exibir:
        df_fila = df_processado[df_processado['Fila'] == fila].sort_values(by='Data')
        
        if df_fila.empty:
            continue

        st.subheader(f"Fila: {fila}")
        
        # Gráfico TME/TMA (Eixos Duplos)
        fig_tme_tma = gerar_grafico_tme_tma_plotly(df_fila, fila)
        st.plotly_chart(fig_tme_tma, use_container_width=True)
        
        # Exibir volume total abaixo do gráfico principal
        volume_total = df_fila['volume'].sum()
        st.markdown(f"**Volume Total de Atendimentos:** {volume_total}")
            
        st.markdown("---")

if __name__ == "__main__":
    main()
