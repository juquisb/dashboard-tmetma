import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime as dt
from typing import Optional
import numpy as np

# ----------------------------
# Configurações iniciais
# ----------------------------
st.set_page_config(
    page_title="Dashboard TME/TMA",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Dashboard de Monitoramento TME/TMA v2.0"
    }
)

# Filas monitoradas e metas (minutos)
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
    'N2-Administrativo': {'TME': 30.0, 'TMA': 120.0}
}

# Paleta de cores profissional
CORES = {
    'primary': '#2E5BFF',
    'success': '#00C48C',
    'warning': '#FFA756',
    'danger': '#FF647C',
    'info': '#00D4FF',
    'dark': '#1E293B',
    'light': '#F8FAFC',
    'muted': '#64748B'
}

# ----------------------------
# Funções utilitárias
# ----------------------------
def converter_tempo_para_minutos(tempo_obj) -> float:
    """Converte tempo (HH:MM:SS, datetime.time, timedelta, ou número) para minutos decimais."""
    try:
        if pd.isna(tempo_obj):
            return 0.0
        if isinstance(tempo_obj, (int, float)):
            return float(tempo_obj)
        if isinstance(tempo_obj, dt.time):
            return tempo_obj.hour * 60 + tempo_obj.minute + tempo_obj.second / 60
        if isinstance(tempo_obj, dt.timedelta):
            return tempo_obj.total_seconds() / 60
        if isinstance(tempo_obj, str):
            parts = tempo_obj.strip().split(':')
            if len(parts) == 3:
                h, m, s = [int(x) for x in parts]
                return h * 60 + m + s / 60
            if len(parts) == 2:
                m, s = [int(x) for x in parts]
                return m + s / 60
            return float(tempo_obj.replace(',', '.'))
        return float(tempo_obj)
    except Exception:
        return 0.0

def formatar_tempo_hhmmss(minutos: float) -> str:
    """Converte minutos decimais para o formato hh:mm:ss."""
    try:
        if minutos is None or pd.isna(minutos) or minutos <= 0:
            return "00:00:00"
        total_segundos = int(round(minutos * 60))
        horas = total_segundos // 3600
        minutos_restantes = (total_segundos % 3600) // 60
        segundos = total_segundos % 60
        return f"{horas:02d}:{minutos_restantes:02d}:{segundos:02d}"
    except Exception:
        return "00:00:00"

@st.cache_data
def ler_e_processar_dados(uploaded_file) -> Optional[pd.DataFrame]:
    """Lê o arquivo Excel e calcula TME/TMA em minutos por Data e Fila."""
    try:
        df = pd.read_excel(uploaded_file)
        required = ['Inicio da ação', 'Fila', 'Tempo na Fila', 'Tempo de atendimento']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Coluna obrigatória não encontrada: '{col}'")

        df['Inicio da ação'] = pd.to_datetime(df['Inicio da ação'], errors='coerce')
        df = df.dropna(subset=['Inicio da ação'])
        df['Data'] = pd.to_datetime(df['Inicio da ação']).dt.normalize()  # garante datetime (00:00:00)

        df['TME_minutos'] = df['Tempo na Fila'].apply(converter_tempo_para_minutos)
        df['TMA_minutos'] = df['Tempo de atendimento'].apply(converter_tempo_para_minutos)

        resultados = df.groupby(['Data', 'Fila']).agg(
            TME=('TME_minutos', 'mean'),
            TMA=('TMA_minutos', 'mean'),
            volume=('Fila', 'size')
        ).reset_index()

        resultados = resultados.sort_values(by=['Data'])
        return resultados
    except Exception as e:
        st.error(f"❌ Erro ao processar o arquivo: {e}")
        return None

def cores_card(valor: float, meta: float) -> str:
    """Retorna cor para o card com base na comparação valor vs meta."""
    try:
        if valor <= 0:
            return CORES['muted']
        return CORES['success'] if valor <= meta else CORES['danger']
    except Exception:
        return CORES['muted']

def agregar_metricas(df: pd.DataFrame) -> dict:
    """Calcula métricas agregadas a partir do df filtrado."""
    resultado = {}
    if df.empty:
        resultado['TME_medio'] = 0.0
        resultado['TMA_medio'] = 0.0
        resultado['volume_total'] = 0
        return resultado
    resultado['TME_medio'] = float(df['TME'].mean())
    resultado['TMA_medio'] = float(df['TMA'].mean())
    resultado['volume_total'] = int(df['volume'].sum())
    return resultado

def gerar_grafico_tme_tma_plotly(df_fila: pd.DataFrame, filas_selecionadas: list):
    """Gera gráfico Plotly com:
       - TME: linha tracejada
       - TMA: linha SOLIDA (cor distinta) + marcadores coloridos por meta
       - Axis ticks em hh:mm:ss (garantindo inclusão das metas)
       - Rótulos e hover em hh:mm:ss
    """
    paleta = [CORES['primary'], CORES['warning'], CORES['success'], '#8B5CF6', '#EC4899', '#14B8A6']
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # metas selecionadas (usadas também para garantir que as linhas de meta estejam visíveis)
    metas_tme = [METAS.get(fila, {'TME': 0})['TME'] for fila in filas_selecionadas if fila in METAS]
    metas_tma = [METAS.get(fila, {'TMA': 0})['TMA'] for fila in filas_selecionadas if fila in METAS]

    # coletar todos os valores para definir ticks adequados no eixo (inclui metas)
    valores_minutos = []
    for fila in filas_selecionadas:
        df_tmp = df_fila[df_fila['Fila'] == fila]
        if df_tmp.empty:
            continue
        valores_minutos.extend(df_tmp['TME'].tolist())
        valores_minutos.extend(df_tmp['TMA'].tolist())
    # incluir metas na escala para garantir visibilidade das linhas de meta
    valores_minutos.extend(metas_tme)
    valores_minutos.extend(metas_tma)

    # função interna para gerar ticks em minutos e labels em hh:mm:ss
    def gerar_ticks_em_minutos(min_val, max_val):
        if min_val is None or max_val is None or np.isnan(min_val) or np.isnan(max_val):
            return None, None
        span = max_val - min_val
        # escolher passo conforme o span
        if span <= 30:
            step = 5
        elif span <= 120:
            step = 10
        elif span <= 360:
            step = 30
        else:
            step = 60
        low = max(0, int(np.floor(min_val / step)) * step)
        high = int(np.ceil(max_val / step)) * step
        tickvals = list(range(low, high + 1, step))
        ticktext = [formatar_tempo_hhmmss(v) for v in tickvals]
        return tickvals, ticktext

    # loop por fila (desenha TME e TMA)
    for i, fila in enumerate(filas_selecionadas):
        df_f = df_fila[df_fila['Fila'] == fila].copy().sort_values('Data')
        if df_f.empty:
            continue

        cor_tme = paleta[i % len(paleta)]
        # cor da linha do TMA (sólida) — escolha distinta para contraste
        cor_tma_base = '#0f172a'  # tom escuro para destacar a linha sólida do TMA

        meta_tma = METAS.get(fila, {'TMA': 0})['TMA']

        # TME: linha tracejada, rótulos em hh:mm:ss
        fig.add_trace(
            go.Scatter(
                x=df_f['Data'],
                y=df_f['TME'],
                mode='lines+markers+text',
                name=f"{fila} - TME",
                marker=dict(symbol='circle', size=8, color=cor_tme),
                line=dict(width=2, dash='dash', color=cor_tme),
                text=[formatar_tempo_hhmmss(v) for v in df_f['TME']],
                textposition='top center',
                textfont=dict(size=10, color=cor_tme, family="Inter, sans-serif"),
                hovertemplate="<b>%{x|%d/%m/%Y}</b><br>TME: %{y:.2f} min<br>Tempo: %{customdata}<extra></extra>",
                customdata=[formatar_tempo_hhmmss(v) for v in df_f['TME']]
            ),
            secondary_y=False
        )

        # TMA: linha SOLIDA (cor base) + marcadores coloridos por meta (verde/vermelho)
        tma_marker_colors = [
            CORES['success'] if (v is not None and not pd.isna(v) and v <= meta_tma) else CORES['danger']
            for v in df_f['TMA']
        ]

        fig.add_trace(
            go.Scatter(
                x=df_f['Data'],
                y=df_f['TMA'],
                mode='lines+markers+text',
                name=f"{fila} - TMA",
                marker=dict(symbol='square', size=9, color=tma_marker_colors, line=dict(width=1, color='rgba(0,0,0,0.08)')),
                line=dict(width=3, dash='solid', color=cor_tma_base),  # <-- SOLIDA garantida aqui
                text=[formatar_tempo_hhmmss(v) for v in df_f['TMA']],
                textposition='bottom center',
                textfont=dict(size=10, color=cor_tma_base, family="Inter, sans-serif"),
                hovertemplate="<b>%{x|%d/%m/%Y}</b><br>TMA: %{y:.2f} min<br>Tempo: %{customdata}<extra></extra>",
                customdata=[formatar_tempo_hhmmss(v) for v in df_f['TMA']]
            ),
            secondary_y=True
        )

    # adicionar linhas de meta (usa o menor meta entre as filas selecionadas)
    # pegar todas as datas presentes no df_fila para desenhar a reta de meta
    all_dates = sorted(df_fila['Data'].unique()) if not df_fila.empty else []

    if all_dates:
        if metas_tme:
            meta_tme_min = min(metas_tme)
            fig.add_trace(
                go.Scatter(x=all_dates, y=[meta_tme_min] * len(all_dates), mode='lines',
                           name=f"Meta TME ({meta_tme_min:.0f}m)",
                           line=dict(color=CORES['info'], dash='dot', width=2),
                           hovertemplate="Meta TME: %{y:.2f} min<extra></extra>"),
                secondary_y=False
            )
        if metas_tma:
            meta_tma_min = min(metas_tma)
            fig.add_trace(
                go.Scatter(x=all_dates, y=[meta_tma_min] * len(all_dates), mode='lines',
                           name=f"Meta TMA ({meta_tma_min:.0f}m)",
                           line=dict(color=CORES['warning'], dash='dot', width=2),
                           hovertemplate="Meta TMA: %{y:.2f} min<extra></extra>"),
                secondary_y=True
            )

    fig.update_layout(
        title=None,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.05,
            xanchor='right',
            x=1,
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='#CBD5E1',
            borderwidth=1,
            font=dict(size=13, color='#1E293B', family="Inter, sans-serif")
        ),
        margin=dict(t=80, b=70, l=80, r=80),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Inter, system-ui, -apple-system, sans-serif", size=13, color='#1E293B'),
        height=520
    )

    # configurar ticks dos eixos Y para mostrar hh:mm:ss (garantindo que as metas estejam incluídas)
    if valores_minutos:
        valores_clean = [v for v in valores_minutos if v is not None and not pd.isna(v)]
        if valores_clean:
            mn = float(np.nanmin(valores_clean))
            mx = float(np.nanmax(valores_clean))
            tickvals, ticktext = gerar_ticks_em_minutos_local(mn, mx)
            if tickvals and ticktext:
                fig.update_yaxes(
                    title_text="<b>TME (hh:mm:ss)</b>",
                    tickvals=tickvals,
                    ticktext=ticktext,
                    tickfont=dict(size=12, color='#475569'),
                    secondary_y=False,
                    showgrid=True,
                    gridcolor='#E2E8F0',
                    gridwidth=1,
                    zeroline=False,
                    showline=True,
                    linewidth=1,
                    linecolor='#CBD5E1'
                )
                fig.update_yaxes(
                    title_text="<b>TMA (hh:mm:ss)</b>",
                    tickvals=tickvals,
                    ticktext=ticktext,
                    tickfont=dict(size=12, color='#475569'),
                    secondary_y=True,
                    showgrid=False,
                    zeroline=False,
                    showline=True,
                    linewidth=1,
                    linecolor='#CBD5E1'
                )
    else:
        fig.update_yaxes(secondary_y=False)
        fig.update_yaxes(secondary_y=True)

    # eixo X
    fig.update_xaxes(
        title_text="<b>Data</b>",
        title_font=dict(size=14, color='#1E293B', family="Inter, sans-serif"),
        tickformat="%d/%b",
        tickfont=dict(size=12, color='#475569'),
        showgrid=True,
        gridcolor='#E2E8F0',
        gridwidth=1,
        zeroline=False,
        showline=True,
        linewidth=1,
        linecolor='#CBD5E1'
    )

    return fig

# NOTE: mover a função gerar_ticks_em_minutos para cima do uso (declaro aqui para evitar erro)
def gerar_ticks_em_minutos_local(min_val, max_val):
    import math
    if min_val is None or max_val is None or np.isnan(min_val) or np.isnan(max_val):
        return None, None
    span = max_val - min_val
    if span <= 30:
        step = 5
    elif span <= 120:
        step = 10
    elif span <= 360:
        step = 30
    else:
        step = 60
    low = max(0, int(np.floor(min_val / step)) * step)
    high = int(np.ceil(max_val / step)) * step
    tickvals = list(range(low, high + 1, step))
    ticktext = [formatar_tempo_hhmmss(v) for v in tickvals]
    return tickvals, ticktext

# ----------------------------
# CSS customizado - Visual Profissional
# ----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Reset e configurações gerais */
* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* Container principal */
.main {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    background-attachment: fixed;
}

div.block-container {
    max-width: 1400px;
    padding: 2rem 1.5rem;
    background: transparent;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    border-right: none;
}

[data-testid="stSidebar"] * {
    color: #F1F5F9 !important;
}

/* Headers */
h1, h2, h3 {
    color: white !important;
    font-weight: 700;
    text-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

/* Card de métricas moderno */
.metric-card {
    background: white;
    padding: 1.5rem;
    border-radius: 16px;
    border: none;
    box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 4px;
    height: 100%;
    background: currentColor;
}

.metric-label {
    font-size: 0.875rem;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.5rem;
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #1E293B;
    line-height: 1.2;
}

.metric-subtitle {
    font-size: 0.875rem;
    color: #64748B;
    margin-top: 0.5rem;
}

/* Gráficos com visual melhorado */
.stPlotlyChart {
    background: white;
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    margin: 1.5rem 0;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    background: white;
    border-radius: 16px;
    padding: 1rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.15);
}

/* Footer */
.footer-text {
    background: rgba(255,255,255,0.1);
    color: white;
    padding: 1rem;
    border-radius: 12px;
    text-align: center;
    font-size: 0.875rem;
    backdrop-filter: blur(10px);
}

/* Download button */
.stDownloadButton button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    transition: all 0.3s;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------
# Sidebar: Upload e controles
# ----------------------------
with st.sidebar:
    st.title("⚙️ Configurações")

    st.markdown("**📤 Upload do arquivo**")
    uploaded_file = st.file_uploader(
        "Selecione o arquivo Excel",
        type=['xlsx'],
        help="Arquivo deve conter: 'Inicio da ação', 'Fila', 'Tempo na Fila', 'Tempo de atendimento'"
    )

    st.markdown("---")

    st.markdown("**📊 Navegação**")
    page = st.radio(
        "Escolha a página",
        ["📈 Visão Geral", "🔍 Visualização por Fila", "📁 Upload & Dados"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown("**🎯 Filtros**")
    filas_selecionadas = st.multiselect(
        "Filas",
        options=FILAS_MONITORADAS,
        default=FILAS_MONITORADAS[:1],
        help="Selecione uma ou mais filas para análise"
    )

    hoje = dt.date.today()
    col1, col2 = st.columns(2)
    with col1:
        periodo_inicio = st.date_input(
            "De",
            value=hoje - dt.timedelta(days=14),
            help="Data inicial"
        )
    with col2:
        periodo_fim = st.date_input(
            "Até",
            value=hoje,
            help="Data final"
        )

    if periodo_inicio > periodo_fim:
        st.error("⚠️ Data início > Data fim")

    st.markdown("---")
    st.caption("💡 Selecione múltiplas filas para comparação")

# ----------------------------
# Validação de upload
# ----------------------------
if uploaded_file is None:
    st.title("📊 Dashboard TME/TMA")
    st.markdown("### Bem-vindo ao Dashboard de Monitoramento")

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.info("👈 Faça upload do arquivo Excel na barra lateral para começar")

        st.markdown("""
        **Requisitos do arquivo:**
        - Formato: Excel (.xlsx)
        - Colunas obrigatórias:
          - `Inicio da ação`
          - `Fila`
          - `Tempo na Fila`
          - `Tempo de atendimento`
        """)
    st.stop()

# ----------------------------
# Processamento dos dados
# ----------------------------
with st.spinner("⏳ Processando dados..."):
    df_processado = ler_e_processar_dados(uploaded_file)

if df_processado is None or df_processado.empty:
    st.warning("⚠️ Nenhum dado válido encontrado. Verifique o arquivo.")
    st.stop()

# Aplicar filtros
df_processado['Data'] = pd.to_datetime(df_processado['Data'])
mask_periodo = (df_processado['Data'].dt.date >= periodo_inicio) & (df_processado['Data'].dt.date <= periodo_fim)
df_periodo = df_processado.loc[mask_periodo].copy()

if df_periodo.empty:
    st.warning("⚠️ Sem dados no período selecionado.")
    st.stop()

if not filas_selecionadas:
    filas_selecionadas = sorted(df_periodo['Fila'].unique().tolist())

# ----------------------------
# Páginas
# ----------------------------
page_clean = page.split(" ", 1)[1] if " " in page else page

if page_clean == "Visão Geral":
    st.title("📈 Visão Geral")
    st.markdown("Panorama completo das métricas de atendimento")

    # Cards de métricas
    agregados = agregar_metricas(df_periodo[df_periodo['Fila'].isin(filas_selecionadas)])

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        meta_tme = min([METAS.get(f,{'TME':999})['TME'] for f in filas_selecionadas if f in METAS], default=999)
        cor = cores_card(agregados['TME_medio'], meta_tme)
        st.markdown(f"""
        <div class='metric-card' style='color:{cor}'>
            <div class='metric-label'>⏱️ TME Médio</div>
            <div class='metric-value'>{agregados['TME_medio']:.1f}<span style='font-size:1rem;color:#64748B'> min</span></div>
            <div class='metric-subtitle'>{formatar_tempo_hhmmss(agregados['TME_medio'])}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        meta_tma = min([METAS.get(f,{'TMA':999})['TMA'] for f in filas_selecionadas if f in METAS], default=999)
        cor = cores_card(agregados['TMA_medio'], meta_tma)
        st.markdown(f"""
        <div class='metric-card' style='color:{cor}'>
            <div class='metric-label'>📞 TMA Médio</div>
            <div class='metric-value'>{agregados['TMA_medio']:.1f}<span style='font-size:1rem;color:#64748B'> min</span></div>
            <div class='metric-subtitle'>{formatar_tempo_hhmmss(agregados['TMA_medio'])}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class='metric-card' style='color:{CORES['primary']}'>
            <div class='metric-label'>📊 Volume Total</div>
            <div class='metric-value'>{agregados['volume_total']:,}</div>
            <div class='metric-subtitle'>atendimentos</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class='metric-card' style='color:{CORES['info']}'>
            <div class='metric-label'>🎯 Período</div>
            <div class='metric-value' style='font-size:1rem'>{len(filas_selecionadas)} fila(s)</div>
            <div class='metric-subtitle'>{periodo_inicio.strftime('%d/%m')} — {periodo_fim.strftime('%d/%m/%Y')}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Gráfico principal
    st.markdown("### 📉 Evolução Temporal")
    fig_geral = gerar_grafico_tme_tma_plotly(df_periodo, filas_selecionadas)
    st.plotly_chart(fig_geral, use_container_width=True)

    st.markdown("### 📋 Resumo por Fila")
    resumo_por_fila = df_periodo[df_periodo['Fila'].isin(filas_selecionadas)].groupby('Fila').agg(
        TME_media=('TME', 'mean'),
        TMA_media=('TMA', 'mean'),
        Volume=('volume', 'sum')
    ).reset_index()

    st.dataframe(
        resumo_por_fila.style.format({"TME_media":"{:.2f}", "TMA_media":"{:.2f}"}),
        use_container_width=True,
        height=300
    )

elif page_clean == "Visualização por Fila":
    st.title("🔍 Visualização por Fila")
    st.markdown("Análise detalhada por fila de atendimento")

    filas_para_visual = st.multiselect(
        "Selecione as filas para visualizar",
        options=filas_selecionadas,
        default=filas_selecionadas[:1]
    )

    if not filas_para_visual:
        st.warning("⚠️ Selecione pelo menos uma fila")
        st.stop()

    for idx, fila in enumerate(filas_para_visual):
        if idx > 0:
            st.markdown("<br><br>", unsafe_allow_html=True)

        st.markdown(f"### 📌 {fila}")

        df_fila = df_periodo[df_periodo['Fila'] == fila].sort_values('Data')
        if df_fila.empty:
            st.info("ℹ️ Sem dados para este período")
            continue

        tme_medio = float(df_fila['TME'].mean())
        tma_medio = float(df_fila['TMA'].mean())
        volume = int(df_fila['volume'].sum())
        meta_tme = METAS.get(fila, {'TME': 0})['TME']
        meta_tma = METAS.get(fila, {'TMA': 0})['TMA']

        c1, c2, c3 = st.columns(3)

        with c1:
            cor = cores_card(tme_medio, meta_tme)
            status = "✓" if tme_medio <= meta_tme else "⚠"
            st.markdown(f"""
            <div class='metric-card' style='color:{cor}'>
                <div class='metric-label'>⏱️ TME Médio {status}</div>
                <div class='metric-value'>{tme_medio:.1f}<span style='font-size:1rem;color:#64748B'> min</span></div>
                <div class='metric-subtitle'>Meta: {formatar_tempo_hhmmss(meta_tme)}</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            cor = cores_card(tma_medio, meta_tma)
            status = "✓" if tma_medio <= meta_tma else "⚠"
            st.markdown(f"""
            <div class='metric-card' style='color:{cor}'>
                <div class='metric-label'>📞 TMA Médio {status}</div>
                <div class='metric-value'>{tma_medio:.1f}<span style='font-size:1rem;color:#64748B'> min</span></div>
                <div class='metric-subtitle'>Meta: {formatar_tempo_hhmmss(meta_tma)}</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class='metric-card' style='color:{CORES['primary']}'>
                <div class='metric-label'>📊 Volume</div>
                <div class='metric-value'>{volume:,}</div>
                <div class='metric-subtitle'>atendimentos</div>
            </div>
            """, unsafe_allow_html=True)

        fig_f = gerar_grafico_tme_tma_plotly(df_periodo[df_periodo['Fila'].isin([fila])], [fila])
        st.plotly_chart(fig_f, use_container_width=True)

elif page_clean == "Upload & Dados":
    st.title("📁 Upload & Dados")
    st.markdown("Validação e exportação de dados processados")

    tab1, tab2 = st.tabs(["📄 Amostra de Dados", "📊 Estatísticas"])

    with tab1:
        st.markdown("### Primeiras 200 linhas")
        st.dataframe(df_periodo.head(200), use_container_width=True, height=400)

        st.markdown("### 💾 Download")
        csv = df_periodo.to_csv(index=False).encode('utf-8')
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.download_button(
                "📥 Baixar Dados Processados (CSV)",
                data=csv,
                file_name=f"dados_processados_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

    with tab2:
        st.markdown("### Resumo Estatístico por Fila")
        resumo = df_periodo.groupby('Fila').agg(
            TME_Media=('TME','mean'),
            TME_Min=('TME','min'),
            TME_Max=('TME','max'),
            TMA_Media=('TMA','mean'),
            TMA_Min=('TMA','min'),
            TMA_Max=('TMA','max'),
            Volume_Total=('volume','sum')
        ).reset_index()

        st.dataframe(
            resumo.style.format({
                "TME_Media": "{:.2f}",
                "TME_Min": "{:.2f}",
                "TME_Max": "{:.2f}",
                "TMA_Media": "{:.2f}",
                "TMA_Min": "{:.2f}",
                "TMA_Max": "{:.2f}"
            }),
            use_container_width=True,
            height=400
        )

# ----------------------------
# Rodapé
# ----------------------------
st.markdown("<br>", unsafe_allow_html=True)
nome_arquivo = uploaded_file.name if uploaded_file is not None else "Nenhum arquivo"
ultima_atualizacao = dt.datetime.now().strftime('%d/%m/%Y às %H:%M:%S')

st.markdown(f"""
<div class='footer-text'>
    <strong>📄 Arquivo:</strong> {nome_arquivo} &nbsp;|&nbsp; 
    <strong>🕐 Atualizado:</strong> {ultima_atualizacao} &nbsp;|&nbsp;
    <strong>📊 Dashboard TME/TMA v2.0</strong>
</div>
""", unsafe_allow_html=True)
