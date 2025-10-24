import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime as dt
from typing import Optional

# ----------------------------
# Configurações iniciais
# ----------------------------
st.set_page_config(page_title="Dashboard TME/TMA", layout="wide", initial_sidebar_state="expanded")

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
    'N2-Administrativo': {'TME': 3.0, 'TMA': 30.0}
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
            # Formatos esperados: HH:MM:SS ou MM:SS ou apenas número
            parts = tempo_obj.strip().split(':')
            if len(parts) == 3:
                h, m, s = [int(x) for x in parts]
                return h * 60 + m + s / 60
            if len(parts) == 2:
                m, s = [int(x) for x in parts]
                return m + s / 60
            # tenta converter número
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
        # Verificações mínimas
        required = ['Inicio da ação', 'Fila', 'Tempo na Fila', 'Tempo de atendimento']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Coluna obrigatória não encontrada: '{col}'")

        # Conversão de datas
        df['Inicio da ação'] = pd.to_datetime(df['Inicio da ação'], errors='coerce')
        df = df.dropna(subset=['Inicio da ação'])
        df['Data'] = df['Inicio da ação'].dt.date

        # Converter tempos para minutos
        df['TME_minutos'] = df['Tempo na Fila'].apply(converter_tempo_para_minutos)
        df['TMA_minutos'] = df['Tempo de atendimento'].apply(converter_tempo_para_minutos)

        # Agrupar por Data e Fila
        resultados = df.groupby(['Data', 'Fila']).agg(
            TME=('TME_minutos', 'mean'),
            TMA=('TMA_minutos', 'mean'),
            volume=('Fila', 'size')
        ).reset_index()

        # Ordenar por data
        resultados = resultados.sort_values(by=['Data'])
        return resultados
    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")
        return None

def cores_card(valor: float, meta: float) -> str:
    """Retorna cor (hex) para o card com base na comparação valor vs meta (verde = bom)."""
    try:
        if valor <= 0:
            return "#6c757d"  # cinza neutro
        return "#198754" if valor <= meta else "#dc3545"  # verde se dentro da meta, vermelho se fora
    except Exception:
        return "#6c757d"

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
    """Gera gráfico Plotly com possibilidade de múltiplas filas (traços separados por fila)."""
    # Cores fixas para consistência
    paleta = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#e377c2', '#7f7f7f']
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Para cada fila, criar série de TME e TMA
    for i, fila in enumerate(filas_selecionadas):
        df_f = df_fila[df_fila['Fila'] == fila].copy().sort_values('Data')
        if df_f.empty:
            continue
        cor = paleta[i % len(paleta)]
        # TME
        fig.add_trace(
            go.Scatter(
                x=df_f['Data'], y=df_f['TME'], mode='lines+markers',
                name=f"{fila} - TME",
                marker=dict(symbol='circle', size=8),
                line=dict(width=3, color=cor),
                hovertemplate="<b>%{x|%d/%m/%Y}</b><br>TME: %{y:.2f} min<br>%{text}",
                text=[formatar_tempo_hhmmss(v) for v in df_f['TME']]
            ),
            secondary_y=False
        )
        # TMA no eixo secundário (mesma cor com estilo mais suave)
        fig.add_trace(
            go.Scatter(
                x=df_f['Data'], y=df_f['TMA'], mode='lines+markers',
                name=f"{fila} - TMA",
                marker=dict(symbol='square', size=8),
                line=dict(width=3, dash='dash', color=cor),
                hovertemplate="<b>%{x|%d/%m/%Y}</b><br>TMA: %{y:.2f} min<br>%{text}",
                text=[formatar_tempo_hhmmss(v) for v in df_f['TMA']]
            ),
            secondary_y=True
        )

    # Linhas de metas — quando múltiplas filas, mostramos faixa entre min e max das metas selecionadas
    metas_tme = [METAS.get(fila, {'TME': 0})['TME'] for fila in filas_selecionadas if fila in METAS]
    metas_tma = [METAS.get(fila, {'TMA': 0})['TMA'] for fila in filas_selecionadas if fila in METAS]
    # Apenas desenhar se houver valores válidos
    all_dates = sorted(df_fila['Data'].unique().tolist())
    if all_dates:
        if metas_tme:
            meta_tme_min = min(metas_tme)
            fig.add_trace(
                go.Scatter(x=all_dates, y=[meta_tme_min]*len(all_dates), mode='lines',
                           name=f"Meta TME (min={meta_tme_min:.0f}m)",
                           line=dict(color='#0b74de', dash='dot', width=2)),
                secondary_y=False
            )
        if metas_tma:
            meta_tma_min = min(metas_tma)
            fig.add_trace(
                go.Scatter(x=all_dates, y=[meta_tma_min]*len(all_dates), mode='lines',
                           name=f"Meta TMA (min={meta_tma_min:.0f}m)",
                           line=dict(color='#ff7f0e', dash='dot', width=2)),
                secondary_y=True
            )

    # Layout
    fig.update_layout(
        title_text="<b>Desempenho TME e TMA</b>",
        template='plotly_white',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(t=90, b=40, l=60, r=40),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Segoe UI, Roboto, Arial", color='#111827')
    )

    # Eixos
    fig.update_xaxes(title_text="Data", tickformat="%d/%b", showgrid=True, gridcolor='rgba(220,220,220,0.8)')
    fig.update_yaxes(title_text="TME (min)", secondary_y=False, showgrid=True, gridcolor='rgba(220,220,220,0.8)')
    fig.update_yaxes(title_text="TMA (min)", secondary_y=True, showgrid=False)

    return fig

# ----------------------------
# CSS customizado
# ----------------------------
st.markdown("""
<style>
/* Layout centralizado */
div.block-container {
    max-width: 1400px;
    padding-top: 1.25rem;
    padding-left: 1.5rem;
    padding-right: 1.5rem;
}

/* Sidebar styles */
[data-testid="stSidebar"] {
    background: linear-gradient(#ffffff, #f8f9fb);
    border-right: 1px solid #e6e9ef;
}

/* Header titles */
h1, h2, h3 {
    color: #0f172a;
    font-family: 'Segoe UI', Roboto, Arial;
}

/* Metric card */
.metric-card {
    background: #ffffff;
    padding: 12px;
    border-radius: 10px;
    border: 1px solid #e6e9ef;
    box-shadow: 0 1px 3px rgba(16,24,40,0.04);
}

/* Small helper text */
.small-muted {
    color: #6b7280;
    font-size: 13px;
}

/* Make Plotly container look nicer */
.stPlotlyChart > div {
    border-radius: 12px !important;
}

/* Responsividade - ajustar espaçamentos em telas menores */
@media (max-width: 800px) {
    .stSidebar { display: block; }
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Sidebar: Upload e controles
# ----------------------------
with st.sidebar:
    st.title("Configurações")
    st.markdown("**1. Upload do arquivo**")
    uploaded_file = st.file_uploader("Excel (.xlsx) com colunas: 'Inicio da ação', 'Fila', 'Tempo na Fila', 'Tempo de atendimento'", type=['xlsx'])
    st.markdown("---")
    st.markdown("**2. Navegação**")
    page = st.radio("Escolha a página", ["Visão Geral", "Visualização por Fila", "Upload & Dados"])
    st.markdown("---")
    st.markdown("**3. Filtros rápidos** (aplicados nas páginas)")
    filas_selecionadas = st.multiselect("Filas", options=FILAS_MONITORADAS, default=FILAS_MONITORADAS[:1])
    # Período padrão: últimos 14 dias
    hoje = dt.date.today()
    periodo_inicio = st.date_input("Data início", value=hoje - dt.timedelta(days=14))
    periodo_fim = st.date_input("Data fim", value=hoje)
    if periodo_inicio > periodo_fim:
        st.error("A data início não pode ser posterior à data fim.")
    st.markdown("---")
    st.caption("Dica: selecione múltiplas filas para comparar.")

# Se não há arquivo, exibe instrução nas páginas (exceto Upload)
if uploaded_file is None:
    if page != "Upload & Dados":
        st.warning("Faça upload do arquivo Excel na barra lateral (Upload do arquivo).")
    if page == "Upload & Dados":
        st.info("Use este espaço para validar o arquivo antes do processamento.")
    # Exibe uma área minimal e retorna para não quebrar execução
    if page != "Visão Geral": 
        # Mostrar amostra de template (opcional)
        st.stop()

# ----------------------------
# Processamento dos dados
# ----------------------------
with st.spinner("Processando arquivo..."):
    df_processado = ler_e_processar_dados(uploaded_file)

if df_processado is None or df_processado.empty:
    st.warning("Nenhum dado válido após processamento. Verifique o arquivo e as colunas.")
    st.stop()

# Aplicar filtro por período
df_processado['Data'] = pd.to_datetime(df_processado['Data'])
mask_periodo = (df_processado['Data'].dt.date >= periodo_inicio) & (df_processado['Data'].dt.date <= periodo_fim)
df_periodo = df_processado.loc[mask_periodo].copy()

if df_periodo.empty:
    st.warning("Não há dados no intervalo de datas selecionado.")
    st.stop()

# Se nenhuma fila selecionada, usar todas disponíveis
if not filas_selecionadas:
    filas_selecionadas = sorted(df_periodo['Fila'].unique().tolist())

# ----------------------------
# Páginas
# ----------------------------
if page == "Visão Geral":
    st.header("Visão Geral")
    st.markdown("Resumo das filas selecionadas e tendências.")

    # Agregação por fila (métricas gerais)
    resumo_por_fila = df_periodo[df_periodo['Fila'].isin(filas_selecionadas)].groupby('Fila').agg(
        TME_media=('TME', 'mean'),
        TMA_media=('TMA', 'mean'),
        volume=('volume', 'sum')
    ).reset_index()

    # Top row: cards de métricas agregadas (TME médio, TMA médio e Volume)
    agregados = agregar_metricas(df_periodo[df_periodo['Fila'].isin(filas_selecionadas)])
    col1, col2, col3, col4 = st.columns([1.6,1.6,1.6,2.2])
    with col1:
        cor = cores_card(agregados['TME_medio'], min([METAS.get(f,{'TME':999})['TME'] for f in filas_selecionadas if f in METAS]))
        st.markdown(f"<div class='metric-card' style='border-left:4px solid {cor}'>"
                    f"<div style='font-size:14px;color:#6b7280'>TME Médio (min)</div>"
                    f"<div style='font-size:22px;font-weight:700'>{agregados['TME_medio']:.2f} min</div>"
                    f"<div class='small-muted'>{formatar_tempo_hhmmss(agregados['TME_medio'])}</div>"
                    f"</div>", unsafe_allow_html=True)
    with col2:
        cor = cores_card(agregados['TMA_medio'], min([METAS.get(f,{'TMA':999})['TMA'] for f in filas_selecionadas if f in METAS]))
        st.markdown(f"<div class='metric-card' style='border-left:4px solid {cor}'>"
                    f"<div style='font-size:14px;color:#6b7280'>TMA Médio (min)</div>"
                    f"<div style='font-size:22px;font-weight:700'>{agregados['TMA_medio']:.2f} min</div>"
                    f"<div class='small-muted'>{formatar_tempo_hhmmss(agregados['TMA_medio'])}</div>"
                    f"</div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card' style='border-left:4px solid #0d6efd'>"
                    f"<div style='font-size:14px;color:#6b7280'>Volume Total</div>"
                    f"<div style='font-size:22px;font-weight:700'>{agregados['volume_total']}</div>"
                    f"<div class='small-muted'>Período selecionado</div>"
                    f"</div>", unsafe_allow_html=True)
    with col4:
        st.markdown("<div class='metric-card'>"
                    "<div style='font-size:14px;color:#6b7280'>Filas selecionadas</div>"
                    f"<div style='font-size:16px;font-weight:600'>{', '.join(filas_selecionadas)}</div>"
                    f"<div class='small-muted'>Período: {periodo_inicio.strftime('%d/%m/%Y')} — {periodo_fim.strftime('%d/%m/%Y')}</div>"
                    "</div>", unsafe_allow_html=True)

    st.markdown("---")
    # Gráfico consolidado para todas as filas selecionadas
    st.subheader("Tendência TME/TMA — Filas selecionadas")
    fig_geral = gerar_grafico_tme_tma_plotly(df_periodo, filas_selecionadas)
    st.plotly_chart(fig_geral, use_container_width=True, height=520)

    st.markdown("---")
    st.subheader("Resumo por fila")
    st.dataframe(resumo_por_fila.style.format({"TME_media":"{:.2f}", "TMA_media":"{:.2f}"}), use_container_width=True)

elif page == "Visualização por Fila":
    st.header("Visualização por Fila")
    st.markdown("Escolha a(s) fila(s) e confira o gráfico e métricas por fila.")

    # Permitir selecionar filas específicas (ainda respeitando seleção lateral)
    filas_para_visual = st.multiselect("Filas para visualizar (gráficos separados por fila)", options=filas_selecionadas, default=filas_selecionadas[:1])
    if not filas_para_visual:
        st.warning("Selecione pelo menos uma fila para visualizar.")
        st.stop()

    # Para cada fila selecionada mostrar seção com gráfico e mini-métricas
    for fila in filas_para_visual:
        st.subheader(f"Fila: {fila}")
        df_fila = df_periodo[df_periodo['Fila'] == fila].sort_values('Data')
        if df_fila.empty:
            st.info("Sem dados para este período.")
            continue

        # Métricas da fila
        tme_medio = float(df_fila['TME'].mean())
        tma_medio = float(df_fila['TMA'].mean())
        volume = int(df_fila['volume'].sum())
        meta_tme = METAS.get(fila, {'TME': 0})['TME']
        meta_tma = METAS.get(fila, {'TMA': 0})['TMA']

        c1, c2, c3 = st.columns([1.8,1.8,2.2])
        with c1:
            cor = cores_card(tme_medio, meta_tme)
            st.markdown(f"<div class='metric-card' style='border-left:4px solid {cor}'>"
                        f"<div style='font-size:13px;color:#6b7280'>TME Médio</div>"
                        f"<div style='font-size:18px;font-weight:700'>{tme_medio:.2f} min</div>"
                        f"<div class='small-muted'>{formatar_tempo_hhmmss(tme_medio)} — Meta: {formatar_tempo_hhmmss(meta_tme)}</div>"
                        "</div>", unsafe_allow_html=True)
        with c2:
            cor = cores_card(tma_medio, meta_tma)
            st.markdown(f"<div class='metric-card' style='border-left:4px solid {cor}'>"
                        f"<div style='font-size:13px;color:#6b7280'>TMA Médio</div>"
                        f"<div style='font-size:18px;font-weight:700'>{tma_medio:.2f} min</div>"
                        f"<div class='small-muted'>{formatar_tempo_hhmmss(tma_medio)} — Meta: {formatar_tempo_hhmmss(meta_tma)}</div>"
                        "</div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='metric-card' style='border-left:4px solid #0d6efd'>"
                        f"<div style='font-size:13px;color:#6b7280'>Volume (Período)</div>"
                        f"<div style='font-size:18px;font-weight:700'>{volume}</div>"
                        f"<div class='small-muted'>Total de atendimentos</div>"
                        "</div>", unsafe_allow_html=True)

        # Gráfico desta fila (apenas uma linha de TME + TMA)
        fig_f = gerar_grafico_tme_tma_plotly(df_periodo[df_periodo['Fila'].isin([fila])], [fila])
        st.plotly_chart(fig_f, use_container_width=True, height=480)
        st.markdown("---")

elif page == "Upload & Dados":
    st.header("Upload & Dados (validação)")
    st.markdown("Use esta página para validar rapidamente o arquivo e baixar amostras.")

    st.subheader("Amostra do arquivo processado")
    st.dataframe(df_periodo.head(200), use_container_width=True)

    st.markdown("### Estatísticas rápidas por fila")
    resumo = df_periodo.groupby('Fila').agg(
        TME_media=('TME','mean'),
        TMA_media=('TMA','mean'),
        volume=('volume','sum')
    ).reset_index()
    st.dataframe(resumo.style.format({"TME_media":"{:.2f}", "TMA_media":"{:.2f}"}), use_container_width=True)

    st.markdown("### Download dos dados processados (CSV)")
    csv = df_periodo.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Baixar CSV (dados processados)", data=csv, file_name="dados_processados.csv", mime="text/csv")

# ----------------------------
# Rodapé com última atualização do arquivo
# ----------------------------
st.markdown("---")
ultima_atual = uploaded_file.name if uploaded_file is not None else "Nenhum arquivo"
st.markdown(f"<div style='color:#6b7280;font-size:13px'>Arquivo carregado: <strong>{ultima_atual}</strong> — Atualizado em {dt.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</div>", unsafe_allow_html=True)
