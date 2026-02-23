# ============================================================
#  🌊 PLATAFORMA WAM-IA — Motor Híbrido de Inteligencia Hídrica
#  MAKEY × Integra Sur Norte × UNB
#  CORFO Innova Alta Tecnología 2025
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import pickle
import io

# ─── PAGE CONFIG ───────────────────────────────────────────
st.set_page_config(
    page_title="WAM-IA | Plataforma Hídrica",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── ESTILOS ───────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stSidebar"] { background: #1B3A6B; }
  [data-testid="stSidebar"] * { color: #E8F4FD !important; }
  .metric-card {
    background: linear-gradient(135deg, #1B3A6B 0%, #2E5FA3 100%);
    border-radius: 12px; padding: 20px; color: white;
    text-align: center; margin: 4px;
  }
  .metric-card .value { font-size: 2rem; font-weight: 800; }
  .metric-card .label { font-size: 0.8rem; opacity: 0.85; margin-top: 4px; }
  .alert-critico  { background: #D32F2F22; border-left: 4px solid #D32F2F; padding: 10px; border-radius: 6px; }
  .alert-alto     { background: #F57C0022; border-left: 4px solid #F57C00; padding: 10px; border-radius: 6px; }
  .alert-moderado { background: #F9A82522; border-left: 4px solid #F9A825; padding: 10px; border-radius: 6px; }
  .alert-bajo     { background: #388E3C22; border-left: 4px solid #388E3C; padding: 10px; border-radius: 6px; }
  .section-header { font-size: 1.4rem; font-weight: 700; color: #1B3A6B; margin-bottom: 0.5rem; }
  div[data-testid="stMetricValue"] { font-size: 1.8rem !important; color: #1B3A6B !important; }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTES ────────────────────────────────────────────
FEATURES = ['precipitacion', 'temperatura', 'evapotranspiracion',
            'dtw', 'saturacion_suelo', 'acumulacion_flujo']
TARGET        = 'susceptibilidad'
SEED          = 42
DEVICE        = torch.device('cpu')

COLORES = {
    'azul':    '#1B3A6B', 'azul2':  '#2E5FA3', 'celeste': '#4A90D9',
    'verde':   '#388E3C', 'naranja': '#F57C00', 'rojo':    '#D32F2F',
    'amarillo':'#F9A825', 'morado':  '#7B1FA2', 'gris':    '#546E7A'
}


# ════════════════════════════════════════════════════════════
#  GENERADOR DE DATOS SINTÉTICOS
# ════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def generar_datos(n_dias=1825, seed=42, variabilidad=1.0, frecuencia_extremos=0.03):
    np.random.seed(seed)
    fechas = pd.date_range(start='2019-01-01', periods=n_dias, freq='D')
    t = np.arange(n_dias)

    estacionalidad = 0.5 * (1 - np.cos(2 * np.pi * (t % 365 - 160) / 365))
    prob_lluvia = np.clip(0.05 + 0.45 * estacionalidad, 0, 0.7)
    llueve = np.random.binomial(1, prob_lluvia)
    intensidad = np.random.exponential(scale=8 * variabilidad, size=n_dias)
    extremos = np.random.binomial(1, frecuencia_extremos, size=n_dias)
    precipitacion = llueve * intensidad * (1 + extremos * np.random.uniform(5, 15, n_dias))
    precipitacion = np.clip(precipitacion, 0, 180)

    temp_base = 14 + 8 * np.cos(2 * np.pi * (t % 365 - 15) / 365)
    temperatura = temp_base + np.random.normal(0, 2, n_dias)
    evapotranspiracion = np.clip(0.15 * temperatura + np.random.normal(0, 0.5, n_dias), 0, None)

    lluvia_acum_7d  = pd.Series(precipitacion).rolling(7,  min_periods=1).sum().values
    lluvia_acum_30d = pd.Series(precipitacion).rolling(30, min_periods=1).sum().values

    dtw = np.zeros(n_dias); dtw[0] = 3.5
    for i in range(1, n_dias):
        recarga  = 0.008 * precipitacion[i] + 0.003 * lluvia_acum_7d[i]
        recesion = 0.04 * evapotranspiracion[i] + 0.005
        dtw[i] = max(0.05, min(8.0, dtw[i-1] - recarga + recesion + np.random.normal(0, 0.05)))

    saturacion = np.clip(30 + 2 * lluvia_acum_7d - 0.8 * dtw * 5
                         + np.random.normal(0, 3, n_dias), 5, 100)
    acumulacion_flujo = np.clip(
        1500 + 80 * lluvia_acum_7d + 25 * lluvia_acum_30d + np.random.normal(0, 100, n_dias),
        200, 15000)

    susc_raw = (np.tanh(precipitacion / 30) * 0.35
                + np.exp(-dtw / 2) * 0.30
                + (saturacion / 100) ** 2 * 0.25
                + np.tanh(acumulacion_flujo / 5000) * 0.10)
    susceptibilidad = np.clip(
        pd.Series(susc_raw).rolling(3, min_periods=1).mean().values
        + np.random.normal(0, 0.02, n_dias), 0, 1)

    return pd.DataFrame({
        'fecha': fechas,
        'precipitacion':     np.round(precipitacion, 2),
        'temperatura':       np.round(temperatura, 2),
        'evapotranspiracion':np.round(evapotranspiracion, 3),
        'dtw':               np.round(dtw, 3),
        'saturacion_suelo':  np.round(saturacion, 2),
        'acumulacion_flujo': np.round(acumulacion_flujo, 1),
        'susceptibilidad':   np.round(susceptibilidad, 4)
    }).set_index('fecha')


# ════════════════════════════════════════════════════════════
#  MODELO LSTM
# ════════════════════════════════════════════════════════════
class FloodLSTM(nn.Module):
    def __init__(self, input_size=6, hidden1=64, hidden2=32, horizonte=7, dropout=0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden1, batch_first=True, dropout=dropout)
        self.lstm2 = nn.LSTM(hidden1, hidden2,   batch_first=True, dropout=dropout)
        self.drop  = nn.Dropout(dropout)
        self.fc    = nn.Sequential(
            nn.Linear(hidden2, 16), nn.ReLU(),
            nn.Linear(16, horizonte), nn.Sigmoid()
        )
    def forward(self, x):
        o, _ = self.lstm1(x)
        o, _ = self.lstm2(o)
        return self.fc(self.drop(o[:, -1, :]))


def preparar_secuencias(df, features, target, ventana, horizonte):
    scX = MinMaxScaler(); scY = MinMaxScaler()
    X_s = scX.fit_transform(df[features].values)
    y_s = scY.fit_transform(df[target].values.reshape(-1,1)).flatten()
    Xs, ys = [], []
    for i in range(len(X_s) - ventana - horizonte + 1):
        Xs.append(X_s[i:i+ventana])
        ys.append(y_s[i+ventana:i+ventana+horizonte])
    return np.array(Xs, np.float32), np.array(ys, np.float32), scX, scY


# ════════════════════════════════════════════════════════════
#  UTILIDADES DE ALERTA
# ════════════════════════════════════════════════════════════
def nivel_alerta(s):
    if s >= 0.75: return "🔴 CRÍTICO",   "#D32F2F", "critico"
    if s >= 0.60: return "🟠 ALTO",      "#F57C00", "alto"
    if s >= 0.40: return "🟡 MODERADO",  "#F9A825", "moderado"
    return              "🟢 BAJO",       "#388E3C", "bajo"


# ════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 8px 0;'>
      <div style='font-size:2.5rem'>🌊</div>
      <div style='font-size:1.1rem; font-weight:800; letter-spacing:1px;'>WAM-IA</div>
      <div style='font-size:0.7rem; opacity:0.75; margin-top:2px;'>Motor Híbrido Hídrico</div>
      <div style='font-size:0.65rem; opacity:0.6;'>MAKEY × ISN × UNB</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    pagina = st.radio("Módulos", [
        "📊 Dashboard",
        "🧪 Datos Sintéticos",
        "🏋️ Entrenamiento LSTM",
        "🚨 Alerta Temprana"
    ], label_visibility="collapsed")

    st.divider()
    st.markdown("**Parámetros globales**")
    n_dias = st.slider("Días a simular", 365, 3650, 1825, 365,
                       help="Tamaño del dataset sintético")
    variabilidad = st.slider("Variabilidad climática", 0.5, 2.0, 1.0, 0.1,
                             help="Amplifica intensidad de lluvias")
    frec_extremos = st.slider("Frecuencia eventos extremos", 0.01, 0.10, 0.03, 0.01,
                              help="Prob. diaria de evento extremo")

    st.divider()
    if st.button("⚡ Regenerar datos", use_container_width=True):
        st.cache_data.clear()
        for k in ['model', 'scX', 'scY', 'historia', 'metricas']:
            st.session_state.pop(k, None)
        st.rerun()

    st.markdown("""
    <div style='text-align:center; margin-top:20px; font-size:0.65rem; opacity:0.5;'>
    CORFO Innova Alta Tecnología 2025<br>Fase 2 — MVP v1.0
    </div>""", unsafe_allow_html=True)


# ─── Datos base (compartidos entre páginas) ────────────────
df = generar_datos(n_dias, SEED, variabilidad, frec_extremos)
n_alto = (df['susceptibilidad'] > 0.6).sum()
n_critico = (df['susceptibilidad'] > 0.75).sum()


# ════════════════════════════════════════════════════════════
#  PÁGINA 1 — DASHBOARD
# ════════════════════════════════════════════════════════════
if pagina == "📊 Dashboard":
    st.markdown("## 📊 Dashboard WAM-IA — Cuenca Itata, Ñuble")
    st.caption("Visión general del sistema · datos sintéticos 2019-2024")
    st.divider()

    # KPI cards
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📅 Días simulados",   f"{len(df):,}")
    c2.metric("🌧️ Precip. media",    f"{df['precipitacion'].mean():.1f} mm")
    c3.metric("💧 DTW promedio",     f"{df['dtw'].mean():.2f} m")
    c4.metric("⚠️ Días riesgo alto", f"{n_alto}", delta=f"{n_alto/len(df)*100:.1f}% del total",
              delta_color="inverse")
    c5.metric("🔴 Días críticos",    f"{n_critico}", delta=f"{n_critico/len(df)*100:.1f}%",
              delta_color="inverse")

    st.divider()
    col_a, col_b = st.columns([2, 1])

    with col_a:
        st.markdown("#### Serie temporal de susceptibilidad a inundación")
        df_mes = df['susceptibilidad'].resample('W').mean().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_mes['fecha'], y=df_mes['susceptibilidad'],
            fill='tozeroy', fillcolor='rgba(27,58,107,0.15)',
            line=dict(color=COLORES['azul'], width=2),
            name='Susceptibilidad semanal'
        ))
        fig.add_hrect(y0=0.60, y1=1.0, fillcolor='rgba(211,47,47,0.07)',
                      line_width=0, annotation_text="Zona de riesgo alto")
        fig.add_hline(y=0.60, line_dash="dot", line_color=COLORES['rojo'],
                      annotation_text="Umbral 0.60", annotation_position="top left")
        fig.update_layout(
            height=300, margin=dict(l=0, r=0, t=20, b=0),
            plot_bgcolor='white', paper_bgcolor='white',
            xaxis=dict(showgrid=True, gridcolor='#eee'),
            yaxis=dict(showgrid=True, gridcolor='#eee', range=[0, 1]),
            legend=dict(orientation='h', y=1.1)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("#### Distribución de alertas")
        bins = [
            ("🟢 Bajo",     (df['susceptibilidad'] < 0.4).sum(),  COLORES['verde']),
            ("🟡 Moderado", ((df['susceptibilidad'] >= 0.4) & (df['susceptibilidad'] < 0.6)).sum(), COLORES['amarillo']),
            ("🟠 Alto",     ((df['susceptibilidad'] >= 0.6) & (df['susceptibilidad'] < 0.75)).sum(), COLORES['naranja']),
            ("🔴 Crítico",  (df['susceptibilidad'] >= 0.75).sum(), COLORES['rojo']),
        ]
        fig2 = go.Figure(go.Pie(
            labels=[b[0] for b in bins],
            values=[b[1] for b in bins],
            marker_colors=[b[2] for b in bins],
            hole=0.55,
            textinfo='label+percent'
        ))
        fig2.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0),
                           showlegend=False, paper_bgcolor='white')
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.markdown("#### Correlación entre variables WAM")
    corr = df.corr().round(2)
    fig3 = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r',
                     zmin=-1, zmax=1, aspect='auto')
    fig3.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0),
                       paper_bgcolor='white')
    st.plotly_chart(fig3, use_container_width=True)

    st.divider()
    st.markdown("#### Estacionalidad mensual promedio")
    df_estacional = df.copy()
    df_estacional['mes'] = df_estacional.index.month
    meses_nombres = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic']
    por_mes = df_estacional.groupby('mes')[['precipitacion','susceptibilidad','dtw']].mean()

    fig4 = make_subplots(specs=[[{"secondary_y": True}]])
    fig4.add_trace(go.Bar(x=meses_nombres, y=por_mes['precipitacion'],
                          name='Precipitación (mm)', marker_color=COLORES['celeste'], opacity=0.7))
    fig4.add_trace(go.Scatter(x=meses_nombres, y=por_mes['susceptibilidad'],
                              name='Susceptibilidad', line=dict(color=COLORES['rojo'], width=3),
                              mode='lines+markers'), secondary_y=True)
    fig4.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0),
                       plot_bgcolor='white', paper_bgcolor='white',
                       legend=dict(orientation='h', y=1.15))
    fig4.update_yaxes(title_text="Precipitación (mm)", secondary_y=False)
    fig4.update_yaxes(title_text="Susceptibilidad", secondary_y=True, range=[0, 1])
    st.plotly_chart(fig4, use_container_width=True)


# ════════════════════════════════════════════════════════════
#  PÁGINA 2 — DATOS SINTÉTICOS
# ════════════════════════════════════════════════════════════
elif pagina == "🧪 Datos Sintéticos":
    st.markdown("## 🧪 Generador de Datos Sintéticos Hidrológicos")
    st.caption("Simulación físicamente coherente de la cuenca Itata — parámetros ajustables en la barra lateral")
    st.divider()

    variable = st.selectbox("Variable a visualizar en detalle",
                            ['susceptibilidad', 'precipitacion', 'dtw',
                             'saturacion_suelo', 'acumulacion_flujo',
                             'temperatura', 'evapotranspiracion'])

    col1, col2 = st.columns([3, 1])
    with col1:
        fig = go.Figure()
        color_map = {
            'susceptibilidad': COLORES['morado'], 'precipitacion': COLORES['celeste'],
            'dtw': COLORES['naranja'], 'saturacion_suelo': COLORES['verde'],
            'acumulacion_flujo': COLORES['azul2'], 'temperatura': COLORES['rojo'],
            'evapotranspiracion': COLORES['gris']
        }
        fig.add_trace(go.Scatter(
            x=df.index, y=df[variable],
            fill='tozeroy', fillcolor=color_map[variable] + '22',
            line=dict(color=color_map[variable], width=1.5), name=variable
        ))
        if variable == 'susceptibilidad':
            fig.add_hline(y=0.6, line_dash="dot", line_color=COLORES['rojo'])
            fig.add_hline(y=0.75, line_dash="dash", line_color=COLORES['rojo'])
        if variable == 'saturacion_suelo':
            fig.add_hline(y=80, line_dash="dot", line_color=COLORES['rojo'])
        if variable == 'dtw':
            fig.update_yaxes(autorange="reversed")

        fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0),
                          plot_bgcolor='white', paper_bgcolor='white',
                          xaxis=dict(showgrid=True, gridcolor='#eee'),
                          yaxis=dict(showgrid=True, gridcolor='#eee'))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        stats = df[variable].describe()
        st.markdown(f"**Estadísticas de `{variable}`**")
        for k, v in stats.items():
            st.metric(k, f"{v:.3f}")

    st.divider()
    st.markdown("#### Vista simultánea de las 4 variables WAM clave")
    fig_multi = make_subplots(rows=4, cols=1, shared_xaxes=True,
                               subplot_titles=['Precipitación (mm)', 'DTW (m) ↓ = más peligroso',
                                               'Saturación suelo (%)', 'Susceptibilidad (0-1)'],
                               vertical_spacing=0.06)
    df_vis = df.resample('W').mean()

    fig_multi.add_trace(go.Bar(x=df_vis.index, y=df_vis['precipitacion'],
                                marker_color=COLORES['celeste'], name='Precipitación'), row=1, col=1)
    fig_multi.add_trace(go.Scatter(x=df_vis.index, y=df_vis['dtw'],
                                    fill='tozeroy', fillcolor=COLORES['naranja']+'33',
                                    line=dict(color=COLORES['naranja']), name='DTW'), row=2, col=1)
    fig_multi.add_trace(go.Scatter(x=df_vis.index, y=df_vis['saturacion_suelo'],
                                    fill='tozeroy', fillcolor=COLORES['verde']+'33',
                                    line=dict(color=COLORES['verde']), name='Saturación'), row=3, col=1)
    fig_multi.add_trace(go.Scatter(x=df_vis.index, y=df_vis['susceptibilidad'],
                                    fill='tozeroy', fillcolor=COLORES['morado']+'33',
                                    line=dict(color=COLORES['morado']), name='Susceptibilidad'), row=4, col=1)

    fig_multi.update_yaxes(autorange="reversed", row=2, col=1)
    fig_multi.update_layout(height=650, showlegend=False, margin=dict(l=0, r=0, t=40, b=0),
                             plot_bgcolor='white', paper_bgcolor='white')
    st.plotly_chart(fig_multi, use_container_width=True)

    st.divider()
    st.markdown("#### Descargar dataset sintético")
    csv = df.reset_index().to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Descargar CSV", csv,
                       file_name="dataset_sintetico_itata.csv",
                       mime="text/csv", use_container_width=True)


# ════════════════════════════════════════════════════════════
#  PÁGINA 3 — ENTRENAMIENTO LSTM
# ════════════════════════════════════════════════════════════
elif pagina == "🏋️ Entrenamiento LSTM":
    st.markdown("## 🏋️ Entrenamiento LSTM en Vivo")
    st.caption("FloodLSTM · arquitectura 2 capas · pronóstico 7 días · Motor WAM-IA Fase 2")
    st.divider()

    col_cfg, col_info = st.columns([1, 2])
    with col_cfg:
        st.markdown("**Hiperparámetros**")
        ventana   = st.slider("Ventana entrada (días)", 7, 30, 14)
        horizonte = st.slider("Horizonte pronóstico (días)", 3, 14, 7)
        epochs    = st.slider("Épocas de entrenamiento", 10, 100, 40)
        lr        = st.select_slider("Learning rate", [0.0001, 0.0005, 0.001, 0.005], value=0.001)
        hidden1   = st.select_slider("Neuronas capa 1", [32, 64, 128], value=64)
        hidden2   = st.select_slider("Neuronas capa 2", [16, 32, 64],  value=32)
        dropout   = st.slider("Dropout", 0.0, 0.5, 0.2, 0.05)

    with col_info:
        st.markdown("""
        **Arquitectura FloodLSTM**
        ```
        Input  →  (ventana × 6 features)
           ↓
        LSTM Layer 1   (hidden1 neuronas)
           ↓
        LSTM Layer 2   (hidden2 neuronas)
           ↓
        Dropout + FC   (16 → horizonte)
           ↓
        Sigmoid → susceptibilidad t+1 … t+N
        ```
        **Variables de entrada (6):** precipitación, temperatura,
        evapotranspiración, DTW, saturación suelo, acumulación flujo.

        **Variable objetivo:** índice de susceptibilidad a inundación (0–1).
        """)

    st.divider()
    btn_col, _ = st.columns([1, 3])
    iniciar = btn_col.button("🚀 Iniciar entrenamiento", use_container_width=True,
                              type="primary")

    if iniciar or 'historia' in st.session_state:
        if iniciar:
            # Limpiar estado previo
            for k in ['model', 'scX', 'scY', 'historia', 'metricas']:
                st.session_state.pop(k, None)

            # Preparar datos
            torch.manual_seed(SEED)
            X_seq, y_seq, scX, scY = preparar_secuencias(
                df, FEATURES, TARGET, ventana, horizonte)

            n = len(X_seq)
            n_test  = int(n * 0.2)
            n_val   = int(n * 0.1)
            n_train = n - n_test - n_val

            X_train = torch.tensor(X_seq[:n_train])
            y_train = torch.tensor(y_seq[:n_train])
            X_val   = torch.tensor(X_seq[n_train:n_train+n_val])
            y_val   = torch.tensor(y_seq[n_train:n_train+n_val])
            X_test  = torch.tensor(X_seq[n_train+n_val:])
            y_test  = torch.tensor(y_seq[n_train+n_val:])

            model     = FloodLSTM(len(FEATURES), hidden1, hidden2, horizonte, dropout)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
            criterion = nn.MSELoss()
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=6, factor=0.5, verbose=False)

            # ── Entrenamiento con visualización en vivo ──
            st.markdown("#### 📉 Curva de aprendizaje en tiempo real")
            chart_ph   = st.empty()
            status_ph  = st.empty()
            progress_ph = st.progress(0)

            train_losses, val_losses = [], []
            mejor_val = float('inf')
            mejor_pesos = None

            from torch.utils.data import DataLoader, TensorDataset
            BATCH = 64
            loader_train = DataLoader(TensorDataset(X_train, y_train),
                                      batch_size=BATCH, shuffle=True)

            for ep in range(1, epochs + 1):
                model.train()
                tl = 0
                for Xb, yb in loader_train:
                    optimizer.zero_grad()
                    loss = criterion(model(Xb), yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    tl += loss.item()
                tl /= len(loader_train)

                model.eval()
                with torch.no_grad():
                    vl = criterion(model(X_val), y_val).item()
                scheduler.step(vl)

                train_losses.append(tl)
                val_losses.append(vl)
                if vl < mejor_val:
                    mejor_val = vl
                    mejor_pesos = {k: v.clone() for k, v in model.state_dict().items()}

                # Actualizar gráfico cada 2 épocas
                if ep % 2 == 0 or ep == epochs:
                    fig_live = go.Figure()
                    fig_live.add_trace(go.Scatter(y=train_losses, name='Train',
                                                   line=dict(color=COLORES['azul'], width=2)))
                    fig_live.add_trace(go.Scatter(y=val_losses, name='Validación',
                                                   line=dict(color=COLORES['rojo'],
                                                             width=2, dash='dash')))
                    fig_live.update_layout(
                        height=280, margin=dict(l=0, r=0, t=10, b=0),
                        plot_bgcolor='white', paper_bgcolor='white',
                        xaxis_title='Época', yaxis_title='MSE Loss',
                        legend=dict(orientation='h', y=1.15),
                        yaxis_type='log'
                    )
                    chart_ph.plotly_chart(fig_live, use_container_width=True)
                    status_ph.info(
                        f"Época {ep}/{epochs} — Train: {tl:.6f} | Val: {vl:.6f} | "
                        f"LR: {optimizer.param_groups[0]['lr']:.6f}")
                    progress_ph.progress(ep / epochs)

            model.load_state_dict(mejor_pesos)
            progress_ph.empty()
            status_ph.success(f"✅ Entrenamiento completado — Mejor Val Loss: {mejor_val:.6f}")

            # Métricas en test
            model.eval()
            with torch.no_grad():
                y_pred_s = model(X_test).numpy()
                y_true_s = y_test.numpy()

            def desnorm(arr):
                return scY.inverse_transform(arr.reshape(-1,1)).flatten().reshape(arr.shape)

            y_pred = np.clip(desnorm(y_pred_s), 0, 1)
            y_true = np.clip(desnorm(y_true_s), 0, 1)

            metricas = []
            for d in range(horizonte):
                metricas.append({
                    'Día': f't+{d+1}',
                    'MAE':  round(mean_absolute_error(y_true[:,d], y_pred[:,d]), 4),
                    'RMSE': round(np.sqrt(mean_squared_error(y_true[:,d], y_pred[:,d])), 4),
                    'R²':   round(r2_score(y_true[:,d], y_pred[:,d]), 4)
                })

            # Guardar en session state
            st.session_state['model']    = model
            st.session_state['scX']      = scX
            st.session_state['scY']      = scY
            st.session_state['historia'] = {'train': train_losses, 'val': val_losses}
            st.session_state['metricas'] = metricas
            st.session_state['ventana']  = ventana
            st.session_state['horizonte']= horizonte
            st.session_state['y_pred']   = y_pred
            st.session_state['y_true']   = y_true

        # ── Métricas de evaluación ──
        if 'metricas' in st.session_state:
            st.divider()
            st.markdown("#### 📊 Métricas de evaluación por horizonte")
            df_met = pd.DataFrame(st.session_state['metricas'])

            col_m1, col_m2 = st.columns([1, 2])
            with col_m1:
                st.dataframe(df_met.set_index('Día'), use_container_width=True)
                mae_mean = df_met['MAE'].mean()
                r2_mean  = df_met['R²'].mean()
                color_mae = "normal" if mae_mean < 0.08 else "inverse"
                st.metric("MAE promedio", f"{mae_mean:.4f}",
                           delta="✅ Cumple TRL-5" if mae_mean < 0.08 else "❌ Sobre umbral",
                           delta_color=color_mae)
                st.metric("R² promedio", f"{r2_mean:.4f}")

            with col_m2:
                fig_met = go.Figure()
                fig_met.add_trace(go.Bar(x=df_met['Día'], y=df_met['MAE'],
                                          name='MAE', marker_color=COLORES['celeste']))
                fig_met.add_trace(go.Scatter(x=df_met['Día'], y=df_met['R²'],
                                              name='R²', mode='lines+markers',
                                              line=dict(color=COLORES['rojo'], width=2),
                                              yaxis='y2'))
                fig_met.add_hline(y=0.08, line_dash="dot", line_color=COLORES['rojo'],
                                   annotation_text="Umbral MAE TRL-5")
                fig_met.update_layout(
                    height=320, margin=dict(l=0, r=0, t=20, b=0),
                    plot_bgcolor='white', paper_bgcolor='white',
                    yaxis=dict(title='MAE'),
                    yaxis2=dict(title='R²', overlaying='y', side='right', range=[0, 1]),
                    legend=dict(orientation='h', y=1.15)
                )
                st.plotly_chart(fig_met, use_container_width=True)

            # Comparación real vs predicho
            if 'y_pred' in st.session_state:
                st.divider()
                st.markdown("#### 🔍 Real vs Predicho — Test set")
                hor_sel = st.selectbox(
                    "Horizonte a visualizar",
                    [f"t+{d+1}" for d in range(st.session_state['horizonte'])])
                d_idx = int(hor_sel.replace('t+','')) - 1
                y_p = st.session_state['y_pred'][:120, d_idx]
                y_r = st.session_state['y_true'][:120, d_idx]

                fig_cmp = go.Figure()
                fig_cmp.add_trace(go.Scatter(y=y_r, name='Real',
                                              line=dict(color=COLORES['azul'], width=2)))
                fig_cmp.add_trace(go.Scatter(y=y_p, name='Predicho',
                                              line=dict(color=COLORES['rojo'],
                                                        width=2, dash='dash')))
                fig_cmp.add_hrect(y0=0.6, y1=1.0, fillcolor='rgba(211,47,47,0.05)', line_width=0)
                fig_cmp.add_hline(y=0.6, line_dash="dot", line_color=COLORES['rojo'],
                                   annotation_text="Umbral alerta")
                fig_cmp.update_layout(
                    height=300, margin=dict(l=0, r=0, t=10, b=0),
                    plot_bgcolor='white', paper_bgcolor='white',
                    xaxis_title='Muestras', yaxis_title='Susceptibilidad',
                    yaxis=dict(range=[0, 1]),
                    legend=dict(orientation='h', y=1.15)
                )
                st.plotly_chart(fig_cmp, use_container_width=True)

    else:
        st.info("👆 Configura los hiperparámetros y presiona **Iniciar entrenamiento**")


# ════════════════════════════════════════════════════════════
#  PÁGINA 4 — ALERTA TEMPRANA
# ════════════════════════════════════════════════════════════
elif pagina == "🚨 Alerta Temprana":
    st.markdown("## 🚨 Boletín de Alerta Temprana — 7 Días")
    st.caption("Pronóstico Basado en Impacto (PBI) · Motor WAM-IA · Cuenca Itata, Ñuble")
    st.divider()

    if 'model' not in st.session_state:
        st.warning("""
        ⚠️ **No hay modelo entrenado.**

        Ve al módulo **🏋️ Entrenamiento LSTM** y entrena el modelo primero.
        Luego vuelve aquí para generar el boletín interactivo.
        """)
        st.stop()

    model    = st.session_state['model']
    scX      = st.session_state['scX']
    scY      = st.session_state['scY']
    ventana  = st.session_state['ventana']
    horizonte= st.session_state['horizonte']

    model.eval()

    # Selector de fecha
    fecha_min = df.index[ventana]
    fecha_max = df.index[-1]

    st.markdown("#### Selecciona la fecha de inicio del pronóstico")
    col_f1, col_f2, col_f3 = st.columns([2, 1, 1])

    with col_f1:
        fecha_sel = st.date_input("Fecha base",
                                    value=fecha_max.date(),
                                    min_value=fecha_min.date(),
                                    max_value=fecha_max.date())

    with col_f2:
        if st.button("🎲 Evento extremo aleatorio", use_container_width=True):
            extremos_idx = df.index[df['susceptibilidad'] > 0.70]
            if len(extremos_idx) > 0:
                fecha_sel = extremos_idx[np.random.randint(len(extremos_idx))].date()
                st.session_state['fecha_sel'] = fecha_sel

    if 'fecha_sel' in st.session_state:
        fecha_sel = st.session_state['fecha_sel']

    # Calcular pronóstico
    try:
        idx_base    = df.index.get_loc(pd.Timestamp(fecha_sel))
        datos_vent  = df[FEATURES].iloc[idx_base - ventana: idx_base].values
        vent_scaled = scX.transform(datos_vent)
        X_in        = torch.tensor(vent_scaled, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            pred_s = model(X_in).numpy().flatten()

        pred_real   = np.clip(scY.inverse_transform(pred_s.reshape(-1,1)).flatten(), 0, 1)
        fechas_pred = pd.date_range(
            start=pd.Timestamp(fecha_sel) + pd.Timedelta(days=1),
            periods=horizonte, freq='D')

        # ── Layout del boletín ──
        st.divider()

        # Alerta máxima
        max_susc   = pred_real.max()
        nivel, color_hex, clase = nivel_alerta(max_susc)
        st.markdown(f"""
        <div class='alert-{clase}'>
        <b style='font-size:1.2rem'>{nivel} — Susceptibilidad máxima: {max_susc:.3f}</b><br>
        <span style='font-size:0.9rem'>Pronóstico para {horizonte} días desde {fecha_sel.strftime('%d/%m/%Y') if hasattr(fecha_sel, 'strftime') else str(fecha_sel)}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        # Tarjetas de alerta por día
        cols = st.columns(horizonte)
        for i, (f, s) in enumerate(zip(fechas_pred, pred_real)):
            niv, chex, cls = nivel_alerta(s)
            with cols[i]:
                st.markdown(f"""
                <div style='background:{chex}18; border:2px solid {chex};
                     border-radius:10px; padding:12px; text-align:center;'>
                  <div style='font-size:0.75rem; color:#555;'>{f.strftime('%a %d/%m')}</div>
                  <div style='font-size:1.6rem; font-weight:800; color:{chex};'>{s:.2f}</div>
                  <div style='font-size:0.7rem; color:{chex};'>{niv.split(' ',1)[1]}</div>
                </div>
                """, unsafe_allow_html=True)

        st.divider()

        # Gráfico principal del boletín
        col_g1, col_g2 = st.columns([3, 1])
        with col_g1:
            st.markdown("#### Serie histórica + pronóstico")
            hist_start = max(0, idx_base - 45)
            hist_fechas = df.index[hist_start: idx_base + 1]
            hist_vals   = df['susceptibilidad'].iloc[hist_start: idx_base + 1]

            fig_bol = go.Figure()
            # Histórico
            fig_bol.add_trace(go.Scatter(
                x=hist_fechas, y=hist_vals,
                fill='tozeroy', fillcolor=COLORES['azul']+'22',
                line=dict(color=COLORES['azul'], width=2),
                name='Histórico real'))
            # Línea vertical "Hoy"
            fig_bol.add_vline(x=str(fecha_sel), line_dash='dash',
                               line_color=COLORES['gris'],
                               annotation_text='Hoy', annotation_position='top')
            # Barras pronóstico coloreadas
            for f, s in zip(fechas_pred, pred_real):
                _, chex, _ = nivel_alerta(s)
                fig_bol.add_trace(go.Bar(
                    x=[f], y=[s], marker_color=chex, opacity=0.75,
                    showlegend=False, width=86400000))
            # Línea pronóstico
            fig_bol.add_trace(go.Scatter(
                x=fechas_pred, y=pred_real,
                mode='lines+markers',
                line=dict(color=COLORES['rojo'], width=2, dash='dot'),
                marker=dict(size=8, color=COLORES['rojo']),
                name='Pronóstico LSTM'))

            fig_bol.add_hline(y=0.60, line_dash="dot", line_color=COLORES['naranja'],
                               annotation_text="Umbral alto")
            fig_bol.add_hline(y=0.75, line_dash="dash", line_color=COLORES['rojo'],
                               annotation_text="Umbral crítico")
            fig_bol.update_layout(
                height=360, margin=dict(l=0, r=0, t=10, b=0),
                plot_bgcolor='white', paper_bgcolor='white',
                yaxis=dict(range=[0, 1.05], title='Susceptibilidad',
                           showgrid=True, gridcolor='#eee'),
                xaxis=dict(showgrid=True, gridcolor='#eee'),
                legend=dict(orientation='h', y=1.12),
                barmode='overlay'
            )
            st.plotly_chart(fig_bol, use_container_width=True)

        with col_g2:
            st.markdown("#### Condiciones actuales")
            dia_actual = df.iloc[idx_base]
            indicadores = {
                '🌧️ Precip.':    f"{dia_actual['precipitacion']:.1f} mm",
                '🌡️ Temp.':      f"{dia_actual['temperatura']:.1f} °C",
                '💧 DTW':        f"{dia_actual['dtw']:.2f} m",
                '🌱 Saturación': f"{dia_actual['saturacion_suelo']:.1f}%",
                '🔮 Susc. hoy':  f"{dia_actual['susceptibilidad']:.3f}",
            }
            for k, v in indicadores.items():
                st.metric(k, v)

        # Tabla resumen exportable
        st.divider()
        st.markdown("#### Tabla resumen del pronóstico")
        df_boletin = pd.DataFrame({
            'Fecha': [f.strftime('%d/%m/%Y') for f in fechas_pred],
            'Susceptibilidad': [f"{s:.4f}" for s in pred_real],
            'Nivel de Alerta': [nivel_alerta(s)[0] for s in pred_real],
            'Acción recomendada': [
                'Monitoreo continuo' if nivel_alerta(s)[2] == 'critico' else
                'Activar protocolos' if nivel_alerta(s)[2] == 'alto' else
                'Vigilancia estándar' if nivel_alerta(s)[2] == 'moderado' else
                'Sin acción inmediata'
                for s in pred_real
            ]
        })
        st.dataframe(df_boletin, use_container_width=True, hide_index=True)

        csv_bol = df_boletin.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Descargar boletín CSV", csv_bol,
                           file_name=f"boletin_wam_ia_{fecha_sel}.csv",
                           mime="text/csv", use_container_width=True)

    except Exception as e:
        st.error(f"Error al calcular pronóstico: {e}")
