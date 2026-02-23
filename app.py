# ============================================================
#  🌊 PLATAFORMA WAM-IA — Motor Híbrido de Inteligencia Hídrica
#  MAKEY × Integra Sur Norte × UNB
#  CORFO Innova Alta Tecnología 2025
#  v2.0 — compatible Python 3.13 / Streamlit Cloud
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(
    page_title="WAM-IA | Plataforma Hídrica",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  [data-testid="stSidebar"] { background: #1B3A6B; }
  [data-testid="stSidebar"] * { color: #E8F4FD !important; }
  .alert-critico  { background:#D32F2F22; border-left:4px solid #D32F2F; padding:10px; border-radius:6px; }
  .alert-alto     { background:#F57C0022; border-left:4px solid #F57C00; padding:10px; border-radius:6px; }
  .alert-moderado { background:#F9A82522; border-left:4px solid #F9A825; padding:10px; border-radius:6px; }
  .alert-bajo     { background:#388E3C22; border-left:4px solid #388E3C; padding:10px; border-radius:6px; }
  div[data-testid="stMetricValue"] { font-size:1.8rem !important; color:#1B3A6B !important; }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTES ────────────────────────────────────────────
FEATURES = ['precipitacion','temperatura','evapotranspiracion',
            'dtw','saturacion_suelo','acumulacion_flujo']
TARGET = 'susceptibilidad'
SEED   = 42
C = {
    'azul':'#1B3A6B','azul2':'#2E5FA3','celeste':'#4A90D9',
    'verde':'#388E3C','naranja':'#F57C00','rojo':'#D32F2F',
    'amarillo':'#F9A825','morado':'#7B1FA2','gris':'#546E7A'
}

# ════════════════════════════════════════════════════════════
#  LSTM EN NUMPY PURO
# ════════════════════════════════════════════════════════════
class NumpyLSTM:
    """
    LSTM liviano implementado en numpy puro.
    Sin dependencias de torch/tensorflow — compatible con cualquier entorno.
    Arquitectura: LSTM(input→hidden) → Dense(hidden→horizonte) → Sigmoid
    """
    def __init__(self, input_size, hidden_size, output_size, lr=0.001):
        self.hs  = hidden_size
        self.lr  = lr
        scale    = 0.1
        # Pesos LSTM (concatenados: input + hidden)
        n = input_size + hidden_size
        self.Wf = np.random.randn(hidden_size, n)  * scale
        self.Wi = np.random.randn(hidden_size, n)  * scale
        self.Wc = np.random.randn(hidden_size, n)  * scale
        self.Wo = np.random.randn(hidden_size, n)  * scale
        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))
        # Capa densa de salida
        self.Wy = np.random.randn(output_size, hidden_size) * scale
        self.by = np.zeros((output_size, 1))
        # Momento Adam
        self._init_adam()

    def _init_adam(self):
        self.t  = 0
        self.m  = {}; self.v = {}
        for name in ['Wf','Wi','Wc','Wo','bf','bi','bc','bo','Wy','by']:
            self.m[name] = np.zeros_like(getattr(self, name))
            self.v[name] = np.zeros_like(getattr(self, name))

    @staticmethod
    def sigmoid(x):  return 1 / (1 + np.exp(-np.clip(x, -15, 15)))
    @staticmethod
    def tanh(x):     return np.tanh(np.clip(x, -15, 15))

    def forward(self, X):
        """X: (seq_len, input_size)"""
        T       = X.shape[0]
        h       = np.zeros((self.hs, 1))
        c       = np.zeros((self.hs, 1))
        cache   = []
        for t in range(T):
            x   = X[t].reshape(-1, 1)
            xh  = np.vstack([x, h])
            f   = self.sigmoid(self.Wf @ xh + self.bf)
            i   = self.sigmoid(self.Wi @ xh + self.bi)
            g   = self.tanh(self.Wc @ xh + self.bc)
            o   = self.sigmoid(self.Wo @ xh + self.bo)
            c   = f * c + i * g
            h   = o * self.tanh(c)
            cache.append((x, xh, f, i, g, o, c, h))
        y_raw = self.Wy @ h + self.by
        y     = self.sigmoid(y_raw)
        return y.flatten(), h, c, cache

    def backward(self, X, y_true, y_pred, h_last, c_last, cache):
        y_true = np.array(y_true).reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
        dy     = y_pred - y_true          # MSE grad
        dWy    = dy @ h_last.T
        dby    = dy.copy()
        dh     = self.Wy.T @ dy
        dc     = np.zeros_like(dh)

        grads = {k: np.zeros_like(getattr(self, k))
                 for k in ['Wf','Wi','Wc','Wo','bf','bi','bc','bo']}

        for t in reversed(range(len(cache))):
            x, xh, f, i, g, o, c_t, h_t = cache[t]
            c_prev = cache[t-1][6] if t > 0 else np.zeros_like(c_t)

            tanh_c = self.tanh(c_t)
            do  = dh * tanh_c
            dc  += dh * o * (1 - tanh_c**2)
            df  = dc * c_prev
            di  = dc * g
            dg  = dc * i
            dc  = dc * f

            ddo = do * o * (1 - o)
            ddf = df * f * (1 - f)
            ddi = di * i * (1 - i)
            ddg = dg * (1 - g**2)

            grads['Wo'] += ddo @ xh.T; grads['bo'] += ddo
            grads['Wf'] += ddf @ xh.T; grads['bf'] += ddf
            grads['Wi'] += ddi @ xh.T; grads['bi'] += ddi
            grads['Wc'] += ddg @ xh.T; grads['bc'] += ddg
            dh = (self.Wf.T @ ddf + self.Wi.T @ ddi +
                  self.Wc.T @ ddg + self.Wo.T @ ddo)[:self.hs]

        # Clip de gradientes
        for k in grads:
            grads[k] = np.clip(grads[k], -1, 1)
        dWy = np.clip(dWy, -1, 1)
        dby = np.clip(dby, -1, 1)

        # Adam update
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for name, grad in {**grads, 'Wy': dWy, 'by': dby}.items():
            self.m[name] = b1 * self.m[name] + (1-b1) * grad
            self.v[name] = b2 * self.v[name] + (1-b2) * grad**2
            mh = self.m[name] / (1 - b1**self.t)
            vh = self.v[name] / (1 - b2**self.t)
            setattr(self, name, getattr(self, name) - self.lr * mh / (np.sqrt(vh) + eps))

        return float(np.mean((y_pred - y_true)**2))

    def predict(self, X):
        y, _, _, _ = self.forward(X)
        return y


# ════════════════════════════════════════════════════════════
#  GENERADOR DE DATOS SINTÉTICOS
# ════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def generar_datos(n_dias=1825, seed=42, variabilidad=1.0, frec_ext=0.03):
    np.random.seed(seed)
    fechas = pd.date_range('2019-01-01', periods=n_dias, freq='D')
    t = np.arange(n_dias)

    estac = 0.5 * (1 - np.cos(2*np.pi*(t%365-160)/365))
    prob  = np.clip(0.05 + 0.45*estac, 0, 0.7)
    llueve = np.random.binomial(1, prob)
    intens = np.random.exponential(8*variabilidad, n_dias)
    ext    = np.random.binomial(1, frec_ext, n_dias)
    precip = np.clip(llueve*intens*(1+ext*np.random.uniform(5,15,n_dias)), 0, 180)

    temp_b = 14 + 8*np.cos(2*np.pi*(t%365-15)/365)
    temp   = temp_b + np.random.normal(0, 2, n_dias)
    etp    = np.clip(0.15*temp + np.random.normal(0,0.5,n_dias), 0, None)

    r7  = pd.Series(precip).rolling(7,  min_periods=1).sum().values
    r30 = pd.Series(precip).rolling(30, min_periods=1).sum().values

    dtw = np.zeros(n_dias); dtw[0] = 3.5
    for i in range(1, n_dias):
        dtw[i] = max(0.05, min(8.0, dtw[i-1]
                               - 0.008*precip[i] - 0.003*r7[i]
                               + 0.04*etp[i] + 0.005
                               + np.random.normal(0, 0.05)))

    sat  = np.clip(30+2*r7-0.8*dtw*5+np.random.normal(0,3,n_dias), 5, 100)
    afl  = np.clip(1500+80*r7+25*r30+np.random.normal(0,100,n_dias), 200, 15000)
    susc = np.clip(
        pd.Series(np.tanh(precip/30)*0.35 + np.exp(-dtw/2)*0.30
                  + (sat/100)**2*0.25 + np.tanh(afl/5000)*0.10
                  ).rolling(3, min_periods=1).mean().values
        + np.random.normal(0,0.02,n_dias), 0, 1)

    return pd.DataFrame({
        'precipitacion':     np.round(precip,2),
        'temperatura':       np.round(temp,2),
        'evapotranspiracion':np.round(etp,3),
        'dtw':               np.round(dtw,3),
        'saturacion_suelo':  np.round(sat,2),
        'acumulacion_flujo': np.round(afl,1),
        'susceptibilidad':   np.round(susc,4)
    }, index=fechas)


def preparar_secuencias(df, ventana, horizonte):
    scX = MinMaxScaler(); scY = MinMaxScaler()
    Xs  = scX.fit_transform(df[FEATURES].values)
    ys  = scY.fit_transform(df[TARGET].values.reshape(-1,1)).flatten()
    seqX, seqY = [], []
    for i in range(len(Xs)-ventana-horizonte+1):
        seqX.append(Xs[i:i+ventana])
        seqY.append(ys[i+ventana:i+ventana+horizonte])
    return np.array(seqX), np.array(seqY), scX, scY


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
    <div style='text-align:center;padding:16px 0 8px 0'>
      <div style='font-size:2.5rem'>🌊</div>
      <div style='font-size:1.1rem;font-weight:800;letter-spacing:1px'>WAM-IA</div>
      <div style='font-size:0.7rem;opacity:.75;margin-top:2px'>Motor Híbrido Hídrico</div>
      <div style='font-size:0.65rem;opacity:.6'>MAKEY × ISN × UNB</div>
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
    n_dias       = st.slider("Días a simular",           365, 3650, 1825, 365)
    variabilidad = st.slider("Variabilidad climática",   0.5,  2.0,  1.0, 0.1)
    frec_ext     = st.slider("Frec. eventos extremos",  0.01, 0.10, 0.03, 0.01)

    st.divider()
    if st.button("⚡ Regenerar datos", use_container_width=True):
        st.cache_data.clear()
        for k in ['model','scX','scY','historia','metricas','ventana',
                  'horizonte','y_pred','y_true']:
            st.session_state.pop(k, None)
        st.rerun()

    st.markdown("""
    <div style='text-align:center;margin-top:20px;font-size:.65rem;opacity:.5'>
    CORFO Innova Alta Tecnología 2025<br>Fase 2 — MVP v2.0
    </div>""", unsafe_allow_html=True)


df = generar_datos(n_dias, SEED, variabilidad, frec_ext)
n_alto    = (df['susceptibilidad'] > 0.60).sum()
n_critico = (df['susceptibilidad'] > 0.75).sum()


# ════════════════════════════════════════════════════════════
#  PÁGINA 1 — DASHBOARD
# ════════════════════════════════════════════════════════════
if pagina == "📊 Dashboard":
    st.markdown("## 📊 Dashboard WAM-IA — Cuenca Itata, Ñuble")
    st.caption("Visión general del sistema · datos sintéticos 2019-2024")
    st.divider()

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("📅 Días simulados",   f"{len(df):,}")
    c2.metric("🌧️ Precip. media",    f"{df['precipitacion'].mean():.1f} mm")
    c3.metric("💧 DTW promedio",     f"{df['dtw'].mean():.2f} m")
    c4.metric("⚠️ Días riesgo alto", f"{n_alto}",
              delta=f"{n_alto/len(df)*100:.1f}%", delta_color="inverse")
    c5.metric("🔴 Días críticos",    f"{n_critico}",
              delta=f"{n_critico/len(df)*100:.1f}%", delta_color="inverse")

    st.divider()
    col_a, col_b = st.columns([2,1])

    with col_a:
        st.markdown("#### Serie temporal de susceptibilidad")
        dfm = df['susceptibilidad'].resample('W').mean().reset_index()
        dfm.columns = ['fecha','susceptibilidad']
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dfm['fecha'], y=dfm['susceptibilidad'],
            fill='tozeroy', fillcolor='rgba(27,58,107,.15)',
            line=dict(color=C['azul'], width=2), name='Susceptibilidad semanal'))
        fig.add_hrect(y0=.6, y1=1.0, fillcolor='rgba(211,47,47,.07)', line_width=0)
        fig.add_hline(y=.6, line_dash='dot', line_color=C['rojo'],
                      annotation_text='Umbral 0.60', annotation_position='top left')
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=20,b=0),
                          plot_bgcolor='white', paper_bgcolor='white',
                          yaxis=dict(range=[0,1]))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("#### Distribución de alertas")
        bins = [
            ("🟢 Bajo",     (df['susceptibilidad']<.4).sum(),              C['verde']),
            ("🟡 Moderado", ((df['susceptibilidad']>=.4)&(df['susceptibilidad']<.6)).sum(), C['amarillo']),
            ("🟠 Alto",     ((df['susceptibilidad']>=.6)&(df['susceptibilidad']<.75)).sum(), C['naranja']),
            ("🔴 Crítico",  (df['susceptibilidad']>=.75).sum(),            C['rojo']),
        ]
        fig2 = go.Figure(go.Pie(
            labels=[b[0] for b in bins], values=[b[1] for b in bins],
            marker_colors=[b[2] for b in bins], hole=.55, textinfo='label+percent'))
        fig2.update_layout(height=300, margin=dict(l=0,r=0,t=20,b=0),
                           showlegend=False, paper_bgcolor='white')
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.markdown("#### Correlación entre variables WAM")
    corr = df.corr().round(2)
    fig3 = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r',
                     zmin=-1, zmax=1, aspect='auto')
    fig3.update_layout(height=360, margin=dict(l=0,r=0,t=10,b=0), paper_bgcolor='white')
    st.plotly_chart(fig3, use_container_width=True)

    st.divider()
    st.markdown("#### Estacionalidad mensual promedio")
    meses = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic']
    por_mes = df.copy()
    por_mes['mes'] = por_mes.index.month
    pm = por_mes.groupby('mes')[['precipitacion','susceptibilidad']].mean()
    fig4 = make_subplots(specs=[[{"secondary_y": True}]])
    fig4.add_trace(go.Bar(x=meses, y=pm['precipitacion'],
                          name='Precipitación (mm)', marker_color=C['celeste'], opacity=.7))
    fig4.add_trace(go.Scatter(x=meses, y=pm['susceptibilidad'], name='Susceptibilidad',
                              line=dict(color=C['rojo'], width=3), mode='lines+markers'),
                  secondary_y=True)
    fig4.update_layout(height=300, margin=dict(l=0,r=0,t=20,b=0),
                       plot_bgcolor='white', paper_bgcolor='white',
                       legend=dict(orientation='h', y=1.15))
    fig4.update_yaxes(title_text="Precipitación (mm)", secondary_y=False)
    fig4.update_yaxes(title_text="Susceptibilidad", secondary_y=True, range=[0,1])
    st.plotly_chart(fig4, use_container_width=True)


# ════════════════════════════════════════════════════════════
#  PÁGINA 2 — DATOS SINTÉTICOS
# ════════════════════════════════════════════════════════════
elif pagina == "🧪 Datos Sintéticos":
    st.markdown("## 🧪 Generador de Datos Sintéticos Hidrológicos")
    st.caption("Simulación físicamente coherente · parámetros ajustables en la barra lateral")
    st.divider()

    variable = st.selectbox("Variable a visualizar",
        ['susceptibilidad','precipitacion','dtw','saturacion_suelo',
         'acumulacion_flujo','temperatura','evapotranspiracion'])

    cmap = {'susceptibilidad':C['morado'],'precipitacion':C['celeste'],
            'dtw':C['naranja'],'saturacion_suelo':C['verde'],
            'acumulacion_flujo':C['azul2'],'temperatura':C['rojo'],
            'evapotranspiracion':C['gris']}

    col1, col2 = st.columns([3,1])
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df[variable],
            fill='tozeroy', fillcolor=cmap[variable]+'22',
            line=dict(color=cmap[variable], width=1.5), name=variable))
        if variable == 'susceptibilidad':
            fig.add_hline(y=0.6,  line_dash='dot',  line_color=C['rojo'])
            fig.add_hline(y=0.75, line_dash='dash', line_color=C['rojo'])
        if variable == 'saturacion_suelo':
            fig.add_hline(y=80, line_dash='dot', line_color=C['rojo'])
        if variable == 'dtw':
            fig.update_yaxes(autorange='reversed')
        fig.update_layout(height=350, margin=dict(l=0,r=0,t=10,b=0),
                          plot_bgcolor='white', paper_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown(f"**Estadísticas de `{variable}`**")
        for k, v in df[variable].describe().items():
            st.metric(k, f"{v:.3f}")

    st.divider()
    st.markdown("#### Vista simultánea — 4 variables clave")
    fig_m = make_subplots(rows=4, cols=1, shared_xaxes=True,
        subplot_titles=['Precipitación (mm)','DTW (m)','Saturación (%)','Susceptibilidad (0-1)'],
        vertical_spacing=0.06)
    dv = df.resample('W').mean()
    fig_m.add_trace(go.Bar(x=dv.index, y=dv['precipitacion'],
                            marker_color=C['celeste'], name='Precip.'), row=1, col=1)
    fig_m.add_trace(go.Scatter(x=dv.index, y=dv['dtw'], fill='tozeroy',
                                fillcolor=C['naranja']+'33',
                                line=dict(color=C['naranja']), name='DTW'), row=2, col=1)
    fig_m.add_trace(go.Scatter(x=dv.index, y=dv['saturacion_suelo'], fill='tozeroy',
                                fillcolor=C['verde']+'33',
                                line=dict(color=C['verde']), name='Sat.'), row=3, col=1)
    fig_m.add_trace(go.Scatter(x=dv.index, y=dv['susceptibilidad'], fill='tozeroy',
                                fillcolor=C['morado']+'33',
                                line=dict(color=C['morado']), name='Susc.'), row=4, col=1)
    fig_m.update_yaxes(autorange='reversed', row=2, col=1)
    fig_m.update_layout(height=650, showlegend=False,
                         margin=dict(l=0,r=0,t=40,b=0),
                         plot_bgcolor='white', paper_bgcolor='white')
    st.plotly_chart(fig_m, use_container_width=True)

    st.divider()
    csv = df.reset_index().to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Descargar CSV", csv,
                       file_name="dataset_sintetico_itata.csv",
                       mime="text/csv", use_container_width=True)


# ════════════════════════════════════════════════════════════
#  PÁGINA 3 — ENTRENAMIENTO LSTM
# ════════════════════════════════════════════════════════════
elif pagina == "🏋️ Entrenamiento LSTM":
    st.markdown("## 🏋️ Entrenamiento LSTM en Vivo")
    st.caption("FloodLSTM en NumPy puro · pronóstico 7 días · Motor WAM-IA Fase 2")
    st.divider()

    col_cfg, col_info = st.columns([1,2])
    with col_cfg:
        st.markdown("**Hiperparámetros**")
        ventana   = st.slider("Ventana entrada (días)",    7, 30, 14)
        horizonte = st.slider("Horizonte pronóstico (días)", 3, 14, 7)
        epochs    = st.slider("Épocas",                   10, 80, 30)
        hidden    = st.select_slider("Neuronas LSTM", [16,32,64], value=32)
        lr        = st.select_slider("Learning rate", [0.0005,0.001,0.005], value=0.001)

    with col_info:
        st.markdown(f"""
        **Arquitectura FloodLSTM (NumPy)**
        ```
        Input  →  ({ventana} días × 6 features)
           ↓
        LSTM   ({hidden} neuronas ocultas)
           ↓
        Dense  ({hidden} → {horizonte})
           ↓
        Sigmoid → susceptibilidad t+1 … t+{horizonte}
        ```
        Optimizador **Adam** con gradient clipping.  
        Sin dependencias pesadas — corre en cualquier entorno.
        """)

    st.divider()
    iniciar = st.button("🚀 Iniciar entrenamiento", type="primary",
                         use_container_width=False)

    if iniciar or 'historia' in st.session_state:
        if iniciar:
            for k in ['model','scX','scY','historia','metricas',
                      'ventana','horizonte','y_pred','y_true']:
                st.session_state.pop(k, None)

            np.random.seed(SEED)
            X_seq, y_seq, scX, scY = preparar_secuencias(df, ventana, horizonte)
            n       = len(X_seq)
            n_test  = int(n * 0.2)
            n_val   = int(n * 0.1)
            n_train = n - n_test - n_val

            X_train, y_train = X_seq[:n_train],          y_seq[:n_train]
            X_val,   y_val   = X_seq[n_train:n_train+n_val], y_seq[n_train:n_train+n_val]
            X_test,  y_test  = X_seq[n_train+n_val:],    y_seq[n_train+n_val:]

            model = NumpyLSTM(len(FEATURES), hidden, horizonte, lr)

            st.markdown("#### 📉 Curva de aprendizaje en tiempo real")
            chart_ph    = st.empty()
            status_ph   = st.empty()
            progress_ph = st.progress(0)

            train_losses, val_losses = [], []

            for ep in range(1, epochs+1):
                # Mini-batch manual (shuffle indices)
                idx = np.random.permutation(len(X_train))
                batch_size = 32
                tl_ep = 0; nb = 0
                for start in range(0, len(idx), batch_size):
                    b = idx[start:start+batch_size]
                    bl = 0
                    for j in b:
                        y_p, h, c, cache = model.forward(X_train[j])
                        bl += model.backward(X_train[j], y_train[j], y_p, h, c, cache)
                    tl_ep += bl / len(b); nb += 1
                tl = tl_ep / nb

                # Validación
                vl = 0
                for j in range(len(X_val)):
                    y_p, _, _, _ = model.forward(X_val[j])
                    vl += float(np.mean((y_p - y_val[j])**2))
                vl /= len(X_val)

                train_losses.append(tl)
                val_losses.append(vl)

                if ep % 3 == 0 or ep == epochs:
                    fig_live = go.Figure()
                    fig_live.add_trace(go.Scatter(y=train_losses, name='Train',
                                                   line=dict(color=C['azul'], width=2)))
                    fig_live.add_trace(go.Scatter(y=val_losses, name='Val',
                                                   line=dict(color=C['rojo'],
                                                             width=2, dash='dash')))
                    fig_live.update_layout(
                        height=260, margin=dict(l=0,r=0,t=10,b=0),
                        plot_bgcolor='white', paper_bgcolor='white',
                        xaxis_title='Época', yaxis_title='MSE',
                        yaxis_type='log', legend=dict(orientation='h', y=1.15))
                    chart_ph.plotly_chart(fig_live, use_container_width=True)
                    status_ph.info(f"Época {ep}/{epochs} — Train: {tl:.5f} | Val: {vl:.5f}")
                    progress_ph.progress(ep / epochs)

            progress_ph.empty()
            status_ph.success(f"✅ Entrenamiento completado · Val Loss final: {vl:.5f}")

            # Predicciones test
            y_pred_s = np.array([model.predict(X_test[j]) for j in range(len(X_test))])
            y_true_s = y_test

            def desnorm(arr):
                return np.clip(
                    scY.inverse_transform(arr.reshape(-1,1)).flatten().reshape(arr.shape),
                    0, 1)

            y_pred = desnorm(y_pred_s)
            y_true = desnorm(y_true_s)

            metricas = []
            for d in range(horizonte):
                metricas.append({
                    'Día':  f't+{d+1}',
                    'MAE':  round(mean_absolute_error(y_true[:,d], y_pred[:,d]), 4),
                    'RMSE': round(np.sqrt(mean_squared_error(y_true[:,d], y_pred[:,d])), 4),
                    'R²':   round(r2_score(y_true[:,d], y_pred[:,d]), 4)
                })

            st.session_state.update({
                'model': model, 'scX': scX, 'scY': scY,
                'historia': {'train': train_losses, 'val': val_losses},
                'metricas': metricas, 'ventana': ventana, 'horizonte': horizonte,
                'y_pred': y_pred, 'y_true': y_true
            })

        # ── Métricas ──
        if 'metricas' in st.session_state:
            st.divider()
            st.markdown("#### 📊 Métricas por horizonte")
            df_met = pd.DataFrame(st.session_state['metricas'])

            col_m1, col_m2 = st.columns([1,2])
            with col_m1:
                st.dataframe(df_met.set_index('Día'), use_container_width=True)
                mae_m = df_met['MAE'].mean(); r2_m = df_met['R²'].mean()
                st.metric("MAE promedio", f"{mae_m:.4f}",
                           delta="✅ Cumple TRL-5" if mae_m < 0.08 else "⚠️ Sobre umbral",
                           delta_color="normal" if mae_m < 0.08 else "inverse")
                st.metric("R² promedio", f"{r2_m:.4f}")

            with col_m2:
                fig_m = go.Figure()
                fig_m.add_trace(go.Bar(x=df_met['Día'], y=df_met['MAE'],
                                        name='MAE', marker_color=C['celeste']))
                fig_m.add_trace(go.Scatter(x=df_met['Día'], y=df_met['R²'], name='R²',
                                            mode='lines+markers',
                                            line=dict(color=C['rojo'], width=2),
                                            yaxis='y2'))
                fig_m.add_hline(y=0.08, line_dash='dot', line_color=C['rojo'],
                                 annotation_text='Umbral TRL-5')
                fig_m.update_layout(
                    height=300, margin=dict(l=0,r=0,t=20,b=0),
                    plot_bgcolor='white', paper_bgcolor='white',
                    yaxis2=dict(title='R²', overlaying='y', side='right', range=[0,1]),
                    legend=dict(orientation='h', y=1.15))
                st.plotly_chart(fig_m, use_container_width=True)

            if 'y_pred' in st.session_state:
                st.divider()
                st.markdown("#### 🔍 Real vs Predicho")
                hor_sel = st.selectbox("Horizonte",
                    [f"t+{d+1}" for d in range(st.session_state['horizonte'])])
                d_idx = int(hor_sel.replace('t+','')) - 1
                yp = st.session_state['y_pred'][:150, d_idx]
                yr = st.session_state['y_true'][:150, d_idx]
                fig_c = go.Figure()
                fig_c.add_trace(go.Scatter(y=yr, name='Real',
                                            line=dict(color=C['azul'], width=2)))
                fig_c.add_trace(go.Scatter(y=yp, name='Predicho',
                                            line=dict(color=C['rojo'], width=2, dash='dash')))
                fig_c.add_hline(y=0.6, line_dash='dot', line_color=C['rojo'],
                                 annotation_text='Umbral alerta')
                fig_c.update_layout(
                    height=280, margin=dict(l=0,r=0,t=10,b=0),
                    plot_bgcolor='white', paper_bgcolor='white',
                    yaxis=dict(range=[0,1]),
                    legend=dict(orientation='h', y=1.15))
                st.plotly_chart(fig_c, use_container_width=True)
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
        st.warning("⚠️ **No hay modelo entrenado.**  \nVe a **🏋️ Entrenamiento LSTM**, entrena el modelo y vuelve aquí.")
        st.stop()

    model    = st.session_state['model']
    scX      = st.session_state['scX']
    scY      = st.session_state['scY']
    ventana  = st.session_state['ventana']
    horizonte= st.session_state['horizonte']

    col_f1, col_f2 = st.columns([2,1])
    with col_f1:
        fecha_sel = st.date_input("Fecha base del pronóstico",
                                   value=df.index[-1].date(),
                                   min_value=df.index[ventana].date(),
                                   max_value=df.index[-1].date())
    with col_f2:
        st.markdown(" ")
        if st.button("🎲 Evento extremo aleatorio", use_container_width=True):
            cand = df.index[df['susceptibilidad'] > 0.70]
            if len(cand):
                st.session_state['fecha_sel'] = cand[np.random.randint(len(cand))].date()
                st.rerun()

    if 'fecha_sel' in st.session_state:
        fecha_sel = st.session_state['fecha_sel']

    try:
        idx_base   = df.index.get_loc(pd.Timestamp(fecha_sel))
        datos_vent = df[FEATURES].iloc[idx_base-ventana: idx_base].values
        vent_scl   = scX.transform(datos_vent)
        pred_s     = model.predict(vent_scl)
        pred_real  = np.clip(
            scY.inverse_transform(pred_s.reshape(-1,1)).flatten(), 0, 1)
        fechas_pred= pd.date_range(
            start=pd.Timestamp(fecha_sel)+pd.Timedelta(days=1),
            periods=horizonte, freq='D')

        # Alerta máxima
        max_s = pred_real.max()
        niv, chex, cls = nivel_alerta(max_s)
        st.markdown(f"""
        <div class='alert-{cls}'>
        <b style='font-size:1.2rem'>{niv} — Susceptibilidad máxima: {max_s:.3f}</b><br>
        <span>Pronóstico {horizonte} días desde {str(fecha_sel)}</span>
        </div>""", unsafe_allow_html=True)
        st.markdown("")

        # Tarjetas por día
        cols = st.columns(horizonte)
        for i, (f, s) in enumerate(zip(fechas_pred, pred_real)):
            nv, ch, cl = nivel_alerta(s)
            with cols[i]:
                st.markdown(f"""
                <div style='background:{ch}18;border:2px solid {ch};
                     border-radius:10px;padding:12px;text-align:center'>
                  <div style='font-size:.75rem;color:#555'>{f.strftime('%a %d/%m')}</div>
                  <div style='font-size:1.6rem;font-weight:800;color:{ch}'>{s:.2f}</div>
                  <div style='font-size:.7rem;color:{ch}'>{nv.split(' ',1)[1]}</div>
                </div>""", unsafe_allow_html=True)

        st.divider()
        col_g1, col_g2 = st.columns([3,1])
        with col_g1:
            st.markdown("#### Serie histórica + pronóstico")
            hs = max(0, idx_base-45)
            hf = df.index[hs:idx_base+1]
            hv = df['susceptibilidad'].iloc[hs:idx_base+1]

            fig_b = go.Figure()
            fig_b.add_trace(go.Scatter(x=hf, y=hv, fill='tozeroy',
                                        fillcolor=C['azul']+'22',
                                        line=dict(color=C['azul'], width=2),
                                        name='Histórico real'))
            fig_b.add_vline(x=str(fecha_sel), line_dash='dash',
                             line_color=C['gris'], annotation_text='Hoy')
            for f, s in zip(fechas_pred, pred_real):
                _, ch, _ = nivel_alerta(s)
                fig_b.add_trace(go.Bar(x=[f], y=[s], marker_color=ch,
                                        opacity=.75, showlegend=False, width=86400000))
            fig_b.add_trace(go.Scatter(x=fechas_pred, y=pred_real,
                                        mode='lines+markers',
                                        line=dict(color=C['rojo'], width=2, dash='dot'),
                                        marker=dict(size=8), name='Pronóstico'))
            fig_b.add_hline(y=.6,  line_dash='dot',  line_color=C['naranja'])
            fig_b.add_hline(y=.75, line_dash='dash', line_color=C['rojo'])
            fig_b.update_layout(height=360, margin=dict(l=0,r=0,t=10,b=0),
                                 plot_bgcolor='white', paper_bgcolor='white',
                                 yaxis=dict(range=[0,1.05]),
                                 legend=dict(orientation='h', y=1.12),
                                 barmode='overlay')
            st.plotly_chart(fig_b, use_container_width=True)

        with col_g2:
            st.markdown("#### Condiciones actuales")
            dia = df.iloc[idx_base]
            for lbl, val in [
                ('🌧️ Precip.',   f"{dia['precipitacion']:.1f} mm"),
                ('🌡️ Temp.',     f"{dia['temperatura']:.1f} °C"),
                ('💧 DTW',       f"{dia['dtw']:.2f} m"),
                ('🌱 Saturación',f"{dia['saturacion_suelo']:.1f}%"),
                ('🔮 Susc. hoy', f"{dia['susceptibilidad']:.3f}"),
            ]:
                st.metric(lbl, val)

        st.divider()
        st.markdown("#### Tabla resumen exportable")
        df_bol = pd.DataFrame({
            'Fecha':         [f.strftime('%d/%m/%Y') for f in fechas_pred],
            'Susceptibilidad':[f"{s:.4f}" for s in pred_real],
            'Nivel':         [nivel_alerta(s)[0] for s in pred_real],
            'Acción':        ['Monitoreo continuo' if nivel_alerta(s)[2]=='critico' else
                              'Activar protocolos' if nivel_alerta(s)[2]=='alto' else
                              'Vigilancia estándar' if nivel_alerta(s)[2]=='moderado' else
                              'Sin acción inmediata' for s in pred_real]
        })
        st.dataframe(df_bol, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Descargar boletín CSV",
                           df_bol.to_csv(index=False).encode('utf-8'),
                           file_name=f"boletin_wam_{fecha_sel}.csv",
                           mime="text/csv", use_container_width=True)

    except Exception as e:
        st.error(f"Error al calcular pronóstico: {e}")
