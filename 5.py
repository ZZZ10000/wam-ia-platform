# ============================================================
#  🌊 PLATAFORMA WAM-IA — Motor Híbrido de Inteligencia Hídrica
#  MAKEY × Integra Sur Norte × UNB
#  v3.0 — solo librerías preinstaladas en Streamlit Cloud
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
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
  .alerta-box {
      padding: 14px 18px; border-radius: 8px;
      margin-bottom: 12px; font-size: 1.05rem;
  }
  div[data-testid="stMetricValue"] { color: #1B3A6B !important; }
</style>
""", unsafe_allow_html=True)

FEATURES = ['precipitacion','temperatura','evapotranspiracion',
            'dtw','saturacion_suelo','acumulacion_flujo']
TARGET = 'susceptibilidad'
SEED   = 42

# ════════════════════════════════════════════════════════════
#  LSTM EN NUMPY PURO
# ════════════════════════════════════════════════════════════
class NumpyLSTM:
    def __init__(self, input_size, hidden_size, output_size, lr=0.001):
        self.hs = hidden_size
        self.lr = lr
        s = 0.1
        n = input_size + hidden_size
        self.Wf = np.random.randn(hidden_size, n) * s
        self.Wi = np.random.randn(hidden_size, n) * s
        self.Wc = np.random.randn(hidden_size, n) * s
        self.Wo = np.random.randn(hidden_size, n) * s
        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))
        self.Wy = np.random.randn(output_size, hidden_size) * s
        self.by = np.zeros((output_size, 1))
        self.t  = 0
        self.m  = {k: np.zeros_like(getattr(self,k))
                   for k in ['Wf','Wi','Wc','Wo','bf','bi','bc','bo','Wy','by']}
        self.v  = {k: np.zeros_like(getattr(self,k))
                   for k in ['Wf','Wi','Wc','Wo','bf','bi','bc','bo','Wy','by']}

    @staticmethod
    def sig(x):  return 1/(1+np.exp(-np.clip(x,-15,15)))
    @staticmethod
    def tanh(x): return np.tanh(np.clip(x,-15,15))

    def forward(self, X):
        h = np.zeros((self.hs,1)); c = np.zeros((self.hs,1))
        cache = []
        for t in range(X.shape[0]):
            x  = X[t].reshape(-1,1); xh = np.vstack([x,h])
            f  = self.sig(self.Wf@xh+self.bf)
            i  = self.sig(self.Wi@xh+self.bi)
            g  = self.tanh(self.Wc@xh+self.bc)
            o  = self.sig(self.Wo@xh+self.bo)
            c  = f*c + i*g; h = o*self.tanh(c)
            cache.append((x,xh,f,i,g,o,c,h))
        y = self.sig(self.Wy@h+self.by)
        return y.flatten(), h, c, cache

    def backward(self, X, y_true, y_pred, h_last, c_last, cache):
        yt = np.array(y_true).reshape(-1,1)
        yp = y_pred.reshape(-1,1)
        dy = yp - yt
        dWy = dy@h_last.T; dby = dy.copy()
        dh  = self.Wy.T@dy; dc = np.zeros_like(dh)
        grads = {k: np.zeros_like(getattr(self,k))
                 for k in ['Wf','Wi','Wc','Wo','bf','bi','bc','bo']}
        for t in reversed(range(len(cache))):
            x,xh,f,i,g,o,c_t,h_t = cache[t]
            cp = cache[t-1][6] if t>0 else np.zeros_like(c_t)
            tc = self.tanh(c_t)
            do=dh*tc; dc+=dh*o*(1-tc**2)
            df=dc*cp; di=dc*g; dg=dc*i; dc=dc*f
            ddo=do*o*(1-o); ddf=df*f*(1-f)
            ddi=di*i*(1-i); ddg=dg*(1-g**2)
            for nm,dd in [('Wo',ddo),('Wf',ddf),('Wi',ddi),('Wc',ddg)]:
                grads[nm]+=dd@xh.T; grads[nm.replace('W','b')]+=dd
            dh=(self.Wf.T@ddf+self.Wi.T@ddi+self.Wc.T@ddg+self.Wo.T@ddo)[:self.hs]
        for k in grads: grads[k]=np.clip(grads[k],-1,1)
        dWy=np.clip(dWy,-1,1); dby=np.clip(dby,-1,1)
        self.t+=1; b1,b2,eps=0.9,0.999,1e-8
        for nm,g2 in {**grads,'Wy':dWy,'by':dby}.items():
            self.m[nm]=b1*self.m[nm]+(1-b1)*g2
            self.v[nm]=b2*self.v[nm]+(1-b2)*g2**2
            mh=self.m[nm]/(1-b1**self.t); vh=self.v[nm]/(1-b2**self.t)
            setattr(self,nm,getattr(self,nm)-self.lr*mh/(np.sqrt(vh)+eps))
        return float(np.mean((yp-yt)**2))

    def predict(self, X):
        y,_,_,_ = self.forward(X); return y


# ════════════════════════════════════════════════════════════
#  DATOS SINTÉTICOS
# ════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def generar_datos(n_dias=1825, seed=42, variabilidad=1.0, frec_ext=0.03):
    np.random.seed(seed)
    fechas = pd.date_range('2019-01-01', periods=n_dias, freq='D')
    t = np.arange(n_dias)
    estac  = 0.5*(1-np.cos(2*np.pi*(t%365-160)/365))
    prob   = np.clip(0.05+0.45*estac,0,0.7)
    llueve = np.random.binomial(1,prob)
    intens = np.random.exponential(8*variabilidad,n_dias)
    ext    = np.random.binomial(1,frec_ext,n_dias)
    precip = np.clip(llueve*intens*(1+ext*np.random.uniform(5,15,n_dias)),0,180)
    temp   = 14+8*np.cos(2*np.pi*(t%365-15)/365)+np.random.normal(0,2,n_dias)
    etp    = np.clip(0.15*temp+np.random.normal(0,.5,n_dias),0,None)
    r7     = pd.Series(precip).rolling(7,min_periods=1).sum().values
    r30    = pd.Series(precip).rolling(30,min_periods=1).sum().values
    dtw    = np.zeros(n_dias); dtw[0]=3.5
    for i in range(1,n_dias):
        dtw[i]=max(0.05,min(8.0,dtw[i-1]-0.008*precip[i]-0.003*r7[i]
                              +0.04*etp[i]+0.005+np.random.normal(0,.05)))
    sat  = np.clip(30+2*r7-0.8*dtw*5+np.random.normal(0,3,n_dias),5,100)
    afl  = np.clip(1500+80*r7+25*r30+np.random.normal(0,100,n_dias),200,15000)
    susc = np.clip(
        pd.Series(np.tanh(precip/30)*.35+np.exp(-dtw/2)*.30
                  +(sat/100)**2*.25+np.tanh(afl/5000)*.10
                  ).rolling(3,min_periods=1).mean().values
        +np.random.normal(0,.02,n_dias),0,1)
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
    scX=MinMaxScaler(); scY=MinMaxScaler()
    Xs=scX.fit_transform(df[FEATURES].values)
    ys=scY.fit_transform(df[TARGET].values.reshape(-1,1)).flatten()
    sX,sY=[],[]
    for i in range(len(Xs)-ventana-horizonte+1):
        sX.append(Xs[i:i+ventana]); sY.append(ys[i+ventana:i+ventana+horizonte])
    return np.array(sX),np.array(sY),scX,scY


def nivel_alerta(s):
    if s>=0.75: return "🔴 CRÍTICO",   "#D32F2F","critico"
    if s>=0.60: return "🟠 ALTO",      "#F57C00","alto"
    if s>=0.40: return "🟡 MODERADO",  "#F9A825","moderado"
    return             "🟢 BAJO",      "#388E3C","bajo"


# ════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:16px 0 8px'>
      <div style='font-size:2.5rem'>🌊</div>
      <div style='font-size:1.1rem;font-weight:800'>WAM-IA</div>
      <div style='font-size:.7rem;opacity:.75'>Motor Híbrido Hídrico</div>
      <div style='font-size:.65rem;opacity:.6'>MAKEY × ISN × UNB</div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    pagina = st.radio("Módulos",[
        "📊 Dashboard",
        "🧪 Datos Sintéticos",
        "🏋️ Entrenamiento LSTM",
        "🚨 Alerta Temprana"
    ], label_visibility="collapsed")

    st.divider()
    st.markdown("**Parámetros globales**")
    n_dias       = st.slider("Días a simular",          365,3650,1825,365)
    variabilidad = st.slider("Variabilidad climática",  0.5, 2.0, 1.0, 0.1)
    frec_ext     = st.slider("Frec. eventos extremos", 0.01,0.10,0.03,0.01)

    st.divider()
    if st.button("⚡ Regenerar datos", use_container_width=True):
        st.cache_data.clear()
        for k in ['model','scX','scY','historia','metricas',
                  'ventana','horizonte','y_pred','y_true','fecha_sel']:
            st.session_state.pop(k,None)
        st.rerun()

    st.markdown("""
    <div style='text-align:center;margin-top:20px;font-size:.65rem;opacity:.5'>
    CORFO Innova Alta Tecnología 2025<br>Fase 2 — MVP v3.0
    </div>""", unsafe_allow_html=True)


df = generar_datos(n_dias, SEED, variabilidad, frec_ext)
n_alto    = int((df['susceptibilidad']>0.60).sum())
n_critico = int((df['susceptibilidad']>0.75).sum())


# ════════════════════════════════════════════════════════════
#  PÁGINA 1 — DASHBOARD
# ════════════════════════════════════════════════════════════
if pagina == "📊 Dashboard":
    st.markdown("## 📊 Dashboard WAM-IA — Cuenca Itata, Ñuble")
    st.caption("Visión general del sistema · datos sintéticos 2019-2024")
    st.divider()

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("📅 Días simulados",    f"{len(df):,}")
    c2.metric("🌧️ Precip. media",     f"{df['precipitacion'].mean():.1f} mm")
    c3.metric("💧 DTW promedio",      f"{df['dtw'].mean():.2f} m")
    c4.metric("⚠️ Días riesgo alto",  f"{n_alto}",
              delta=f"{n_alto/len(df)*100:.1f}%", delta_color="inverse")
    c5.metric("🔴 Días críticos",     f"{n_critico}",
              delta=f"{n_critico/len(df)*100:.1f}%", delta_color="inverse")

    st.divider()
    st.markdown("#### Serie temporal — Susceptibilidad semanal")
    dfm = df['susceptibilidad'].resample('W').mean().reset_index()
    dfm.columns = ['fecha','susceptibilidad']

    line = alt.Chart(dfm).mark_area(
        line={'color':'#1B3A6B','strokeWidth':2},
        color=alt.Gradient(
            gradient='linear', x1=0,x2=0,y1=1,y2=0,
            stops=[alt.GradientStop(color='white',offset=0),
                   alt.GradientStop(color='#1B3A6B',offset=1)])
    ).encode(
        x=alt.X('fecha:T', title='Fecha'),
        y=alt.Y('susceptibilidad:Q', title='Susceptibilidad', scale=alt.Scale(domain=[0,1]))
    )
    umbral = alt.Chart(pd.DataFrame({'y':[0.6]})).mark_rule(
        color='red', strokeDash=[4,4], strokeWidth=1.5
    ).encode(y='y:Q')
    st.altair_chart((line+umbral).properties(height=280), use_container_width=True)

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Estacionalidad mensual")
        dfmes = df.copy(); dfmes['mes'] = dfmes.index.month
        pm = dfmes.groupby('mes')[['precipitacion','susceptibilidad']].mean().reset_index()
        meses_map = {1:'Ene',2:'Feb',3:'Mar',4:'Abr',5:'May',6:'Jun',
                     7:'Jul',8:'Ago',9:'Sep',10:'Oct',11:'Nov',12:'Dic'}
        pm['mes_str'] = pm['mes'].map(meses_map)

        bars = alt.Chart(pm).mark_bar(color='#4A90D9', opacity=0.7).encode(
            x=alt.X('mes_str:N', sort=list(meses_map.values()), title='Mes'),
            y=alt.Y('precipitacion:Q', title='Precipitación (mm)')
        )
        line2 = alt.Chart(pm).mark_line(color='#D32F2F',strokeWidth=2.5,point=True).encode(
            x=alt.X('mes_str:N', sort=list(meses_map.values())),
            y=alt.Y('susceptibilidad:Q', title='Susceptibilidad', scale=alt.Scale(domain=[0,1])),
        )
        chart_mes = alt.layer(bars, line2).resolve_scale(y='independent')
        st.altair_chart(chart_mes.properties(height=260), use_container_width=True)

    with col_b:
        st.markdown("#### Distribución de niveles de alerta")
        alertas = pd.DataFrame({
            'Nivel':  ['🟢 Bajo','🟡 Moderado','🟠 Alto','🔴 Crítico'],
            'Días':   [
                int((df['susceptibilidad']<.4).sum()),
                int(((df['susceptibilidad']>=.4)&(df['susceptibilidad']<.6)).sum()),
                int(((df['susceptibilidad']>=.6)&(df['susceptibilidad']<.75)).sum()),
                int((df['susceptibilidad']>=.75).sum())
            ],
            'Color':  ['#388E3C','#F9A825','#F57C00','#D32F2F']
        })
        pie = alt.Chart(alertas).mark_arc(innerRadius=60).encode(
            theta=alt.Theta('Días:Q'),
            color=alt.Color('Nivel:N', scale=alt.Scale(
                domain=alertas['Nivel'].tolist(),
                range=alertas['Color'].tolist())),
            tooltip=['Nivel','Días']
        )
        st.altair_chart(pie.properties(height=260), use_container_width=True)

    st.divider()
    st.markdown("#### Correlación entre variables WAM")
    corr = df.corr().round(2).reset_index().melt('index')
    corr.columns = ['var1','var2','correlacion']
    heatmap = alt.Chart(corr).mark_rect().encode(
        x=alt.X('var1:N', title=''),
        y=alt.Y('var2:N', title=''),
        color=alt.Color('correlacion:Q',
                        scale=alt.Scale(scheme='redblue', domain=[-1,1])),
        tooltip=['var1','var2','correlacion']
    )
    text_corr = alt.Chart(corr).mark_text(fontSize=10).encode(
        x='var1:N', y='var2:N',
        text=alt.Text('correlacion:Q', format='.2f'),
        color=alt.condition(
            alt.datum.correlacion > 0.5, alt.value('white'), alt.value('black'))
    )
    st.altair_chart((heatmap+text_corr).properties(height=360), use_container_width=True)


# ════════════════════════════════════════════════════════════
#  PÁGINA 2 — DATOS SINTÉTICOS
# ════════════════════════════════════════════════════════════
elif pagina == "🧪 Datos Sintéticos":
    st.markdown("## 🧪 Generador de Datos Sintéticos Hidrológicos")
    st.caption("Simulación físicamente coherente · parámetros ajustables en la barra lateral")
    st.divider()

    variable = st.selectbox("Variable a visualizar", FEATURES + ['susceptibilidad'])

    colores_var = {
        'precipitacion':'#4A90D9','temperatura':'#D32F2F',
        'evapotranspiracion':'#546E7A','dtw':'#F57C00',
        'saturacion_suelo':'#388E3C','acumulacion_flujo':'#2E5FA3',
        'susceptibilidad':'#7B1FA2'
    }

    col1, col2 = st.columns([3,1])
    with col1:
        dv = df[[variable]].reset_index()
        dv.columns = ['fecha', variable]
        area_chart = alt.Chart(dv).mark_area(
            line={'color': colores_var[variable], 'strokeWidth':1.5},
            color=alt.Gradient(
                gradient='linear', x1=0,x2=0,y1=1,y2=0,
                stops=[alt.GradientStop(color='white',offset=0),
                       alt.GradientStop(color=colores_var[variable],offset=1)])
        ).encode(
            x=alt.X('fecha:T', title='Fecha'),
            y=alt.Y(f'{variable}:Q', title=variable),
            tooltip=['fecha:T', f'{variable}:Q']
        ).properties(height=320)

        rules = []
        if variable == 'susceptibilidad':
            rules = [0.60, 0.75]
        if variable == 'saturacion_suelo':
            rules = [80.0]

        chart_final = area_chart
        for r in rules:
            rule = alt.Chart(pd.DataFrame({'y':[r]})).mark_rule(
                color='red', strokeDash=[4,4]).encode(y='y:Q')
            chart_final = chart_final + rule

        st.altair_chart(chart_final, use_container_width=True)

    with col2:
        st.markdown(f"**Estadísticas**")
        for k, v in df[variable].describe().items():
            st.metric(k, f"{v:.3f}")

    st.divider()
    st.markdown("#### Panel de 4 variables WAM clave (resumen semanal)")

    dfw = df.resample('W').mean().reset_index()
    dfw.columns = ['fecha'] + list(df.columns)

    vars_panel = [
        ('precipitacion', '#4A90D9', 'bar'),
        ('dtw',           '#F57C00', 'area'),
        ('saturacion_suelo','#388E3C','area'),
        ('susceptibilidad', '#7B1FA2', 'area'),
    ]
    charts = []
    for var, color, tipo in vars_panel:
        base = alt.Chart(dfw).encode(
            x=alt.X('fecha:T', title=''),
            y=alt.Y(f'{var}:Q', title=var),
            tooltip=['fecha:T', f'{var}:Q']
        )
        if tipo == 'bar':
            c = base.mark_bar(color=color, opacity=0.7)
        else:
            c = base.mark_area(
                line={'color':color,'strokeWidth':1.5},
                color=alt.Gradient(gradient='linear',x1=0,x2=0,y1=1,y2=0,
                    stops=[alt.GradientStop(color='white',offset=0),
                           alt.GradientStop(color=color,offset=1)]))
        charts.append(c.properties(height=140, title=var))

    panel = alt.vconcat(*charts, spacing=8)
    st.altair_chart(panel, use_container_width=True)

    st.divider()
    csv = df.reset_index().to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Descargar dataset CSV", csv,
                       "dataset_sintetico_itata.csv","text/csv",
                       use_container_width=True)


# ════════════════════════════════════════════════════════════
#  PÁGINA 3 — ENTRENAMIENTO LSTM
# ════════════════════════════════════════════════════════════
elif pagina == "🏋️ Entrenamiento LSTM":
    st.markdown("## 🏋️ Entrenamiento LSTM en Vivo")
    st.caption("FloodLSTM · NumPy puro · pronóstico 7 días · Motor WAM-IA Fase 2")
    st.divider()

    col_cfg, col_info = st.columns([1,2])
    with col_cfg:
        st.markdown("**Hiperparámetros**")
        ventana   = st.slider("Ventana entrada (días)",      7, 30, 14)
        horizonte = st.slider("Horizonte pronóstico (días)", 3, 14,  7)
        epochs    = st.slider("Épocas de entrenamiento",    10, 60, 25)
        hidden    = st.select_slider("Neuronas LSTM",  [16,32,64], value=32)
        lr        = st.select_slider("Learning rate",
                                     [0.0005,0.001,0.005], value=0.001)
    with col_info:
        st.markdown(f"""
        **Arquitectura FloodLSTM (NumPy)**
        ```
        Input  →  ({ventana} días × 6 features)
           ↓
        LSTM   ({hidden} neuronas)
           ↓
        Dense  ({hidden} → {horizonte}) + Sigmoid
           ↓
        susceptibilidad t+1 … t+{horizonte}
        ```
        Optimizador **Adam** · Gradient clipping ±1  
        Sin torch · corre en cualquier entorno ✅
        """)

    st.divider()
    iniciar = st.button("🚀 Iniciar entrenamiento", type="primary")

    if iniciar or 'historia' in st.session_state:
        if iniciar:
            for k in ['model','scX','scY','historia','metricas',
                      'ventana','horizonte','y_pred','y_true']:
                st.session_state.pop(k, None)

            np.random.seed(SEED)
            X_seq,y_seq,scX,scY = preparar_secuencias(df,ventana,horizonte)
            n=len(X_seq); n_test=int(n*.2); n_val=int(n*.1); n_train=n-n_test-n_val
            X_tr,y_tr = X_seq[:n_train],        y_seq[:n_train]
            X_va,y_va = X_seq[n_train:n_train+n_val], y_seq[n_train:n_train+n_val]
            X_te,y_te = X_seq[n_train+n_val:],  y_seq[n_train+n_val:]

            model = NumpyLSTM(len(FEATURES), hidden, horizonte, lr)

            st.markdown("#### 📉 Curva de aprendizaje")
            chart_ph    = st.empty()
            status_ph   = st.empty()
            prog_ph     = st.progress(0)

            tl_hist, vl_hist = [], []
            BATCH = 32

            for ep in range(1, epochs+1):
                idx = np.random.permutation(len(X_tr))
                tl_ep=0; nb=0
                for start in range(0,len(idx),BATCH):
                    b=idx[start:start+BATCH]; bl=0
                    for j in b:
                        yp,h,c,cache = model.forward(X_tr[j])
                        bl += model.backward(X_tr[j],y_tr[j],yp,h,c,cache)
                    tl_ep+=bl/len(b); nb+=1
                tl = tl_ep/nb

                vl=0
                for j in range(len(X_va)):
                    yp,_,_,_=model.forward(X_va[j])
                    vl+=float(np.mean((yp-y_va[j])**2))
                vl/=len(X_va)
                tl_hist.append(tl); vl_hist.append(vl)

                if ep%3==0 or ep==epochs:
                    df_live = pd.DataFrame({
                        'época': list(range(1,len(tl_hist)+1))*2,
                        'loss':  tl_hist+vl_hist,
                        'tipo':  ['Train']*len(tl_hist)+['Validación']*len(vl_hist)
                    })
                    live_chart = alt.Chart(df_live).mark_line().encode(
                        x=alt.X('época:Q',title='Época'),
                        y=alt.Y('loss:Q',title='MSE Loss',
                                scale=alt.Scale(type='log')),
                        color=alt.Color('tipo:N',
                            scale=alt.Scale(domain=['Train','Validación'],
                                            range=['#1B3A6B','#D32F2F'])),
                        strokeDash=alt.condition(
                            alt.datum.tipo=='Validación',
                            alt.value([6,3]), alt.value([1,0]))
                    ).properties(height=240)
                    chart_ph.altair_chart(live_chart, use_container_width=True)
                    status_ph.info(
                        f"Época {ep}/{epochs} — Train: {tl:.5f} | Val: {vl:.5f}")
                    prog_ph.progress(ep/epochs)

            prog_ph.empty()
            status_ph.success(f"✅ Completado · Val Loss final: {vl:.5f}")

            y_pred_s = np.array([model.predict(X_te[j]) for j in range(len(X_te))])
            def desnorm(a):
                return np.clip(scY.inverse_transform(
                    a.reshape(-1,1)).flatten().reshape(a.shape),0,1)
            y_pred = desnorm(y_pred_s); y_true = desnorm(y_te)

            metricas=[]
            for d in range(horizonte):
                metricas.append({
                    'Día':  f't+{d+1}',
                    'MAE':  round(mean_absolute_error(y_true[:,d],y_pred[:,d]),4),
                    'RMSE': round(np.sqrt(mean_squared_error(y_true[:,d],y_pred[:,d])),4),
                    'R²':   round(r2_score(y_true[:,d],y_pred[:,d]),4)
                })
            st.session_state.update({
                'model':model,'scX':scX,'scY':scY,
                'historia':{'train':tl_hist,'val':vl_hist},
                'metricas':metricas,'ventana':ventana,'horizonte':horizonte,
                'y_pred':y_pred,'y_true':y_true
            })

        if 'metricas' in st.session_state:
            st.divider()
            st.markdown("#### 📊 Métricas por horizonte")
            df_met = pd.DataFrame(st.session_state['metricas'])
            col_m1, col_m2 = st.columns([1,2])
            with col_m1:
                st.dataframe(df_met.set_index('Día'), use_container_width=True)
                mae_m = df_met['MAE'].mean(); r2_m = df_met['R²'].mean()
                st.metric("MAE promedio", f"{mae_m:.4f}",
                           delta="✅ Cumple TRL-5" if mae_m<.08 else "⚠️ Sobre umbral",
                           delta_color="normal" if mae_m<.08 else "inverse")
                st.metric("R² promedio", f"{r2_m:.4f}")
            with col_m2:
                df_met_long = df_met.melt('Día',['MAE','RMSE','R²'],'métrica','valor')
                bar_met = alt.Chart(df_met_long[df_met_long['métrica']!='R²']).mark_bar(
                    opacity=.8).encode(
                    x='Día:N',
                    y=alt.Y('valor:Q',title='Valor'),
                    color=alt.Color('métrica:N',
                        scale=alt.Scale(domain=['MAE','RMSE'],
                                        range=['#4A90D9','#1B3A6B'])),
                    xOffset='métrica:N'
                )
                r2_line = alt.Chart(df_met).mark_line(
                    color='#D32F2F',strokeWidth=2,point=True).encode(
                    x='Día:N',
                    y=alt.Y('R²:Q',title='R²',scale=alt.Scale(domain=[0,1]))
                )
                rule_trl = alt.Chart(pd.DataFrame({'y':[.08]})).mark_rule(
                    color='red',strokeDash=[4,4]).encode(y='y:Q')
                combo = alt.layer(bar_met, rule_trl).resolve_scale(y='shared')
                st.altair_chart(combo.properties(height=280), use_container_width=True)

            if 'y_pred' in st.session_state:
                st.divider()
                st.markdown("#### 🔍 Real vs Predicho")
                hz = st.session_state['horizonte']
                hor_sel = st.selectbox("Horizonte", [f"t+{d+1}" for d in range(hz)])
                d_idx = int(hor_sel.replace('t+',''))-1
                n_show = min(150, len(st.session_state['y_pred']))
                df_cmp = pd.DataFrame({
                    'muestra': list(range(n_show))*2,
                    'susceptibilidad': (
                        list(st.session_state['y_true'][:n_show,d_idx])+
                        list(st.session_state['y_pred'][:n_show,d_idx])),
                    'serie': (['Real']*n_show + ['Predicho']*n_show)
                })
                cmp_chart = alt.Chart(df_cmp).mark_line().encode(
                    x=alt.X('muestra:Q',title='Muestras test'),
                    y=alt.Y('susceptibilidad:Q',scale=alt.Scale(domain=[0,1])),
                    color=alt.Color('serie:N',
                        scale=alt.Scale(domain=['Real','Predicho'],
                                        range=['#1B3A6B','#D32F2F'])),
                    strokeDash=alt.condition(
                        alt.datum.serie=='Predicho',
                        alt.value([6,3]),alt.value([1,0]))
                )
                umbral_line = alt.Chart(
                    pd.DataFrame({'y':[0.6]})).mark_rule(
                    color='orange',strokeDash=[4,4]).encode(y='y:Q')
                st.altair_chart(
                    (cmp_chart+umbral_line).properties(height=260),
                    use_container_width=True)

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
        
        Ve a **🏋️ Entrenamiento LSTM**, entrena el modelo y vuelve aquí.
        """)
        st.stop()

    model     = st.session_state['model']
    scX       = st.session_state['scX']
    scY       = st.session_state['scY']
    ventana   = st.session_state['ventana']
    horizonte = st.session_state['horizonte']

    col_f1, col_f2 = st.columns([2,1])
    with col_f1:
        fecha_sel = st.date_input(
            "Fecha base del pronóstico",
            value=df.index[-1].date(),
            min_value=df.index[ventana].date(),
            max_value=df.index[-1].date())
    with col_f2:
        st.markdown(" ")
        if st.button("🎲 Evento extremo aleatorio", use_container_width=True):
            cand = df.index[df['susceptibilidad']>0.70]
            if len(cand):
                st.session_state['fecha_sel'] = cand[
                    np.random.randint(len(cand))].date()
                st.rerun()

    if 'fecha_sel' in st.session_state:
        fecha_sel = st.session_state['fecha_sel']

    try:
        idx_base   = df.index.get_loc(pd.Timestamp(fecha_sel))
        datos_vent = df[FEATURES].iloc[idx_base-ventana:idx_base].values
        vent_scl   = scX.transform(datos_vent)
        pred_s     = model.predict(vent_scl)
        pred_real  = np.clip(
            scY.inverse_transform(pred_s.reshape(-1,1)).flatten(), 0, 1)
        fechas_pred = pd.date_range(
            start=pd.Timestamp(fecha_sel)+pd.Timedelta(days=1),
            periods=horizonte, freq='D')

        # Alerta principal
        max_s = float(pred_real.max())
        niv, chex, cls = nivel_alerta(max_s)
        fondo = {"critico":"#FFEBEE","alto":"#FFF3E0",
                 "moderado":"#FFFDE7","bajo":"#E8F5E9"}
        st.markdown(f"""
        <div class='alerta-box' style='background:{fondo[cls]};
             border-left:5px solid {chex}'>
          <span style='font-size:1.3rem;font-weight:800;color:{chex}'>{niv}</span>
          &nbsp;&nbsp;Susceptibilidad máxima: <b>{max_s:.3f}</b><br>
          <span style='font-size:.9rem;color:#555'>
          Pronóstico {horizonte} días desde {str(fecha_sel)}</span>
        </div>""", unsafe_allow_html=True)

        # Tarjetas por día
        cols = st.columns(horizonte)
        for i,(f,s) in enumerate(zip(fechas_pred,pred_real)):
            nv,ch,cl = nivel_alerta(float(s))
            with cols[i]:
                st.markdown(f"""
                <div style='background:{ch}15;border:2px solid {ch};
                     border-radius:10px;padding:12px;text-align:center'>
                  <div style='font-size:.72rem;color:#555'>{f.strftime('%a %d/%m')}</div>
                  <div style='font-size:1.6rem;font-weight:800;color:{ch}'>{s:.2f}</div>
                  <div style='font-size:.68rem;color:{ch}'>{nv.split(' ',1)[1]}</div>
                </div>""", unsafe_allow_html=True)

        st.divider()
        col_g1, col_g2 = st.columns([3,1])

        with col_g1:
            st.markdown("#### Serie histórica + pronóstico")
            hs = max(0, idx_base-45)
            hist_df = df['susceptibilidad'].iloc[hs:idx_base+1].reset_index()
            hist_df.columns = ['fecha','susceptibilidad']
            hist_df['tipo'] = 'Histórico'

            pred_df = pd.DataFrame({
                'fecha': fechas_pred,
                'susceptibilidad': pred_real,
                'tipo': ['Pronóstico']*horizonte
            })
            pred_df['color_hex'] = [nivel_alerta(float(s))[1] for s in pred_real]

            hist_line = alt.Chart(hist_df).mark_area(
                line={'color':'#1B3A6B','strokeWidth':2},
                color=alt.Gradient(gradient='linear',x1=0,x2=0,y1=1,y2=0,
                    stops=[alt.GradientStop(color='white',offset=0),
                           alt.GradientStop(color='#1B3A6B33',offset=1)])
            ).encode(
                x=alt.X('fecha:T',title='Fecha'),
                y=alt.Y('susceptibilidad:Q',
                        scale=alt.Scale(domain=[0,1.05]),title='Susceptibilidad'),
                tooltip=['fecha:T','susceptibilidad:Q']
            )
            pred_line = alt.Chart(pred_df).mark_line(
                color='#D32F2F',strokeWidth=2,strokeDash=[6,3],
                point=alt.OverlayMarkDef(color='#D32F2F',size=80)
            ).encode(
                x='fecha:T',
                y=alt.Y('susceptibilidad:Q',scale=alt.Scale(domain=[0,1.05])),
                tooltip=['fecha:T','susceptibilidad:Q']
            )
            pred_bars = alt.Chart(pred_df).mark_bar(opacity=0.4,width=18).encode(
                x='fecha:T',
                y=alt.Y('susceptibilidad:Q',scale=alt.Scale(domain=[0,1.05])),
                color=alt.Color('color_hex:N',
                    scale=alt.Scale(domain=pred_df['color_hex'].tolist(),
                                    range=pred_df['color_hex'].tolist()),
                    legend=None)
            )
            u60 = alt.Chart(pd.DataFrame({'y':[.6]})).mark_rule(
                color='orange',strokeDash=[4,4],strokeWidth=1.5).encode(y='y:Q')
            u75 = alt.Chart(pd.DataFrame({'y':[.75]})).mark_rule(
                color='red',strokeDash=[6,3],strokeWidth=1.5).encode(y='y:Q')

            boletin_chart = alt.layer(
                hist_line, pred_bars, pred_line, u60, u75
            ).properties(height=340)
            st.altair_chart(boletin_chart, use_container_width=True)

        with col_g2:
            st.markdown("#### Condiciones actuales")
            dia = df.iloc[idx_base]
            for lbl,val in [
                ('🌧️ Precip.',    f"{dia['precipitacion']:.1f} mm"),
                ('🌡️ Temp.',      f"{dia['temperatura']:.1f} °C"),
                ('💧 DTW',        f"{dia['dtw']:.2f} m"),
                ('🌱 Saturación', f"{dia['saturacion_suelo']:.1f}%"),
                ('🔮 Susc. hoy',  f"{dia['susceptibilidad']:.3f}"),
            ]:
                st.metric(lbl, val)

        st.divider()
        st.markdown("#### Tabla resumen")
        df_bol = pd.DataFrame({
            'Fecha':          [f.strftime('%d/%m/%Y') for f in fechas_pred],
            'Susceptibilidad':[f"{s:.4f}" for s in pred_real],
            'Nivel':          [nivel_alerta(float(s))[0] for s in pred_real],
            'Acción':         [
                'Monitoreo continuo'  if nivel_alerta(float(s))[2]=='critico' else
                'Activar protocolos'  if nivel_alerta(float(s))[2]=='alto'    else
                'Vigilancia estándar' if nivel_alerta(float(s))[2]=='moderado'else
                'Sin acción inmediata'
                for s in pred_real]
        })
        st.dataframe(df_bol, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Descargar boletín CSV",
            df_bol.to_csv(index=False).encode('utf-8'),
            file_name=f"boletin_wam_{fecha_sel}.csv",
            mime="text/csv", use_container_width=True)

    except Exception as e:
        st.error(f"Error al calcular pronóstico: {e}")

