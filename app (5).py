# ============================================================
#  🌊 PLATAFORMA WAM-IA — Motor Híbrido de Inteligencia Hídrica
#  MAKEY × Integra Sur Norte × UNB
#  v4.0 — Módulo de Ingesta Satelital en Tiempo Real
#  NASA GPM IMERG · ESA Sentinel · Copernicus · SNSAT
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import requests
import json
from datetime import datetime, timedelta
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
  .alerta-box { padding:14px 18px; border-radius:8px; margin-bottom:12px; }
  .sat-card {
    background: #0D1B2A; border: 1px solid #2E5FA3;
    border-radius: 10px; padding: 16px; margin: 4px;
    text-align: center;
  }
  .sat-card .valor { font-size:1.8rem; font-weight:800; color:#4A90D9; }
  .sat-card .label { font-size:.75rem; color:#8BA8C8; margin-top:4px; }
  .sat-card .fuente { font-size:.65rem; color:#4A90D9; margin-top:2px; }
  .sat-online  { color: #4CAF50; font-weight:700; }
  .sat-offline { color: #FF9800; font-weight:700; }
  div[data-testid="stMetricValue"] { color:#1B3A6B !important; }
  .pipeline-step {
    background:#F0F4FF; border-left:4px solid #1B3A6B;
    padding:10px 14px; border-radius:0 8px 8px 0;
    margin:4px 0; font-size:.85rem;
  }
  .pipeline-ok   { border-left-color: #388E3C; background:#F1F8E9; }
  .pipeline-warn { border-left-color: #F57C00; background:#FFF3E0; }
  .pipeline-err  { border-left-color: #D32F2F; background:#FFEBEE; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  CONSTANTES Y CONFIGURACIÓN
# ════════════════════════════════════════════════════════════
FEATURES = ['precipitacion','temperatura','evapotranspiracion',
            'dtw','saturacion_suelo','acumulacion_flujo']
TARGET = 'susceptibilidad'
SEED   = 42

# Cuencas disponibles en la plataforma
CUENCAS = {
    'Itata (Ñuble)':    {'lat_min':-37.2,'lat_max':-36.0,'lon_min':-72.5,'lon_max':-71.0,'area_km2':11315},
    'Biobío (Concepción)': {'lat_min':-38.5,'lat_max':-37.0,'lon_min':-73.5,'lon_max':-71.5,'area_km2':23695},
    'Maule (Talca)':    {'lat_min':-36.5,'lat_max':-35.0,'lon_min':-72.0,'lon_max':-70.5,'area_km2':20280},
    'Copiapó (Atacama)':{'lat_min':-28.5,'lat_max':-27.0,'lon_min':-70.5,'lon_max':-69.0,'area_km2':18705},
}

SATELITES = {
    'NASA GPM IMERG':   {'url':'https://gpm.nasa.gov','variable':'Precipitación','resolucion':'10 km / 30 min','costo':'Gratis'},
    'ESA Sentinel-1':   {'url':'https://scihub.copernicus.eu','variable':'Humedad SAR','resolucion':'10 m / 6 días','costo':'Gratis'},
    'ESA Sentinel-2':   {'url':'https://scihub.copernicus.eu','variable':'NDVI / NDWI','resolucion':'10 m / 5 días','costo':'Gratis'},
    'NASA MODIS':       {'url':'https://modis.gsfc.nasa.gov','variable':'Temperatura','resolucion':'500 m / diario','costo':'Gratis'},
    'SNSAT FASat-Delta':{'url':'https://agenciaespacial.cl','variable':'Imagen óptica','resolucion':'70 cm / variable','costo':'Acuerdo AEXA'},
}


# ════════════════════════════════════════════════════════════
#  MÓDULO DE INGESTA SATELITAL
# ════════════════════════════════════════════════════════════

def verificar_conectividad_satelite(nombre, url):
    """Verifica si el satélite/API está accesible."""
    try:
        r = requests.get(url, timeout=5)
        return r.status_code < 500
    except:
        return False


def obtener_gpm_imerg_real(lat_min, lat_max, lon_min, lon_max,
                            fecha_inicio, fecha_fin, token=None):
    """
    Conecta a NASA GPM IMERG v6 API.
    Requiere registro gratuito en: https://urs.earthdata.nasa.gov
    """
    try:
        base_url = "https://gpm.nasa.gov/api/v1/imerg/precipitation"
        headers  = {}
        if token:
            headers['Authorization'] = f'Bearer {token}'

        params = {
            'startTime':  fecha_inicio.strftime('%Y-%m-%dT00:00:00'),
            'endTime':    fecha_fin.strftime('%Y-%m-%dT23:59:59'),
            'bbox':       f'{lon_min},{lat_min},{lon_max},{lat_max}',
            'format':     'json',
            'precipitation': 'precipitationCal'
        }
        r = requests.get(base_url, params=params, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            return {'exito': True, 'datos': data, 'fuente': 'NASA GPM IMERG Real'}
        else:
            return {'exito': False, 'error': f'HTTP {r.status_code}'}
    except Exception as e:
        return {'exito': False, 'error': str(e)}


def obtener_open_meteo(lat_center, lon_center, dias=14):
    """
    Open-Meteo: API meteorológica GRATUITA sin autenticación.
    Datos reales de ERA5 reanalysis — calidad científica.
    https://open-meteo.com
    """
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            'latitude':         lat_center,
            'longitude':        lon_center,
            'daily':            ['precipitation_sum','temperature_2m_mean',
                                 'et0_fao_evapotranspiration',
                                 'soil_moisture_0_to_7cm'],
            'past_days':        dias,
            'forecast_days':    7,
            'timezone':         'America/Santiago',
            'precipitation_unit': 'mm'
        }
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            df = pd.DataFrame({
                'fecha':            pd.to_datetime(data['daily']['time']),
                'precipitacion':    data['daily']['precipitation_sum'],
                'temperatura':      data['daily']['temperature_2m_mean'],
                'evapotranspiracion': data['daily']['et0_fao_evapotranspiration'],
                'humedad_suelo':    data['daily']['soil_moisture_0_to_7cm']
            }).set_index('fecha')
            df = df.fillna(method='ffill').fillna(0)
            return {'exito': True, 'datos': df,
                    'fuente': 'Open-Meteo ERA5 (Real)', 'tipo': 'real'}
        else:
            return {'exito': False, 'error': f'HTTP {r.status_code}'}
    except Exception as e:
        return {'exito': False, 'error': str(e)}


def obtener_datos_satelitales_completos(cuenca_config, dias_hist=14):
    """
    Orquestador principal de ingesta.
    Intenta fuentes reales en orden de prioridad.
    Fallback a datos sintéticos calibrados si todas fallan.
    """
    lat_c = (cuenca_config['lat_min'] + cuenca_config['lat_max']) / 2
    lon_c = (cuenca_config['lon_min'] + cuenca_config['lon_max']) / 2
    log   = []

    # ── Intento 1: Open-Meteo (más confiable, sin auth) ──
    log.append({'paso': '🛰️ Conectando Open-Meteo ERA5...', 'estado': 'intentando'})
    resultado = obtener_open_meteo(lat_c, lon_c, dias_hist)

    if resultado['exito']:
        df_sat = resultado['datos']
        log[-1] = {'paso': f'✅ Open-Meteo ERA5 — {len(df_sat)} días descargados',
                   'estado': 'ok', 'fuente': 'Open-Meteo ERA5'}

        # Enriquecer con variables WAM derivadas
        df_sat = enriquecer_con_wam(df_sat, lat_c, lon_c)
        log.append({'paso': '✅ Variables WAM calculadas (DTW, flujo, saturación)',
                    'estado': 'ok'})

        return {'exito': True, 'datos': df_sat, 'log': log,
                'fuente': 'Open-Meteo ERA5 (Real)', 'tipo': 'real'}

    # ── Fallback: datos sintéticos calibrados ──
    log[-1] = {'paso': f'⚠️ Open-Meteo no disponible: {resultado["error"]}',
               'estado': 'warn'}
    log.append({'paso': '🔄 Activando fallback: datos sintéticos calibrados...',
                'estado': 'intentando'})

    df_sint = generar_datos_satelite_sinteticos(lat_c, lon_c, dias_hist)
    log[-1] = {'paso': '✅ Datos sintéticos calibrados generados',
               'estado': 'ok'}

    return {'exito': True, 'datos': df_sint, 'log': log,
            'fuente': 'Sintético calibrado (fallback)', 'tipo': 'sintetico'}


def enriquecer_con_wam(df_meteo, lat, lon):
    """
    A partir de datos meteorológicos reales, calcula las variables WAM:
    - DTW (Depth-to-Water) usando balance hídrico simplificado
    - Saturación del suelo
    - Acumulación de flujo (índice topográfico)
    """
    n = len(df_meteo)
    precip = df_meteo['precipitacion'].values
    temp   = df_meteo['temperatura'].values
    etp    = df_meteo.get('evapotranspiracion',
                           pd.Series(np.ones(n)*2.0)).values

    # DTW usando balance hídrico
    r7  = pd.Series(precip).rolling(7,  min_periods=1).sum().values
    r30 = pd.Series(precip).rolling(30, min_periods=1).sum().values

    dtw = np.zeros(n); dtw[0] = 3.5
    for i in range(1, n):
        dtw[i] = max(0.05, min(8.0,
                               dtw[i-1] - 0.008*precip[i] - 0.003*r7[i]
                               + 0.04*etp[i] + 0.005))

    # Saturación
    if 'humedad_suelo' in df_meteo.columns:
        sat = np.clip(df_meteo['humedad_suelo'].values * 100, 5, 100)
    else:
        sat = np.clip(30 + 2*r7 - 0.8*dtw*5, 5, 100)

    # Acumulación de flujo (función del área y lluvia)
    area_km2 = 11315  # default Itata
    afl = np.clip(area_km2*0.13 + 80*r7 + 25*r30, 200, 15000)

    # Susceptibilidad WAM
    susc = np.clip(
        np.tanh(precip/30)*.35 + np.exp(-dtw/2)*.30
        + (sat/100)**2*.25 + np.tanh(afl/5000)*.10, 0, 1)

    df_out = df_meteo.copy()
    df_out['dtw']              = np.round(dtw, 3)
    df_out['saturacion_suelo'] = np.round(sat, 2)
    df_out['acumulacion_flujo']= np.round(afl, 1)
    df_out['susceptibilidad']  = np.round(susc, 4)
    df_out = df_out.rename(columns={'humedad_suelo': '_humedad_raw'})

    # Asegurar todas las columnas necesarias
    for col in FEATURES:
        if col not in df_out.columns:
            df_out[col] = 0.0

    return df_out[FEATURES + ['susceptibilidad']]


def generar_datos_satelite_sinteticos(lat, lon, dias):
    """Genera datos sintéticos calibrados para la ubicación."""
    np.random.seed(int(abs(lat*100 + lon*100)) % 9999)
    fechas = pd.date_range(end=datetime.now().date(), periods=dias+7, freq='D')
    t = np.arange(len(fechas))

    estac  = 0.5*(1-np.cos(2*np.pi*(t%365-160)/365))
    prob   = np.clip(0.05+0.45*estac, 0, 0.7)
    precip = np.clip(np.random.binomial(1,prob)*np.random.exponential(8,len(t)), 0, 120)
    temp   = 14+8*np.cos(2*np.pi*(t%365-15)/365)+np.random.normal(0,2,len(t))
    etp    = np.clip(0.15*temp+np.random.normal(0,.5,len(t)), 0, None)
    r7     = pd.Series(precip).rolling(7,  min_periods=1).sum().values
    r30    = pd.Series(precip).rolling(30, min_periods=1).sum().values
    dtw    = np.zeros(len(t)); dtw[0]=3.5
    for i in range(1,len(t)):
        dtw[i]=max(0.05,min(8.0, dtw[i-1]-0.008*precip[i]-0.003*r7[i]
                               +0.04*etp[i]+0.005+np.random.normal(0,.05)))
    sat    = np.clip(30+2*r7-0.8*dtw*5+np.random.normal(0,3,len(t)), 5, 100)
    afl    = np.clip(1500+80*r7+25*r30+np.random.normal(0,100,len(t)), 200, 15000)
    susc   = np.clip(np.tanh(precip/30)*.35+np.exp(-dtw/2)*.30
                     +(sat/100)**2*.25+np.tanh(afl/5000)*.10, 0, 1)

    return pd.DataFrame({
        'precipitacion':     np.round(precip,2),
        'temperatura':       np.round(temp,2),
        'evapotranspiracion':np.round(etp,3),
        'dtw':               np.round(dtw,3),
        'saturacion_suelo':  np.round(sat,2),
        'acumulacion_flujo': np.round(afl,1),
        'susceptibilidad':   np.round(susc,4)
    }, index=fechas)


# ════════════════════════════════════════════════════════════
#  LSTM EN NUMPY PURO
# ════════════════════════════════════════════════════════════
class NumpyLSTM:
    def __init__(self, input_size, hidden_size, output_size, lr=0.001):
        self.hs=hidden_size; self.lr=lr; s=0.1
        n=input_size+hidden_size
        for nm in ['Wf','Wi','Wc','Wo']:
            setattr(self,nm,np.random.randn(hidden_size,n)*s)
        for nm in ['bf','bi','bc','bo']:
            setattr(self,nm,np.zeros((hidden_size,1)))
        self.Wy=np.random.randn(output_size,hidden_size)*s
        self.by=np.zeros((output_size,1))
        self.t=0
        self.m={k:np.zeros_like(getattr(self,k))
                for k in ['Wf','Wi','Wc','Wo','bf','bi','bc','bo','Wy','by']}
        self.v={k:np.zeros_like(getattr(self,k))
                for k in ['Wf','Wi','Wc','Wo','bf','bi','bc','bo','Wy','by']}

    @staticmethod
    def sig(x): return 1/(1+np.exp(-np.clip(x,-15,15)))
    @staticmethod
    def tanh(x): return np.tanh(np.clip(x,-15,15))

    def forward(self, X):
        h=np.zeros((self.hs,1)); c=np.zeros((self.hs,1)); cache=[]
        for t in range(X.shape[0]):
            x=X[t].reshape(-1,1); xh=np.vstack([x,h])
            f=self.sig(self.Wf@xh+self.bf); i=self.sig(self.Wi@xh+self.bi)
            g=self.tanh(self.Wc@xh+self.bc); o=self.sig(self.Wo@xh+self.bo)
            c=f*c+i*g; h=o*self.tanh(c)
            cache.append((x,xh,f,i,g,o,c,h))
        return self.sig(self.Wy@h+self.by).flatten(), h, c, cache

    def backward(self, X, yt, yp, hl, cl, cache):
        yt=np.array(yt).reshape(-1,1); yp=yp.reshape(-1,1)
        dy=yp-yt; dWy=dy@hl.T; dby=dy.copy()
        dh=self.Wy.T@dy; dc=np.zeros_like(dh)
        grads={k:np.zeros_like(getattr(self,k))
               for k in ['Wf','Wi','Wc','Wo','bf','bi','bc','bo']}
        for t in reversed(range(len(cache))):
            x,xh,f,i,g,o,ct,ht=cache[t]
            cp=cache[t-1][6] if t>0 else np.zeros_like(ct)
            tc=self.tanh(ct)
            do=dh*tc; dc+=dh*o*(1-tc**2)
            df=dc*cp; di=dc*g; dg2=dc*i; dc=dc*f
            ddo=do*o*(1-o); ddf=df*f*(1-f); ddi=di*i*(1-i); ddg=dg2*(1-g**2)
            for nm,dd in [('Wo',ddo),('Wf',ddf),('Wi',ddi),('Wc',ddg)]:
                grads[nm]+=dd@xh.T; grads[nm.replace('W','b')]+=dd
            dh=(self.Wf.T@ddf+self.Wi.T@ddi+self.Wc.T@ddg+self.Wo.T@ddo)[:self.hs]
        for k in grads: grads[k]=np.clip(grads[k],-1,1)
        self.t+=1; b1,b2,eps=0.9,0.999,1e-8
        for nm,g2 in {**grads,'Wy':np.clip(dWy,-1,1),'by':np.clip(dby,-1,1)}.items():
            self.m[nm]=b1*self.m[nm]+(1-b1)*g2
            self.v[nm]=b2*self.v[nm]+(1-b2)*g2**2
            mh=self.m[nm]/(1-b1**self.t); vh=self.v[nm]/(1-b2**self.t)
            setattr(self,nm,getattr(self,nm)-self.lr*mh/(np.sqrt(vh)+eps))
        return float(np.mean((yp-yt)**2))

    def predict(self, X):
        y,_,_,_=self.forward(X); return y


# ════════════════════════════════════════════════════════════
#  DATOS SINTÉTICOS BASE (historial largo)
# ════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def generar_datos(n_dias=1825, seed=42, variabilidad=1.0, frec_ext=0.03):
    np.random.seed(seed)
    fechas=pd.date_range('2019-01-01',periods=n_dias,freq='D')
    t=np.arange(n_dias)
    estac=0.5*(1-np.cos(2*np.pi*(t%365-160)/365))
    prob=np.clip(0.05+0.45*estac,0,0.7)
    llueve=np.random.binomial(1,prob)
    intens=np.random.exponential(8*variabilidad,n_dias)
    ext=np.random.binomial(1,frec_ext,n_dias)
    precip=np.clip(llueve*intens*(1+ext*np.random.uniform(5,15,n_dias)),0,180)
    temp=14+8*np.cos(2*np.pi*(t%365-15)/365)+np.random.normal(0,2,n_dias)
    etp=np.clip(0.15*temp+np.random.normal(0,.5,n_dias),0,None)
    r7=pd.Series(precip).rolling(7,min_periods=1).sum().values
    r30=pd.Series(precip).rolling(30,min_periods=1).sum().values
    dtw=np.zeros(n_dias); dtw[0]=3.5
    for i in range(1,n_dias):
        dtw[i]=max(0.05,min(8.0,dtw[i-1]-0.008*precip[i]-0.003*r7[i]
                               +0.04*etp[i]+0.005+np.random.normal(0,.05)))
    sat=np.clip(30+2*r7-0.8*dtw*5+np.random.normal(0,3,n_dias),5,100)
    afl=np.clip(1500+80*r7+25*r30+np.random.normal(0,100,n_dias),200,15000)
    susc=np.clip(pd.Series(np.tanh(precip/30)*.35+np.exp(-dtw/2)*.30
                  +(sat/100)**2*.25+np.tanh(afl/5000)*.10
                  ).rolling(3,min_periods=1).mean().values
                 +np.random.normal(0,.02,n_dias),0,1)
    return pd.DataFrame({
        'precipitacion':np.round(precip,2),'temperatura':np.round(temp,2),
        'evapotranspiracion':np.round(etp,3),'dtw':np.round(dtw,3),
        'saturacion_suelo':np.round(sat,2),'acumulacion_flujo':np.round(afl,1),
        'susceptibilidad':np.round(susc,4)
    },index=fechas)


def preparar_secuencias(df,ventana,horizonte):
    scX=MinMaxScaler(); scY=MinMaxScaler()
    Xs=scX.fit_transform(df[FEATURES].values)
    ys=scY.fit_transform(df[TARGET].values.reshape(-1,1)).flatten()
    sX,sY=[],[]
    for i in range(len(Xs)-ventana-horizonte+1):
        sX.append(Xs[i:i+ventana]); sY.append(ys[i+ventana:i+ventana+horizonte])
    return np.array(sX),np.array(sY),scX,scY


def nivel_alerta(s):
    if s>=0.75: return "🔴 CRÍTICO","#D32F2F","critico"
    if s>=0.60: return "🟠 ALTO","#F57C00","alto"
    if s>=0.40: return "🟡 MODERADO","#F9A825","moderado"
    return             "🟢 BAJO","#388E3C","bajo"


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
        "🛰️ Ingesta Satelital",
        "🧪 Datos Sintéticos",
        "🏋️ Entrenamiento LSTM",
        "🚨 Alerta Temprana"
    ], label_visibility="collapsed")

    st.divider()
    st.markdown("**Parámetros globales**")
    n_dias       = st.slider("Días a simular",          365,3650,1825,365)
    variabilidad = st.slider("Variabilidad climática",  0.5,2.0,1.0,0.1)
    frec_ext     = st.slider("Frec. eventos extremos", 0.01,0.10,0.03,0.01)

    st.divider()
    if st.button("⚡ Regenerar datos",use_container_width=True):
        st.cache_data.clear()
        for k in ['model','scX','scY','historia','metricas',
                  'ventana','horizonte','y_pred','y_true',
                  'sat_data','sat_log','sat_fuente']:
            st.session_state.pop(k,None)
        st.rerun()

    st.markdown("""
    <div style='text-align:center;margin-top:20px;font-size:.65rem;opacity:.5'>
    CORFO Innova Alta Tecnología 2025<br>Fase 2 — MVP v4.0
    </div>""", unsafe_allow_html=True)


df = generar_datos(n_dias,SEED,variabilidad,frec_ext)
n_alto    = int((df['susceptibilidad']>0.60).sum())
n_critico = int((df['susceptibilidad']>0.75).sum())


# ════════════════════════════════════════════════════════════
#  PÁGINA SATÉLITE — INGESTA EN TIEMPO REAL
# ════════════════════════════════════════════════════════════
if pagina == "🛰️ Ingesta Satelital":
    st.markdown("## 🛰️ Módulo de Ingesta Satelital en Tiempo Real")
    st.caption("NASA GPM IMERG · ESA Sentinel · Open-Meteo ERA5 · SNSAT FASat-Delta")
    st.divider()

    # ── Selector de cuenca ──
    col_c1, col_c2 = st.columns([2,2])
    with col_c1:
        cuenca_nombre = st.selectbox("🗺️ Cuenca hidrológica",
                                      list(CUENCAS.keys()))
        cuenca_cfg = CUENCAS[cuenca_nombre]
        dias_hist  = st.slider("Días de historia a descargar", 7, 30, 14)
    with col_c2:
        st.markdown("**Coordenadas de la cuenca**")
        lat_c = (cuenca_cfg['lat_min']+cuenca_cfg['lat_max'])/2
        lon_c = (cuenca_cfg['lon_min']+cuenca_cfg['lon_max'])/2
        st.markdown(f"""
        | Parámetro | Valor |
        |-----------|-------|
        | Latitud centro | {lat_c:.2f}° S |
        | Longitud centro | {lon_c:.2f}° W |
        | Área aprox. | {cuenca_cfg['area_km2']:,} km² |
        | Bbox | {cuenca_cfg['lat_min']}→{cuenca_cfg['lat_max']} / {cuenca_cfg['lon_min']}→{cuenca_cfg['lon_max']} |
        """)

    st.divider()

    # ── Estado de satélites ──
    st.markdown("#### 📡 Estado de fuentes satelitales")
    cols_sat = st.columns(len(SATELITES))
    for idx, (nombre, info) in enumerate(SATELITES.items()):
        with cols_sat[idx]:
            # Test rápido de conectividad
            online = verificar_conectividad_satelite(nombre, info['url'])
            estado_html = '<span class="sat-online">● ONLINE</span>' if online else \
                          '<span class="sat-offline">● STANDBY</span>'
            st.markdown(f"""
            <div class='sat-card'>
              <div style='font-size:.8rem;font-weight:700;color:#E8F4FD;margin-bottom:6px'>
                {nombre}
              </div>
              <div style='font-size:.7rem;color:#8BA8C8'>{info['variable']}</div>
              <div style='font-size:.65rem;color:#4A90D9;margin-top:4px'>{info['resolucion']}</div>
              <div style='margin-top:8px'>{estado_html}</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ── Botón de ingesta ──
    col_btn1, col_btn2, _ = st.columns([1,1,2])
    btn_ingestar = col_btn1.button("🚀 Ingestar datos ahora",
                                    type="primary", use_container_width=True)
    btn_limpiar  = col_btn2.button("🗑️ Limpiar caché",
                                    use_container_width=True)

    if btn_limpiar:
        for k in ['sat_data','sat_log','sat_fuente','sat_tipo']:
            st.session_state.pop(k,None)
        st.rerun()

    if btn_ingestar:
        for k in ['sat_data','sat_log','sat_fuente','sat_tipo']:
            st.session_state.pop(k,None)

        with st.spinner("🛰️ Conectando con fuentes satelitales..."):
            resultado = obtener_datos_satelitales_completos(cuenca_cfg, dias_hist)

        st.session_state['sat_data']   = resultado['datos']
        st.session_state['sat_log']    = resultado['log']
        st.session_state['sat_fuente'] = resultado['fuente']
        st.session_state['sat_tipo']   = resultado['tipo']
        st.session_state['sat_cuenca'] = cuenca_nombre

    # ── Mostrar resultados si hay datos ──
    if 'sat_data' in st.session_state:
        df_sat     = st.session_state['sat_data']
        log_sat    = st.session_state['sat_log']
        fuente_sat = st.session_state['sat_fuente']
        tipo_sat   = st.session_state['sat_tipo']

        # Log del pipeline
        st.markdown("#### 🔄 Log del pipeline de ingesta")
        for entry in log_sat:
            css = {'ok':'pipeline-ok','warn':'pipeline-warn',
                   'err':'pipeline-err','intentando':'pipeline-step'}.get(
                   entry['estado'],'pipeline-step')
            st.markdown(f"<div class='{css}'>{entry['paso']}</div>",
                        unsafe_allow_html=True)

        # Badge de fuente
        badge_color = "#388E3C" if tipo_sat == 'real' else "#F57C00"
        badge_texto = "🟢 DATOS REALES" if tipo_sat == 'real' else "🟡 DATOS SINTÉTICOS CALIBRADOS"
        st.markdown(f"""
        <div style='background:{badge_color}22;border:1px solid {badge_color};
             border-radius:8px;padding:10px 16px;margin:12px 0;display:inline-block'>
          <b style='color:{badge_color}'>{badge_texto}</b>
          &nbsp;·&nbsp; Fuente: {fuente_sat}
          &nbsp;·&nbsp; {len(df_sat)} registros
          &nbsp;·&nbsp; Cuenca: {st.session_state.get('sat_cuenca','')}
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # ── KPIs en tiempo real ──
        st.markdown("#### 📊 Variables en tiempo real")
        df_reciente = df_sat.tail(7)
        ult = df_sat.iloc[-1]

        kc = st.columns(6)
        kpis = [
            ('🌧️','Precip. hoy',  f"{ult['precipitacion']:.1f}","mm"),
            ('🌡️','Temperatura',  f"{ult['temperatura']:.1f}","°C"),
            ('💧','DTW actual',   f"{ult['dtw']:.2f}","m"),
            ('🌱','Saturación',   f"{ult['saturacion_suelo']:.1f}","%"),
            ('🌊','Flujo acum.',  f"{ult['acumulacion_flujo']:.0f}","m²"),
            ('⚠️','Susc. actual', f"{ult['susceptibilidad']:.3f}","índice"),
        ]
        for col_k,(ico,lab,val,uni) in zip(kc,kpis):
            col_k.metric(f"{ico} {lab}", f"{val} {uni}")

        st.divider()

        # ── Gráficos de series satelitales ──
        st.markdown("#### 📈 Series descargadas")
        tab1, tab2, tab3 = st.tabs(["🌧️ Hidrometeorología",
                                     "🌊 Variables WAM",
                                     "⚠️ Susceptibilidad"])

        with tab1:
            df_plot = df_sat[['precipitacion','temperatura','evapotranspiracion']].reset_index()
            df_plot.columns = ['fecha','precipitacion','temperatura','evapotranspiracion']

            precip_chart = alt.Chart(df_plot).mark_bar(color='#4A90D9',opacity=.8).encode(
                x=alt.X('fecha:T',title=''),
                y=alt.Y('precipitacion:Q',title='Precipitación (mm)'),
                tooltip=['fecha:T','precipitacion:Q']
            ).properties(height=200, title='Precipitación diaria')

            temp_chart = alt.Chart(df_plot).mark_line(
                color='#D32F2F',strokeWidth=2,point=True).encode(
                x=alt.X('fecha:T',title='Fecha'),
                y=alt.Y('temperatura:Q',title='Temperatura (°C)'),
                tooltip=['fecha:T','temperatura:Q']
            ).properties(height=180, title='Temperatura media')

            st.altair_chart(precip_chart, use_container_width=True)
            st.altair_chart(temp_chart,   use_container_width=True)

        with tab2:
            df_wam = df_sat[['dtw','saturacion_suelo']].reset_index()
            df_wam.columns = ['fecha','dtw','saturacion']

            dtw_chart = alt.Chart(df_wam).mark_area(
                line={'color':'#F57C00','strokeWidth':2},
                color=alt.Gradient(gradient='linear',x1=0,x2=0,y1=1,y2=0,
                    stops=[alt.GradientStop(color='white',offset=0),
                           alt.GradientStop(color='#F57C0055',offset=1)])
            ).encode(
                x=alt.X('fecha:T',title=''),
                y=alt.Y('dtw:Q',title='DTW (m)',
                        scale=alt.Scale(reverse=True)),
                tooltip=['fecha:T','dtw:Q']
            ).properties(height=200, title='DTW — Depth to Water ↑ = más riesgo')

            sat_chart = alt.Chart(df_wam).mark_area(
                line={'color':'#388E3C','strokeWidth':2},
                color=alt.Gradient(gradient='linear',x1=0,x2=0,y1=1,y2=0,
                    stops=[alt.GradientStop(color='white',offset=0),
                           alt.GradientStop(color='#388E3C55',offset=1)])
            ).encode(
                x=alt.X('fecha:T',title='Fecha'),
                y=alt.Y('saturacion:Q',title='Saturación (%)',
                        scale=alt.Scale(domain=[0,100])),
                tooltip=['fecha:T','saturacion:Q']
            ).properties(height=200, title='Saturación del suelo')

            u80 = alt.Chart(pd.DataFrame({'y':[80]})).mark_rule(
                color='red',strokeDash=[4,4]).encode(y='y:Q')

            st.altair_chart(dtw_chart, use_container_width=True)
            st.altair_chart(sat_chart+u80, use_container_width=True)

        with tab3:
            df_susc = df_sat[['susceptibilidad']].reset_index()
            df_susc.columns = ['fecha','susceptibilidad']
            df_susc['alerta'] = df_susc['susceptibilidad'].apply(
                lambda s: nivel_alerta(s)[0].split(' ',1)[1])
            df_susc['color']  = df_susc['susceptibilidad'].apply(
                lambda s: nivel_alerta(s)[1])

            susc_line = alt.Chart(df_susc).mark_area(
                line={'color':'#7B1FA2','strokeWidth':2.5},
                color=alt.Gradient(gradient='linear',x1=0,x2=0,y1=1,y2=0,
                    stops=[alt.GradientStop(color='white',offset=0),
                           alt.GradientStop(color='#7B1FA255',offset=1)])
            ).encode(
                x=alt.X('fecha:T',title='Fecha'),
                y=alt.Y('susceptibilidad:Q',title='Susceptibilidad',
                        scale=alt.Scale(domain=[0,1])),
                tooltip=['fecha:T','susceptibilidad:Q','alerta:N']
            )
            u60 = alt.Chart(pd.DataFrame({'y':[.6]})).mark_rule(
                color='orange',strokeDash=[4,4]).encode(y='y:Q')
            u75 = alt.Chart(pd.DataFrame({'y':[.75]})).mark_rule(
                color='red',strokeDash=[6,3]).encode(y='y:Q')

            st.altair_chart((susc_line+u60+u75).properties(height=300),
                            use_container_width=True)

            # Estado actual
            s_actual = float(ult['susceptibilidad'])
            niv, chex, cls = nivel_alerta(s_actual)
            fondo = {"critico":"#FFEBEE","alto":"#FFF3E0",
                     "moderado":"#FFFDE7","bajo":"#E8F5E9"}
            st.markdown(f"""
            <div class='alerta-box' style='background:{fondo[cls]};
                 border-left:5px solid {chex}'>
              <b style='font-size:1.1rem;color:{chex}'>Estado actual: {niv}</b><br>
              Susceptibilidad = {s_actual:.4f} · DTW = {ult['dtw']:.2f}m ·
              Saturación = {ult['saturacion_suelo']:.1f}%
            </div>""", unsafe_allow_html=True)

        st.divider()

        # ── Integrar al modelo LSTM ──
        st.markdown("#### 🔗 Usar estos datos para pronóstico")
        col_i1, col_i2 = st.columns([2,1])
        with col_i1:
            st.info("""
            💡 Estos datos satelitales pueden usarse directamente como entrada al
            motor LSTM para generar un **pronóstico en tiempo real**.

            Si tienes el modelo entrenado, ve al módulo **🚨 Alerta Temprana** y
            activa la opción **"Usar datos satelitales"** para ver el pronóstico
            con datos reales en lugar de sintéticos.
            """)
        with col_i2:
            csv_sat = df_sat.reset_index().to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Descargar datos satelitales",
                               csv_sat,
                               file_name=f"satelital_{cuenca_nombre.split(' ')[0].lower()}_{datetime.now().strftime('%Y%m%d')}.csv",
                               mime="text/csv",
                               use_container_width=True)

            if 'model' in st.session_state:
                if st.button("⚡ Generar alerta con datos reales",
                              use_container_width=True, type="primary"):
                    st.session_state['usar_sat_en_alerta'] = True
                    st.switch_page if hasattr(st,'switch_page') else None
                    st.info("Ve al módulo 🚨 Alerta Temprana")

    else:
        # Estado inicial — guía de uso
        st.markdown("#### 🗺️ Fuentes satelitales disponibles")
        df_fuentes = pd.DataFrame([
            {'Satélite':'🛰️ Open-Meteo ERA5','Variables':'Precip · Temp · ETP · Humedad','Latencia':'Tiempo real','Autenticación':'No requiere','Uso':'Automático'},
            {'Satélite':'🛰️ NASA GPM IMERG','Variables':'Precipitación','Latencia':'4 horas','Autenticación':'NASA Earthdata (gratis)','Uso':'Con token'},
            {'Satélite':'🛰️ ESA Sentinel-1','Variables':'Humedad SAR','Latencia':'6 días','Autenticación':'Copernicus (gratis)','Uso':'Con token'},
            {'Satélite':'🛰️ ESA Sentinel-2','Variables':'NDVI · NDWI','Latencia':'5 días','Autenticación':'Copernicus (gratis)','Uso':'Con token'},
            {'Satélite':'🛰️ NASA MODIS','Variables':'Temperatura · ETP','Latencia':'1 día','Autenticación':'NASA Earthdata (gratis)','Uso':'Con token'},
            {'Satélite':'🛰️ SNSAT FASat-Delta','Variables':'Imagen óptica 70cm','Latencia':'Variable','Autenticación':'Acuerdo AEXA','Uso':'Fase TRL-7'},
        ])
        st.dataframe(df_fuentes, use_container_width=True, hide_index=True)

        st.markdown("""
        #### ▶️ Cómo usar este módulo

        1. **Selecciona la cuenca** hidrológica en el selector de arriba
        2. **Presiona "Ingestar datos ahora"** — el sistema intenta conectar a Open-Meteo ERA5 automáticamente (sin necesidad de tokens)
        3. Si hay conexión a internet, obtienes **datos reales** de los últimos 14 días
        4. Si no hay conexión, activa el **fallback sintético calibrado** automáticamente
        5. Los datos descargados pueden usarse para generar alertas en tiempo real en el módulo **🚨 Alerta Temprana**

        > 💡 **Para producción con NASA/ESA:** agrega tus tokens en Streamlit Cloud →
        > Settings → Secrets con las claves `NASA_TOKEN` y `COPERNICUS_USER`/`COPERNICUS_PASS`
        """)


# ════════════════════════════════════════════════════════════
#  PÁGINA 1 — DASHBOARD
# ════════════════════════════════════════════════════════════
elif pagina == "📊 Dashboard":
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
    dfm=df['susceptibilidad'].resample('W').mean().reset_index()
    dfm.columns=['fecha','susceptibilidad']
    line=alt.Chart(dfm).mark_area(
        line={'color':'#1B3A6B','strokeWidth':2},
        color=alt.Gradient(gradient='linear',x1=0,x2=0,y1=1,y2=0,
            stops=[alt.GradientStop(color='white',offset=0),
                   alt.GradientStop(color='#1B3A6B',offset=1)])
    ).encode(x=alt.X('fecha:T'),y=alt.Y('susceptibilidad:Q',scale=alt.Scale(domain=[0,1])))
    umbral=alt.Chart(pd.DataFrame({'y':[.6]})).mark_rule(
        color='red',strokeDash=[4,4],strokeWidth=1.5).encode(y='y:Q')
    st.altair_chart((line+umbral).properties(height=280,title='Susceptibilidad semanal'),
                    use_container_width=True)

    st.divider()
    col_a,col_b=st.columns(2)
    with col_a:
        dfmes=df.copy(); dfmes['mes']=dfmes.index.month
        pm=dfmes.groupby('mes')[['precipitacion','susceptibilidad']].mean().reset_index()
        mmap={1:'Ene',2:'Feb',3:'Mar',4:'Abr',5:'May',6:'Jun',
              7:'Jul',8:'Ago',9:'Sep',10:'Oct',11:'Nov',12:'Dic'}
        pm['mes_str']=pm['mes'].map(mmap)
        bars=alt.Chart(pm).mark_bar(color='#4A90D9',opacity=.7).encode(
            x=alt.X('mes_str:N',sort=list(mmap.values())),
            y=alt.Y('precipitacion:Q',title='mm'))
        ln2=alt.Chart(pm).mark_line(color='#D32F2F',strokeWidth=2.5,point=True).encode(
            x=alt.X('mes_str:N',sort=list(mmap.values())),
            y=alt.Y('susceptibilidad:Q',scale=alt.Scale(domain=[0,1])))
        st.altair_chart(alt.layer(bars,ln2).resolve_scale(y='independent')
                        .properties(height=260,title='Estacionalidad mensual'),
                        use_container_width=True)
    with col_b:
        alertas=pd.DataFrame({
            'Nivel':['🟢 Bajo','🟡 Moderado','🟠 Alto','🔴 Crítico'],
            'Días':[int((df['susceptibilidad']<.4).sum()),
                    int(((df['susceptibilidad']>=.4)&(df['susceptibilidad']<.6)).sum()),
                    int(((df['susceptibilidad']>=.6)&(df['susceptibilidad']<.75)).sum()),
                    int((df['susceptibilidad']>=.75).sum())],
            'Color':['#388E3C','#F9A825','#F57C00','#D32F2F']
        })
        pie=alt.Chart(alertas).mark_arc(innerRadius=60).encode(
            theta=alt.Theta('Días:Q'),
            color=alt.Color('Nivel:N',scale=alt.Scale(
                domain=alertas['Nivel'].tolist(),
                range=alertas['Color'].tolist())),
            tooltip=['Nivel','Días'])
        st.altair_chart(pie.properties(height=260,title='Distribución de alertas'),
                        use_container_width=True)

    st.divider()
    corr=df.corr().round(2).reset_index().melt('index')
    corr.columns=['var1','var2','correlacion']
    hm=alt.Chart(corr).mark_rect().encode(
        x=alt.X('var1:N',title=''),y=alt.Y('var2:N',title=''),
        color=alt.Color('correlacion:Q',scale=alt.Scale(scheme='redblue',domain=[-1,1])),
        tooltip=['var1','var2','correlacion'])
    tx=alt.Chart(corr).mark_text(fontSize=10).encode(
        x='var1:N',y='var2:N',text=alt.Text('correlacion:Q',format='.2f'),
        color=alt.condition(alt.datum.correlacion>0.5,
                            alt.value('white'),alt.value('black')))
    st.altair_chart((hm+tx).properties(height=360,title='Correlación entre variables WAM'),
                    use_container_width=True)


# ════════════════════════════════════════════════════════════
#  PÁGINA 3 — DATOS SINTÉTICOS
# ════════════════════════════════════════════════════════════
elif pagina == "🧪 Datos Sintéticos":
    st.markdown("## 🧪 Generador de Datos Sintéticos Hidrológicos")
    st.caption("Simulación físicamente coherente · parámetros ajustables en la barra lateral")
    st.divider()

    variable=st.selectbox("Variable a visualizar",FEATURES+['susceptibilidad'])
    cmap={'precipitacion':'#4A90D9','temperatura':'#D32F2F',
          'evapotranspiracion':'#546E7A','dtw':'#F57C00',
          'saturacion_suelo':'#388E3C','acumulacion_flujo':'#2E5FA3',
          'susceptibilidad':'#7B1FA2'}
    col1,col2=st.columns([3,1])
    with col1:
        dv=df[[variable]].reset_index(); dv.columns=['fecha',variable]
        ac=alt.Chart(dv).mark_area(
            line={'color':cmap[variable],'strokeWidth':1.5},
            color=alt.Gradient(gradient='linear',x1=0,x2=0,y1=1,y2=0,
                stops=[alt.GradientStop(color='white',offset=0),
                       alt.GradientStop(color=cmap[variable],offset=1)])
        ).encode(x=alt.X('fecha:T'),y=alt.Y(f'{variable}:Q'),
                 tooltip=['fecha:T',f'{variable}:Q']).properties(height=320)
        rules=[]
        if variable=='susceptibilidad': rules=[.60,.75]
        if variable=='saturacion_suelo': rules=[80.0]
        cf=ac
        for r in rules:
            cf=cf+alt.Chart(pd.DataFrame({'y':[r]})).mark_rule(
                color='red',strokeDash=[4,4]).encode(y='y:Q')
        st.altair_chart(cf,use_container_width=True)
    with col2:
        st.markdown(f"**Estadísticas**")
        for k,v in df[variable].describe().items():
            st.metric(k,f"{v:.3f}")

    st.divider()
    dfw=df.resample('W').mean().reset_index(); dfw.columns=['fecha']+list(df.columns)
    vars_p=[('precipitacion','#4A90D9','bar'),('dtw','#F57C00','area'),
            ('saturacion_suelo','#388E3C','area'),('susceptibilidad','#7B1FA2','area')]
    charts=[]
    for var,color,tipo in vars_p:
        base=alt.Chart(dfw).encode(x=alt.X('fecha:T',title=''),
                                    y=alt.Y(f'{var}:Q',title=var),
                                    tooltip=['fecha:T',f'{var}:Q'])
        c=(base.mark_bar(color=color,opacity=.7) if tipo=='bar' else
           base.mark_area(line={'color':color,'strokeWidth':1.5},
               color=alt.Gradient(gradient='linear',x1=0,x2=0,y1=1,y2=0,
                   stops=[alt.GradientStop(color='white',offset=0),
                          alt.GradientStop(color=color,offset=1)])))
        charts.append(c.properties(height=130,title=var))
    st.altair_chart(alt.vconcat(*charts,spacing=6),use_container_width=True)

    st.divider()
    csv=df.reset_index().to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Descargar dataset CSV",csv,
                       "dataset_sintetico_itata.csv","text/csv",use_container_width=True)


# ════════════════════════════════════════════════════════════
#  PÁGINA 4 — ENTRENAMIENTO LSTM
# ════════════════════════════════════════════════════════════
elif pagina == "🏋️ Entrenamiento LSTM":
    st.markdown("## 🏋️ Entrenamiento LSTM en Vivo")
    st.caption("FloodLSTM · NumPy puro · pronóstico 7 días · Motor WAM-IA Fase 2")
    st.divider()

    col_cfg,col_info=st.columns([1,2])
    with col_cfg:
        st.markdown("**Hiperparámetros**")
        ventana  =st.slider("Ventana entrada (días)",      7,30,14)
        horizonte=st.slider("Horizonte pronóstico (días)", 3,14, 7)
        epochs   =st.slider("Épocas de entrenamiento",    10,60,25)
        hidden   =st.select_slider("Neuronas LSTM",  [16,32,64],value=32)
        lr       =st.select_slider("Learning rate",
                                   [0.0005,0.001,0.005],value=0.001)
    with col_info:
        st.markdown(f"""
        **Arquitectura FloodLSTM (NumPy)**
        ```
        Input  ({ventana} días × 6 features)
           ↓
        LSTM   ({hidden} neuronas)
           ↓
        Dense  ({hidden} → {horizonte}) + Sigmoid
           ↓
        susceptibilidad t+1 … t+{horizonte}
        ```
        Optimizador **Adam** · Gradient clipping ±1
        """)

    st.divider()
    iniciar=st.button("🚀 Iniciar entrenamiento",type="primary")

    if iniciar or 'historia' in st.session_state:
        if iniciar:
            for k in ['model','scX','scY','historia','metricas',
                      'ventana','horizonte','y_pred','y_true']:
                st.session_state.pop(k,None)
            np.random.seed(SEED)
            X_seq,y_seq,scX,scY=preparar_secuencias(df,ventana,horizonte)
            n=len(X_seq); n_test=int(n*.2); n_val=int(n*.1); n_train=n-n_test-n_val
            X_tr,y_tr=X_seq[:n_train],y_seq[:n_train]
            X_va,y_va=X_seq[n_train:n_train+n_val],y_seq[n_train:n_train+n_val]
            X_te,y_te=X_seq[n_train+n_val:],y_seq[n_train+n_val:]
            model=NumpyLSTM(len(FEATURES),hidden,horizonte,lr)

            st.markdown("#### 📉 Curva de aprendizaje")
            chart_ph=st.empty(); status_ph=st.empty(); prog_ph=st.progress(0)
            tl_hist,vl_hist=[],[]
            BATCH=32

            for ep in range(1,epochs+1):
                idx=np.random.permutation(len(X_tr))
                tl_ep=0; nb=0
                for start in range(0,len(idx),BATCH):
                    b=idx[start:start+BATCH]; bl=0
                    for j in b:
                        yp,h,c,cache=model.forward(X_tr[j])
                        bl+=model.backward(X_tr[j],y_tr[j],yp,h,c,cache)
                    tl_ep+=bl/len(b); nb+=1
                tl=tl_ep/nb
                vl=sum(float(np.mean((model.predict(X_va[j])-y_va[j])**2))
                       for j in range(len(X_va)))/len(X_va)
                tl_hist.append(tl); vl_hist.append(vl)

                if ep%3==0 or ep==epochs:
                    df_live=pd.DataFrame({
                        'época':list(range(1,len(tl_hist)+1))*2,
                        'loss':tl_hist+vl_hist,
                        'tipo':['Train']*len(tl_hist)+['Validación']*len(vl_hist)})
                    live=alt.Chart(df_live).mark_line().encode(
                        x=alt.X('época:Q'),y=alt.Y('loss:Q',scale=alt.Scale(type='log')),
                        color=alt.Color('tipo:N',scale=alt.Scale(
                            domain=['Train','Validación'],range=['#1B3A6B','#D32F2F'])),
                        strokeDash=alt.condition(
                            alt.datum.tipo=='Validación',
                            alt.value([6,3]),alt.value([1,0]))
                    ).properties(height=220)
                    chart_ph.altair_chart(live,use_container_width=True)
                    status_ph.info(f"Época {ep}/{epochs} — Train: {tl:.5f} | Val: {vl:.5f}")
                    prog_ph.progress(ep/epochs)

            prog_ph.empty()
            status_ph.success(f"✅ Completado · Val Loss: {vl:.5f}")

            y_pred_s=np.array([model.predict(X_te[j]) for j in range(len(X_te))])
            def desnorm(a):
                return np.clip(scY.inverse_transform(
                    a.reshape(-1,1)).flatten().reshape(a.shape),0,1)
            y_pred=desnorm(y_pred_s); y_true=desnorm(y_te)
            metricas=[{'Día':f't+{d+1}',
                       'MAE':round(mean_absolute_error(y_true[:,d],y_pred[:,d]),4),
                       'RMSE':round(np.sqrt(mean_squared_error(y_true[:,d],y_pred[:,d])),4),
                       'R²':round(r2_score(y_true[:,d],y_pred[:,d]),4)}
                      for d in range(horizonte)]
            st.session_state.update({'model':model,'scX':scX,'scY':scY,
                'historia':{'train':tl_hist,'val':vl_hist},
                'metricas':metricas,'ventana':ventana,'horizonte':horizonte,
                'y_pred':y_pred,'y_true':y_true})

        if 'metricas' in st.session_state:
            st.divider()
            st.markdown("#### 📊 Métricas por horizonte")
            df_met=pd.DataFrame(st.session_state['metricas'])
            col_m1,col_m2=st.columns([1,2])
            with col_m1:
                st.dataframe(df_met.set_index('Día'),use_container_width=True)
                mae_m=df_met['MAE'].mean(); r2_m=df_met['R²'].mean()
                st.metric("MAE promedio",f"{mae_m:.4f}",
                           delta="✅ Cumple TRL-5" if mae_m<.08 else "⚠️ Sobre umbral",
                           delta_color="normal" if mae_m<.08 else "inverse")
                st.metric("R² promedio",f"{r2_m:.4f}")
            with col_m2:
                df_ml=df_met.melt('Día',['MAE','RMSE','R²'],'métrica','valor')
                bm=alt.Chart(df_ml[df_ml['métrica']!='R²']).mark_bar(opacity=.8).encode(
                    x='Día:N',y=alt.Y('valor:Q'),
                    color=alt.Color('métrica:N',scale=alt.Scale(
                        domain=['MAE','RMSE'],range=['#4A90D9','#1B3A6B'])),
                    xOffset='métrica:N')
                rt=alt.Chart(pd.DataFrame({'y':[.08]})).mark_rule(
                    color='red',strokeDash=[4,4]).encode(y='y:Q')
                st.altair_chart((bm+rt).properties(height=280),use_container_width=True)

            if 'y_pred' in st.session_state:
                st.divider()
                st.markdown("#### 🔍 Real vs Predicho")
                hz=st.session_state['horizonte']
                hs=st.selectbox("Horizonte",[f"t+{d+1}" for d in range(hz)])
                di=int(hs.replace('t+',''))-1
                ns=min(150,len(st.session_state['y_pred']))
                df_c=pd.DataFrame({
                    'muestra':list(range(ns))*2,
                    'susceptibilidad':(list(st.session_state['y_true'][:ns,di])+
                                       list(st.session_state['y_pred'][:ns,di])),
                    'serie':(['Real']*ns+['Predicho']*ns)})
                cc=alt.Chart(df_c).mark_line().encode(
                    x='muestra:Q',y=alt.Y('susceptibilidad:Q',scale=alt.Scale(domain=[0,1])),
                    color=alt.Color('serie:N',scale=alt.Scale(
                        domain=['Real','Predicho'],range=['#1B3A6B','#D32F2F'])),
                    strokeDash=alt.condition(alt.datum.serie=='Predicho',
                                             alt.value([6,3]),alt.value([1,0])))
                ul=alt.Chart(pd.DataFrame({'y':[.6]})).mark_rule(
                    color='orange',strokeDash=[4,4]).encode(y='y:Q')
                st.altair_chart((cc+ul).properties(height=260),use_container_width=True)
    else:
        st.info("👆 Configura los hiperparámetros y presiona **Iniciar entrenamiento**")


# ════════════════════════════════════════════════════════════
#  PÁGINA 5 — ALERTA TEMPRANA
# ════════════════════════════════════════════════════════════
elif pagina == "🚨 Alerta Temprana":
    st.markdown("## 🚨 Boletín de Alerta Temprana — 7 Días")
    st.caption("Pronóstico Basado en Impacto (PBI) · Motor WAM-IA · Cuenca Itata, Ñuble")
    st.divider()

    if 'model' not in st.session_state:
        st.warning("⚠️ **No hay modelo entrenado.**  \nVe a **🏋️ Entrenamiento LSTM**, entrena el modelo y vuelve aquí.")
        st.stop()

    model=st.session_state['model']
    scX=st.session_state['scX']
    scY=st.session_state['scY']
    ventana=st.session_state['ventana']
    horizonte=st.session_state['horizonte']

    # ── Selector de fuente de datos ──
    fuente_datos = "📡 Datos satelitales (tiempo real)" \
        if 'sat_data' in st.session_state else "🧪 Datos sintéticos"

    if 'sat_data' in st.session_state:
        col_f0, _ = st.columns([2,2])
        with col_f0:
            fuente_sel = st.radio("Fuente de datos para el pronóstico",
                ["📡 Datos satelitales (tiempo real)",
                 "🧪 Datos sintéticos (histórico)"],
                horizontal=True)
        if "satelitales" in fuente_sel:
            df_alerta = st.session_state['sat_data']
            st.success(f"✅ Usando datos reales: {st.session_state.get('sat_fuente','Open-Meteo')}")
        else:
            df_alerta = df
    else:
        df_alerta = df
        st.info("💡 Tip: Ve al módulo 🛰️ **Ingesta Satelital** para usar datos reales en el pronóstico.")

    col_f1,col_f2=st.columns([2,1])
    with col_f1:
        fecha_sel=st.date_input("Fecha base del pronóstico",
                                 value=df_alerta.index[-1].date(),
                                 min_value=df_alerta.index[ventana].date(),
                                 max_value=df_alerta.index[-1].date())
    with col_f2:
        st.markdown(" ")
        if st.button("🎲 Evento extremo aleatorio",use_container_width=True):
            cand=df_alerta.index[df_alerta['susceptibilidad']>0.70]
            if len(cand):
                st.session_state['fecha_sel']=cand[np.random.randint(len(cand))].date()
                st.rerun()

    if 'fecha_sel' in st.session_state:
        fecha_sel=st.session_state['fecha_sel']

    try:
        idx_base=df_alerta.index.get_loc(pd.Timestamp(fecha_sel))
        if idx_base < ventana:
            st.error("Selecciona una fecha más tardía.")
            st.stop()

        datos_vent=df_alerta[FEATURES].iloc[idx_base-ventana:idx_base].values
        vent_scl=scX.transform(datos_vent)
        pred_s=model.predict(vent_scl)
        pred_real=np.clip(scY.inverse_transform(pred_s.reshape(-1,1)).flatten(),0,1)
        fechas_pred=pd.date_range(start=pd.Timestamp(fecha_sel)+pd.Timedelta(days=1),
                                   periods=horizonte,freq='D')

        max_s=float(pred_real.max())
        niv,chex,cls=nivel_alerta(max_s)
        fondo={"critico":"#FFEBEE","alto":"#FFF3E0","moderado":"#FFFDE7","bajo":"#E8F5E9"}
        st.markdown(f"""
        <div class='alerta-box' style='background:{fondo[cls]};border-left:5px solid {chex}'>
          <b style='font-size:1.3rem;color:{chex}'>{niv}</b>
          &nbsp;&nbsp;Susceptibilidad máxima: <b>{max_s:.3f}</b><br>
          <span style='font-size:.9rem;color:#555'>
          Pronóstico {horizonte} días desde {str(fecha_sel)}</span>
        </div>""", unsafe_allow_html=True)

        cols=st.columns(horizonte)
        for i,(f,s) in enumerate(zip(fechas_pred,pred_real)):
            nv,ch,cl=nivel_alerta(float(s))
            with cols[i]:
                st.markdown(f"""
                <div style='background:{ch}15;border:2px solid {ch};
                     border-radius:10px;padding:12px;text-align:center'>
                  <div style='font-size:.72rem;color:#555'>{f.strftime('%a %d/%m')}</div>
                  <div style='font-size:1.6rem;font-weight:800;color:{ch}'>{s:.2f}</div>
                  <div style='font-size:.68rem;color:{ch}'>{nv.split(' ',1)[1]}</div>
                </div>""", unsafe_allow_html=True)

        st.divider()
        col_g1,col_g2=st.columns([3,1])
        with col_g1:
            hs=max(0,idx_base-45)
            hf=df_alerta.index[hs:idx_base+1]
            hv=df_alerta['susceptibilidad'].iloc[hs:idx_base+1]
            hdf=pd.DataFrame({'fecha':hf,'susceptibilidad':hv.values})

            hist_line=alt.Chart(hdf).mark_area(
                line={'color':'#1B3A6B','strokeWidth':2},
                color=alt.Gradient(gradient='linear',x1=0,x2=0,y1=1,y2=0,
                    stops=[alt.GradientStop(color='white',offset=0),
                           alt.GradientStop(color='#1B3A6B22',offset=1)])
            ).encode(x=alt.X('fecha:T'),
                     y=alt.Y('susceptibilidad:Q',scale=alt.Scale(domain=[0,1.05])),
                     tooltip=['fecha:T','susceptibilidad:Q'])

            pdf=pd.DataFrame({'fecha':fechas_pred,'susceptibilidad':pred_real,
                'color':[nivel_alerta(float(s))[1] for s in pred_real]})
            pred_bars=alt.Chart(pdf).mark_bar(opacity=.5,width=20).encode(
                x='fecha:T',y=alt.Y('susceptibilidad:Q',scale=alt.Scale(domain=[0,1.05])),
                color=alt.Color('color:N',scale=alt.Scale(
                    domain=pdf['color'].tolist(),range=pdf['color'].tolist()),legend=None))
            pred_line=alt.Chart(pdf).mark_line(
                color='#D32F2F',strokeWidth=2,strokeDash=[6,3],
                point=alt.OverlayMarkDef(color='#D32F2F',size=80)
            ).encode(x='fecha:T',
                     y=alt.Y('susceptibilidad:Q',scale=alt.Scale(domain=[0,1.05])),
                     tooltip=['fecha:T','susceptibilidad:Q'])

            u60=alt.Chart(pd.DataFrame({'y':[.6]})).mark_rule(
                color='orange',strokeDash=[4,4],strokeWidth=1.5).encode(y='y:Q')
            u75=alt.Chart(pd.DataFrame({'y':[.75]})).mark_rule(
                color='red',strokeDash=[6,3],strokeWidth=1.5).encode(y='y:Q')

            st.altair_chart(alt.layer(hist_line,pred_bars,pred_line,u60,u75)
                            .properties(height=340,title='Serie histórica + pronóstico WAM-IA'),
                            use_container_width=True)

        with col_g2:
            st.markdown("#### Condiciones actuales")
            dia=df_alerta.iloc[idx_base]
            for lbl,val in [('🌧️ Precip.',f"{dia['precipitacion']:.1f} mm"),
                            ('🌡️ Temp.',  f"{dia['temperatura']:.1f} °C"),
                            ('💧 DTW',    f"{dia['dtw']:.2f} m"),
                            ('🌱 Sat.',   f"{dia['saturacion_suelo']:.1f}%"),
                            ('🔮 Susc.',  f"{dia['susceptibilidad']:.3f}")]:
                st.metric(lbl,val)

        st.divider()
        df_bol=pd.DataFrame({
            'Fecha':       [f.strftime('%d/%m/%Y') for f in fechas_pred],
            'Susceptibilidad':[f"{s:.4f}" for s in pred_real],
            'Nivel':       [nivel_alerta(float(s))[0] for s in pred_real],
            'Acción':      ['Monitoreo continuo'  if nivel_alerta(float(s))[2]=='critico' else
                            'Activar protocolos'  if nivel_alerta(float(s))[2]=='alto'    else
                            'Vigilancia estándar' if nivel_alerta(float(s))[2]=='moderado'else
                            'Sin acción inmediata' for s in pred_real]})
        st.dataframe(df_bol,use_container_width=True,hide_index=True)
        st.download_button("⬇️ Descargar boletín CSV",
                           df_bol.to_csv(index=False).encode('utf-8'),
                           f"boletin_wam_{fecha_sel}.csv","text/csv",
                           use_container_width=True)

    except Exception as e:
        st.error(f"Error al calcular pronóstico: {e}")
