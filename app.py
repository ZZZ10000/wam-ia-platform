# ============================================================
#  🌊 PLATAFORMA WAM-IA — Motor Híbrido de Inteligencia Hídrica
#  MAKEY × Integra Sur Norte × UNB
#  v6.0 UNIFIED — Dashboard + Satélite + Mapa 3D + LSTM + Alertas
#  Open-Meteo ERA5 · pydeck · NumPy LSTM · Altair · CORFO TRL-5
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import requests
import json
import pydeck as pdk
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(
    page_title="WAM-IA | Plataforma Hídrica Unificada",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  [data-testid="stSidebar"] { background: #0D1B2A; }
  [data-testid="stSidebar"] * { color: #E8F4FD !important; }

  .alerta-box  { padding:14px 18px; border-radius:8px; margin-bottom:12px; }
  .info-card   { background:#F0F4FF; border-left:4px solid #1B3A6B;
                 padding:10px 14px; border-radius:0 8px 8px 0; margin:4px 0; font-size:.86rem; }
  .pipeline-ok   { border-left:4px solid #388E3C; background:#F1F8E9;
                   padding:9px 14px; border-radius:0 6px 6px 0; margin:3px 0; font-size:.84rem; }
  .pipeline-warn { border-left:4px solid #F57C00; background:#FFF3E0;
                   padding:9px 14px; border-radius:0 6px 6px 0; margin:3px 0; font-size:.84rem; }
  .pipeline-step { border-left:4px solid #1B3A6B; background:#F0F4FF;
                   padding:9px 14px; border-radius:0 6px 6px 0; margin:3px 0; font-size:.84rem; }
  .sat-card    { background:#0D1B2A; border:1px solid #2E5FA3; border-radius:10px;
                 padding:14px; text-align:center; }
  .kpi-section { background:#F8FAFF; border-radius:10px; padding:12px; margin:4px 0; }

  div[data-testid="stMetricValue"] { color:#1B3A6B !important; }
  .sat-online  { color:#4CAF50; font-weight:700; }
  .sat-offline { color:#FF9800; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  CONSTANTES GLOBALES
# ════════════════════════════════════════════════════════════
FEATURES = ['precipitacion','temperatura','evapotranspiracion',
            'dtw','saturacion_suelo','acumulacion_flujo']
TARGET = 'susceptibilidad'
SEED   = 42

CUENCAS = {
    '🌊 Itata — Ñuble':        {'lat':-36.6,'lon':-71.8,'lat_min':-37.2,'lat_max':-36.0,
                                 'lon_min':-72.5,'lon_max':-71.0,'zoom':8,'area_km2':11315,
                                 'region':'Ñuble','desc':'Cuenca piloto WAM-IA · validación 2017'},
    '🌊 Biobío — Concepción':  {'lat':-37.8,'lon':-72.5,'lat_min':-38.5,'lat_max':-37.0,
                                 'lon_min':-73.5,'lon_max':-71.5,'zoom':7,'area_km2':23695,
                                 'region':'Biobío','desc':'Cuenca más grande de Chile central'},
    '🌊 Maule — Talca':        {'lat':-35.8,'lon':-71.5,'lat_min':-36.5,'lat_max':-35.0,
                                 'lon_min':-72.0,'lon_max':-70.5,'zoom':8,'area_km2':20280,
                                 'region':'Maule','desc':'Cuenca vitivinícola · alta demanda hídrica'},
    '🌊 Copiapó — Atacama':    {'lat':-27.8,'lon':-69.8,'lat_min':-28.5,'lat_max':-27.0,
                                 'lon_min':-70.5,'lon_max':-69.0,'zoom':8,'area_km2':18705,
                                 'region':'Atacama','desc':'Validación inundación Copiapó 2015'},
    '🌊 Aconcagua — Valparaíso':{'lat':-32.8,'lon':-70.5,'lat_min':-33.2,'lat_max':-32.3,
                                  'lon_min':-71.2,'lon_max':-70.0,'zoom':9,'area_km2':7340,
                                  'region':'Valparaíso','desc':'Cuenca urbana · interfase Andes-Costa'},
}

C = {'azul':'#1B3A6B','azul2':'#2E5FA3','celeste':'#4A90D9',
     'verde':'#388E3C','naranja':'#F57C00','rojo':'#D32F2F',
     'amarillo':'#F9A825','morado':'#7B1FA2','gris':'#546E7A'}


# ════════════════════════════════════════════════════════════
#  UTILIDADES COMPARTIDAS
# ════════════════════════════════════════════════════════════
def nivel_alerta(s):
    if s>=0.75: return "🔴 CRÍTICO","#D32F2F","critico"
    if s>=0.60: return "🟠 ALTO","#F57C00","alto"
    if s>=0.40: return "🟡 MODERADO","#F9A825","moderado"
    return             "🟢 BAJO","#388E3C","bajo"

def susc_a_rgb(s):
    if s>=0.75: return [211,47,47,210]
    if s>=0.60: return [245,124,0,200]
    if s>=0.40: return [249,168,37,180]
    if s>=0.20: return [76,175,80,160]
    return             [33,150,243,140]


# ════════════════════════════════════════════════════════════
#  GENERADOR DE DATOS SINTÉTICOS
# ════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def generar_datos(n_dias=1825, seed=42, variabilidad=1.0, frec_ext=0.03):
    np.random.seed(seed)
    fechas = pd.date_range('2019-01-01', periods=n_dias, freq='D')
    t = np.arange(n_dias)
    estac  = 0.5*(1-np.cos(2*np.pi*(t%365-160)/365))
    prob   = np.clip(0.05+0.45*estac, 0, 0.7)
    precip = np.clip(np.random.binomial(1,prob)*np.random.exponential(8*variabilidad,n_dias)
                     *(1+np.random.binomial(1,frec_ext,n_dias)*np.random.uniform(5,15,n_dias)),0,180)
    temp   = 14+8*np.cos(2*np.pi*(t%365-15)/365)+np.random.normal(0,2,n_dias)
    etp    = np.clip(0.15*temp+np.random.normal(0,.5,n_dias), 0, None)
    r7     = pd.Series(precip).rolling(7,  min_periods=1).sum().values
    r30    = pd.Series(precip).rolling(30, min_periods=1).sum().values
    dtw    = np.zeros(n_dias); dtw[0]=3.5
    for i in range(1,n_dias):
        dtw[i]=max(0.05,min(8.0, dtw[i-1]-0.008*precip[i]-0.003*r7[i]
                               +0.04*etp[i]+0.005+np.random.normal(0,.05)))
    sat  = np.clip(30+2*r7-0.8*dtw*5+np.random.normal(0,3,n_dias), 5, 100)
    afl  = np.clip(1500+80*r7+25*r30+np.random.normal(0,100,n_dias), 200, 15000)
    susc = np.clip(
        pd.Series(np.tanh(precip/30)*.35+np.exp(-dtw/2)*.30
                  +(sat/100)**2*.25+np.tanh(afl/5000)*.10
                  ).rolling(3,min_periods=1).mean().values
        +np.random.normal(0,.02,n_dias), 0, 1)
    return pd.DataFrame({
        'precipitacion':np.round(precip,2),'temperatura':np.round(temp,2),
        'evapotranspiracion':np.round(etp,3),'dtw':np.round(dtw,3),
        'saturacion_suelo':np.round(sat,2),'acumulacion_flujo':np.round(afl,1),
        'susceptibilidad':np.round(susc,4)}, index=fechas)


# ════════════════════════════════════════════════════════════
#  LSTM EN NUMPY PURO
# ════════════════════════════════════════════════════════════
class NumpyLSTM:
    def __init__(self, input_size, hidden_size, output_size, lr=0.001):
        self.hs=hidden_size; self.lr=lr; s=0.1; n=input_size+hidden_size
        for nm in ['Wf','Wi','Wc','Wo']: setattr(self,nm,np.random.randn(hidden_size,n)*s)
        for nm in ['bf','bi','bc','bo']: setattr(self,nm,np.zeros((hidden_size,1)))
        self.Wy=np.random.randn(output_size,hidden_size)*s
        self.by=np.zeros((output_size,1)); self.t=0
        keys=['Wf','Wi','Wc','Wo','bf','bi','bc','bo','Wy','by']
        self.m={k:np.zeros_like(getattr(self,k)) for k in keys}
        self.v={k:np.zeros_like(getattr(self,k)) for k in keys}

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
            c=f*c+i*g; h=o*self.tanh(c); cache.append((x,xh,f,i,g,o,c,h))
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
            tc=self.tanh(ct); do=dh*tc; dc+=dh*o*(1-tc**2)
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


def preparar_secuencias(df, ventana, horizonte):
    scX=MinMaxScaler(); scY=MinMaxScaler()
    Xs=scX.fit_transform(df[FEATURES].values)
    ys=scY.fit_transform(df[TARGET].values.reshape(-1,1)).flatten()
    sX,sY=[],[]
    for i in range(len(Xs)-ventana-horizonte+1):
        sX.append(Xs[i:i+ventana]); sY.append(ys[i+ventana:i+ventana+horizonte])
    return np.array(sX),np.array(sY),scX,scY


# ════════════════════════════════════════════════════════════
#  HELPER — datos sintéticos para ingesta (cuando falla API)
# ════════════════════════════════════════════════════════════
def generar_datos_sat_sint(lat, lon, n):
    np.random.seed(int(abs(lat*100+lon*100))%9999)
    fechas=pd.date_range(end=datetime.now().date(), periods=n, freq='D')
    t=np.arange(n)
    estac=0.5*(1-np.cos(2*np.pi*(t%365-160)/365))
    precip=np.clip(np.random.binomial(1,np.clip(.05+.45*estac,0,.7))
                   *np.random.exponential(8,n), 0, 120)
    temp=14+8*np.cos(2*np.pi*(t%365-15)/365)+np.random.normal(0,2,n)
    etp=np.clip(.15*temp+np.random.normal(0,.5,n), 0, None)
    r7=pd.Series(precip).rolling(7,min_periods=1).sum().values
    r30=pd.Series(precip).rolling(30,min_periods=1).sum().values
    dtw=np.zeros(n); dtw[0]=3.5
    for i in range(1,n):
        dtw[i]=max(.05,min(8., dtw[i-1]-.008*precip[i]-.003*r7[i]
                              +.04*etp[i]+.005+np.random.normal(0,.05)))
    sat=np.clip(30+2*r7-.8*dtw*5+np.random.normal(0,3,n), 5, 100)
    afl=np.clip(1500+80*r7+25*r30+np.random.normal(0,100,n), 200, 15000)
    susc=np.clip(np.tanh(precip/30)*.35+np.exp(-dtw/2)*.30
                 +(sat/100)**2*.25+np.tanh(afl/5000)*.10, 0, 1)
    return pd.DataFrame({
        'precipitacion':np.round(precip,2),'temperatura':np.round(temp,2),
        'evapotranspiracion':np.round(etp,3),'dtw':np.round(dtw,3),
        'saturacion_suelo':np.round(sat,2),'acumulacion_flujo':np.round(afl,1),
        'susceptibilidad':np.round(susc,4)}, index=fechas)


# ════════════════════════════════════════════════════════════
#  MÓDULO SATELITAL — DESCARGA Y PROCESAMIENTO
# ════════════════════════════════════════════════════════════
def verificar_api(url):
    try:
        r=requests.get(url, timeout=5); return r.status_code < 500
    except: return False


def descargar_open_meteo_serie(lat, lon, dias=14):
    """Descarga serie temporal de un punto — para Ingesta."""
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast", params={
            'latitude': lat, 'longitude': lon,
            'daily': ['precipitation_sum','temperature_2m_mean',
                      'et0_fao_evapotranspiration','soil_moisture_0_to_7cm'],
            'past_days': dias, 'forecast_days': 7,
            'timezone': 'America/Santiago'
        }, timeout=10)
        if r.status_code == 200:
            d = r.json()['daily']
            n = len(d['time'])
            precip = np.nan_to_num(np.array(d['precipitation_sum'],    dtype=float), nan=0.0)
            temp   = np.nan_to_num(np.array(d['temperature_2m_mean'],  dtype=float), nan=15.0)
            etp    = np.nan_to_num(np.array(d['et0_fao_evapotranspiration'],dtype=float), nan=2.0)
            hum    = np.nan_to_num(np.array(d.get('soil_moisture_0_to_7cm',[0.3]*n),dtype=float), nan=0.3)
            df = pd.DataFrame({'fecha':pd.to_datetime(d['time']),
                               'precipitacion':precip,'temperatura':temp,
                               'evapotranspiracion':etp,'humedad_suelo':hum}
                              ).set_index('fecha')
            return {'exito':True, 'datos':df, 'fuente':'Open-Meteo ERA5 (Real)'}
        return {'exito':False,'error':f'HTTP {r.status_code}'}
    except Exception as e:
        return {'exito':False,'error':str(e)}


def enriquecer_wam(df_meteo, area_km2=11315):
    """Calcula variables WAM (DTW, saturación, susceptibilidad) desde datos meteorológicos."""
    n=len(df_meteo)
    precip=df_meteo['precipitacion'].values
    etp=df_meteo.get('evapotranspiracion', pd.Series(np.ones(n)*2.0)).values
    r7 =pd.Series(precip).rolling(7,  min_periods=1).sum().values
    r30=pd.Series(precip).rolling(30, min_periods=1).sum().values
    dtw=np.zeros(n); dtw[0]=3.5
    for i in range(1,n):
        dtw[i]=max(0.05,min(8.0, dtw[i-1]-0.008*precip[i]-0.003*r7[i]
                               +0.04*etp[i]+0.005))
    sat = (np.clip(df_meteo['humedad_suelo'].values*250, 5, 100)
           if 'humedad_suelo' in df_meteo.columns
           else np.clip(30+2*r7-0.8*dtw*5, 5, 100))
    afl = np.clip(area_km2*0.13+80*r7+25*r30, 200, 15000)
    susc= np.clip(np.tanh(precip/30)*.35+np.exp(-dtw/2)*.30
                  +(sat/100)**2*.25+np.tanh(afl/5000)*.10, 0, 1)
    df_out = df_meteo.copy()
    df_out['dtw']              = np.round(dtw,3)
    df_out['saturacion_suelo'] = np.round(sat,2)
    df_out['acumulacion_flujo']= np.round(afl,1)
    df_out['susceptibilidad']  = np.round(susc,4)
    for col in FEATURES:
        if col not in df_out.columns: df_out[col]=0.0
    return df_out[FEATURES+['susceptibilidad']]


def punto_sintetico(lat, lon, area_km2=11315):
    np.random.seed(int(abs(lat*100+lon*100))%9999)
    precip=max(0,np.random.exponential(5))
    dtw=max(0.1, 3.5-precip*0.05+np.random.normal(0,.3))
    sat=min(100,max(5, 30+precip*2-dtw*5))
    susc=np.clip(np.tanh(precip/30)*.35+np.exp(-dtw/2)*.30+(sat/100)**2*.25+.05, 0,1)
    return {'lat':lat,'lon':lon,
            'precipitacion':round(precip,2),'temperatura':round(14+np.random.normal(0,3),1),
            'evapotranspiracion':round(max(0,2+np.random.normal(0,.5)),2),
            'dtw':round(dtw,3),'saturacion_suelo':round(sat,1),
            'acumulacion_flujo':round(1500+precip*80,0),
            'susceptibilidad':round(float(susc),4),
            'precip_7d':round(precip*3,1),'prob_lluvia_max':round(min(100,precip*5),0),
            'fuente':'Sintético calibrado','timestamp':datetime.now().strftime('%Y-%m-%d %H:%M')}


def descargar_grilla(cuenca, resolucion=0.35, dias=7):
    """Descarga grilla de puntos sobre la cuenca para el mapa."""
    lats=np.arange(cuenca['lat_min'],cuenca['lat_max'],resolucion)
    lons=np.arange(cuenca['lon_min'],cuenca['lon_max'],resolucion)
    puntos=[(la,lo) for la in lats for lo in lons]
    resultados=[]; barra=st.progress(0); status=st.empty()
    for idx,(lat,lon) in enumerate(puntos):
        barra.progress((idx+1)/len(puntos))
        status.caption(f"📡 Descargando punto {idx+1}/{len(puntos)} — ({lat:.2f}°, {lon:.2f}°)")
        try:
            r=requests.get("https://api.open-meteo.com/v1/forecast", params={
                'latitude':lat,'longitude':lon,
                'daily':['precipitation_sum','temperature_2m_mean',
                         'et0_fao_evapotranspiration','soil_moisture_0_to_7cm',
                         'precipitation_probability_max'],
                'past_days':dias,'forecast_days':7,'timezone':'America/Santiago'
            }, timeout=8)
            if r.status_code==200:
                d=r.json()['daily']; n=len(d['time'])
                precip=np.nan_to_num(np.array(d['precipitation_sum'],dtype=float))
                temp  =np.nan_to_num(np.array(d['temperature_2m_mean'],dtype=float),nan=15.0)
                etp   =np.nan_to_num(np.array(d['et0_fao_evapotranspiration'],dtype=float),nan=2.0)
                hum   =np.nan_to_num(np.array(d.get('soil_moisture_0_to_7cm',[0.3]*n),dtype=float),nan=0.3)
                r7    =np.convolve(precip,np.ones(7)/7,mode='same')
                dtw   =np.clip(3.5-r7*.05, .1, 8.0)
                sat   =np.clip(hum*250, 5, 100)
                afl   =np.clip(cuenca['area_km2']*.13+80*r7, 200, 15000)
                susc  =np.clip(np.tanh(precip/30)*.35+np.exp(-dtw/2)*.30
                               +(sat/100)**2*.25+np.tanh(afl/5000)*.10, 0,1)
                ih=min(dias,n-1)
                resultados.append({
                    'lat':lat,'lon':lon,
                    'precipitacion':float(precip[ih]),'temperatura':float(temp[ih]),
                    'evapotranspiracion':float(etp[ih]),'dtw':float(dtw[ih]),
                    'saturacion_suelo':float(sat[ih]),'acumulacion_flujo':float(afl[ih]),
                    'susceptibilidad':float(susc[ih]),
                    'precip_7d':float(np.sum(precip[max(0,ih-7):ih+1])),
                    'prob_lluvia_max':float(np.nanmax(d.get('precipitation_probability_max',[0]*n))),
                    'nivel_alerta':nivel_alerta(float(susc[ih]))[0],
                    'altura':float(susc[ih])*15000,
                    'fuente':'Open-Meteo ERA5 (Real)',
                    'timestamp':datetime.now().strftime('%Y-%m-%d %H:%M')
                })
            else: resultados.append({**punto_sintetico(lat,lon,cuenca['area_km2']),
                                      'nivel_alerta':nivel_alerta(punto_sintetico(lat,lon)['susceptibilidad'])[0],
                                      'altura':punto_sintetico(lat,lon)['susceptibilidad']*15000})
        except: resultados.append({**punto_sintetico(lat,lon,cuenca['area_km2']),
                                    'nivel_alerta':nivel_alerta(punto_sintetico(lat,lon)['susceptibilidad'])[0],
                                    'altura':punto_sintetico(lat,lon)['susceptibilidad']*15000})
    barra.empty(); status.empty()
    df=pd.DataFrame(resultados)
    colores=df['susceptibilidad'].apply(susc_a_rgb).tolist()
    df['r']=[c[0] for c in colores]; df['g']=[c[1] for c in colores]
    df['b']=[c[2] for c in colores]; df['a']=[c[3] for c in colores]
    df['radio_m']=8000+df['susceptibilidad']*12000
    return df


# ════════════════════════════════════════════════════════════
#  SIDEBAR UNIFICADO
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:16px 0 8px'>
      <div style='font-size:2.4rem'>🌊</div>
      <div style='font-size:1.1rem;font-weight:800;letter-spacing:1px'>WAM-IA</div>
      <div style='font-size:.7rem;opacity:.75'>Motor Híbrido Hídrico</div>
      <div style='font-size:.65rem;opacity:.6'>MAKEY × ISN × UNB</div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    pagina = st.radio("", [
        "📊 Dashboard",
        "🛰️ Ingesta Satelital",
        "🗺️ Mapa WAM 3D",
        "🧪 Datos Sintéticos",
        "🏋️ Entrenamiento LSTM",
        "🚨 Alerta Temprana",
    ], label_visibility="collapsed")

    st.divider()
    st.markdown("**🗺️ Cuenca activa**")
    cuenca_nombre = st.selectbox("", list(CUENCAS.keys()),
                                  label_visibility="collapsed")
    cuenca = CUENCAS[cuenca_nombre]

    st.divider()
    st.markdown("**⚙️ Parámetros globales**")
    n_dias       = st.slider("Días a simular",          365,3650,1825,365)
    variabilidad = st.slider("Variabilidad climática",  0.5,2.0,1.0,0.1)
    frec_ext     = st.slider("Frec. eventos extremos", 0.01,0.10,0.03,0.01)

    st.divider()
    if st.button("⚡ Limpiar caché", use_container_width=True):
        st.cache_data.clear()
        for k in list(st.session_state.keys()): st.session_state.pop(k,None)
        st.rerun()

    # Estado de datos disponibles
    st.divider()
    st.markdown("**📦 Estado de datos**")
    st.markdown(f"{'✅' if 'sat_serie' in st.session_state else '⬜'} Serie satelital")
    st.markdown(f"{'✅' if 'grilla_df' in st.session_state else '⬜'} Grilla mapa")
    st.markdown(f"{'✅' if 'model' in st.session_state else '⬜'} Modelo LSTM")

    st.markdown("""
    <div style='text-align:center;margin-top:16px;font-size:.62rem;opacity:.5'>
    CORFO Innova Alta Tecnología 2025<br>MVP v6.0 Unified
    </div>""", unsafe_allow_html=True)


# Datos sintéticos base
df = generar_datos(n_dias, SEED, variabilidad, frec_ext)
n_alto    = int((df['susceptibilidad']>0.60).sum())
n_critico = int((df['susceptibilidad']>0.75).sum())


# ════════════════════════════════════════════════════════════
#  PÁGINA 1 — DASHBOARD
# ════════════════════════════════════════════════════════════
if pagina == "📊 Dashboard":
    st.markdown(f"## 📊 Dashboard WAM-IA — {cuenca['region']}")
    st.caption("Visión general del sistema · indicadores en tiempo real · CORFO TRL-5")
    st.divider()

    # Banner de estado satelital
    if 'sat_serie' in st.session_state:
        ts = st.session_state.get('sat_ts','')
        st.success(f"🛰️ Datos reales activos — {st.session_state.get('sat_fuente','ERA5')} · {ts}")
    else:
        st.info("💡 Ve a **🛰️ Ingesta Satelital** para activar datos reales en tiempo real.")

    st.divider()
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("📅 Días simulados",   f"{len(df):,}")
    c2.metric("🌧️ Precip. media",    f"{df['precipitacion'].mean():.1f} mm")
    c3.metric("💧 DTW promedio",     f"{df['dtw'].mean():.2f} m")
    c4.metric("🌱 Sat. media",       f"{df['saturacion_suelo'].mean():.1f}%")
    c5.metric("⚠️ Días alto riesgo", f"{n_alto}",
              delta=f"{n_alto/len(df)*100:.1f}%", delta_color="inverse")
    c6.metric("🔴 Días críticos",    f"{n_critico}",
              delta=f"{n_critico/len(df)*100:.1f}%", delta_color="inverse")

    st.divider()
    # Serie temporal + distribución
    col_a, col_b = st.columns([2,1])
    with col_a:
        st.markdown("#### Serie de susceptibilidad semanal")
        dfm=df['susceptibilidad'].resample('W').mean().reset_index()
        dfm.columns=['fecha','susceptibilidad']
        line=alt.Chart(dfm).mark_area(
            line={'color':C['azul'],'strokeWidth':2},
            color=alt.Gradient(gradient='linear',x1=0,x2=0,y1=1,y2=0,
                stops=[alt.GradientStop(color='white',offset=0),
                       alt.GradientStop(color=C['azul'],offset=1)])
        ).encode(x=alt.X('fecha:T',title=''),
                 y=alt.Y('susceptibilidad:Q',scale=alt.Scale(domain=[0,1]),title='Susceptibilidad'),
                 tooltip=['fecha:T','susceptibilidad:Q'])
        u60=alt.Chart(pd.DataFrame({'y':[.6]})).mark_rule(color='red',strokeDash=[4,4],strokeWidth=1.5).encode(y='y:Q')
        u75=alt.Chart(pd.DataFrame({'y':[.75]})).mark_rule(color='darkred',strokeDash=[6,3],strokeWidth=1).encode(y='y:Q')
        st.altair_chart((line+u60+u75).properties(height=260), use_container_width=True)

    with col_b:
        st.markdown("#### Distribución de alertas")
        alertas=pd.DataFrame({
            'Nivel':['🟢 Bajo','🟡 Moderado','🟠 Alto','🔴 Crítico'],
            'Días':[int((df['susceptibilidad']<.4).sum()),
                    int(((df['susceptibilidad']>=.4)&(df['susceptibilidad']<.6)).sum()),
                    int(((df['susceptibilidad']>=.6)&(df['susceptibilidad']<.75)).sum()),
                    int((df['susceptibilidad']>=.75).sum())],
            'Color':['#388E3C','#F9A825','#F57C00','#D32F2F']})
        pie=alt.Chart(alertas).mark_arc(innerRadius=55).encode(
            theta=alt.Theta('Días:Q'),
            color=alt.Color('Nivel:N',scale=alt.Scale(
                domain=alertas['Nivel'].tolist(),range=alertas['Color'].tolist())),
            tooltip=['Nivel','Días'])
        st.altair_chart(pie.properties(height=260), use_container_width=True)

    st.divider()
    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("#### Estacionalidad mensual")
        dfmes=df.copy(); dfmes['mes']=dfmes.index.month
        pm=dfmes.groupby('mes')[['precipitacion','susceptibilidad']].mean().reset_index()
        mmap={1:'Ene',2:'Feb',3:'Mar',4:'Abr',5:'May',6:'Jun',
              7:'Jul',8:'Ago',9:'Sep',10:'Oct',11:'Nov',12:'Dic'}
        pm['mes_str']=pm['mes'].map(mmap)
        bars=alt.Chart(pm).mark_bar(color=C['celeste'],opacity=.7).encode(
            x=alt.X('mes_str:N',sort=list(mmap.values()),title=''),
            y=alt.Y('precipitacion:Q',title='Precipitación (mm)'))
        ln=alt.Chart(pm).mark_line(color=C['rojo'],strokeWidth=2.5,point=True).encode(
            x=alt.X('mes_str:N',sort=list(mmap.values())),
            y=alt.Y('susceptibilidad:Q',scale=alt.Scale(domain=[0,1])))
        st.altair_chart(alt.layer(bars,ln).resolve_scale(y='independent')
                        .properties(height=240), use_container_width=True)

    with col_d:
        st.markdown("#### Correlación variables WAM")
        corr=df.corr().round(2).reset_index().melt('index')
        corr.columns=['var1','var2','correlacion']
        hm=alt.Chart(corr).mark_rect().encode(
            x=alt.X('var1:N',title=''),y=alt.Y('var2:N',title=''),
            color=alt.Color('correlacion:Q',scale=alt.Scale(scheme='redblue',domain=[-1,1])),
            tooltip=['var1','var2','correlacion'])
        tx=alt.Chart(corr).mark_text(fontSize=9).encode(
            x='var1:N',y='var2:N',text=alt.Text('correlacion:Q',format='.2f'),
            color=alt.condition(alt.datum.correlacion>0.5,
                                alt.value('white'),alt.value('black')))
        st.altair_chart((hm+tx).properties(height=240), use_container_width=True)


# ════════════════════════════════════════════════════════════
#  PÁGINA 2 — INGESTA SATELITAL
# ════════════════════════════════════════════════════════════
elif pagina == "🛰️ Ingesta Satelital":
    st.markdown(f"## 🛰️ Ingesta Satelital — {cuenca_nombre}")
    st.caption("Open-Meteo ERA5 · datos reales sin autenticación · fallback sintético calibrado")
    st.divider()

    # Info de cuenca
    col_i1, col_i2 = st.columns([1,2])
    with col_i1:
        dias_hist = st.slider("Días de historia", 7, 21, 14)
        btn_ingestar = st.button("🚀 Descargar datos", type="primary", use_container_width=True)
        btn_limpar   = st.button("🗑️ Limpiar", use_container_width=True)
        if btn_limpar:
            for k in ['sat_serie','sat_log','sat_fuente','sat_ts']:
                st.session_state.pop(k,None)
            st.rerun()
    with col_i2:
        lat_c=(cuenca['lat_min']+cuenca['lat_max'])/2
        lon_c=(cuenca['lon_min']+cuenca['lon_max'])/2
        st.markdown(f"""
        | Parámetro | Valor |
        |-----------|-------|
        | Centro | {lat_c:.2f}°S, {lon_c:.2f}°W |
        | Área | {cuenca['area_km2']:,} km² |
        | Región | {cuenca['region']} |
        | Descripción | {cuenca['desc']} |
        """)

    st.divider()

    # Estado APIs
    st.markdown("#### 📡 Estado de fuentes")
    cs = st.columns(4)
    fuentes = [
        ('Open-Meteo ERA5','https://api.open-meteo.com','Precip+Temp+Hum','Sin token'),
        ('NASA GPM IMERG', 'https://gpm.nasa.gov',     'Precipitación',  'NASA Earthdata'),
        ('ESA Sentinel-1', 'https://scihub.copernicus.eu','Humedad SAR', 'Copernicus'),
        ('SNSAT FASat',    'https://agenciaespacial.cl','Imagen 70cm',   'AEXA'),
    ]
    for col_s,(nombre,url,var,auth) in zip(cs,fuentes):
        ok = verificar_api(url)
        with col_s:
            st.markdown(f"""
            <div class='sat-card'>
              <div style='font-size:.78rem;font-weight:700;color:#E8F4FD'>{nombre}</div>
              <div style='font-size:.68rem;color:#8BA8C8;margin:4px 0'>{var}</div>
              <div style='font-size:.62rem;color:#4A90D9'>{auth}</div>
              <div style='margin-top:8px;font-size:.75rem'>
                {'<span class="sat-online">● ONLINE</span>' if ok
                 else '<span class="sat-offline">● STANDBY</span>'}
              </div>
            </div>""", unsafe_allow_html=True)

    # Descarga
    if btn_ingestar:
        for k in ['sat_serie','sat_log','sat_fuente','sat_ts']:
            st.session_state.pop(k,None)
        log=[]
        lat_c=(cuenca['lat_min']+cuenca['lat_max'])/2
        lon_c=(cuenca['lon_min']+cuenca['lon_max'])/2
        with st.spinner("Conectando a Open-Meteo ERA5..."):
            res=descargar_open_meteo_serie(lat_c, lon_c, dias_hist)
        if res['exito']:
            df_sat=enriquecer_wam(res['datos'], cuenca['area_km2'])
            log.append({'paso':f'✅ Open-Meteo ERA5 — {len(df_sat)} días descargados','estado':'ok'})
            log.append({'paso':'✅ Variables WAM calculadas (DTW, saturación, susceptibilidad)','estado':'ok'})
            tipo='real'
        else:
            log.append({'paso':f'⚠️ Open-Meteo no disponible: {res["error"]}','estado':'warn'})
            log.append({'paso':'🔄 Activando datos sintéticos calibrados...','estado':'warn'})
            df_sint=generar_datos_sat_sint(lat_c, lon_c, dias_hist+7)
            df_sat=df_sint
            tipo='sintetico'
        st.session_state['sat_serie']=df_sat
        st.session_state['sat_log']=log
        st.session_state['sat_fuente']=res.get('fuente','Sintético')
        st.session_state['sat_ts']=datetime.now().strftime('%d/%m/%Y %H:%M')
        st.session_state['sat_tipo']=tipo

    if 'sat_serie' in st.session_state:
        df_sat=st.session_state['sat_serie']
        # Log pipeline
        for entry in st.session_state['sat_log']:
            css='pipeline-ok' if entry['estado']=='ok' else 'pipeline-warn'
            st.markdown(f"<div class='{css}'>{entry['paso']}</div>", unsafe_allow_html=True)

        tipo=st.session_state.get('sat_tipo','real')
        badge_c="#388E3C" if tipo=='real' else "#F57C00"
        badge_t="🟢 DATOS REALES ERA5" if tipo=='real' else "🟡 DATOS SINTÉTICOS CALIBRADOS"
        st.markdown(f"""
        <div style='background:{badge_c}22;border:1px solid {badge_c};border-radius:8px;
             padding:10px 16px;margin:10px 0;display:inline-block'>
          <b style='color:{badge_c}'>{badge_t}</b>
          &nbsp;·&nbsp; {len(df_sat)} registros
          &nbsp;·&nbsp; {st.session_state.get('sat_ts','')}
        </div>""", unsafe_allow_html=True)

        st.divider()
        st.markdown("#### 📊 Variables descargadas")
        ult=df_sat.iloc[-1]
        k1,k2,k3,k4,k5,k6=st.columns(6)
        k1.metric("🌧️ Precip.",    f"{ult['precipitacion']:.1f} mm")
        k2.metric("🌡️ Temp.",      f"{ult['temperatura']:.1f} °C")
        k3.metric("💧 DTW",        f"{ult['dtw']:.2f} m")
        k4.metric("🌱 Saturación", f"{ult['saturacion_suelo']:.1f}%")
        k5.metric("🌊 Flujo",      f"{ult['acumulacion_flujo']:.0f} m²")
        k6.metric("⚠️ Susc.",      f"{ult['susceptibilidad']:.3f}")

        # Gráficos
        tab1,tab2,tab3=st.tabs(["🌧️ Hidrometeorología","🌊 Variables WAM","⚠️ Susceptibilidad"])

        with tab1:
            dp=df_sat[['precipitacion','temperatura']].reset_index()
            dp.columns=['fecha','precipitacion','temperatura']
            p_ch=alt.Chart(dp).mark_bar(color=C['celeste'],opacity=.8).encode(
                x=alt.X('fecha:T',title=''),y=alt.Y('precipitacion:Q',title='mm'),
                tooltip=['fecha:T','precipitacion:Q']).properties(height=180,title='Precipitación diaria')
            t_ch=alt.Chart(dp).mark_line(color=C['rojo'],strokeWidth=2,point=True).encode(
                x=alt.X('fecha:T'),y=alt.Y('temperatura:Q',title='°C'),
                tooltip=['fecha:T','temperatura:Q']).properties(height=150,title='Temperatura')
            st.altair_chart(p_ch, use_container_width=True)
            st.altair_chart(t_ch, use_container_width=True)

        with tab2:
            dw=df_sat[['dtw','saturacion_suelo']].reset_index()
            dw.columns=['fecha','dtw','saturacion']
            dtw_ch=alt.Chart(dw).mark_area(
                line={'color':C['naranja'],'strokeWidth':2},
                color=alt.Gradient(gradient='linear',x1=0,x2=0,y1=1,y2=0,
                    stops=[alt.GradientStop(color='white',offset=0),
                           alt.GradientStop(color=C['naranja']+'55',offset=1)])
            ).encode(x=alt.X('fecha:T',title=''),
                     y=alt.Y('dtw:Q',title='DTW (m)',scale=alt.Scale(reverse=True)),
                     tooltip=['fecha:T','dtw:Q']).properties(height=180,title='DTW — eje invertido (arriba = más riesgo)')
            sat_ch=alt.Chart(dw).mark_area(
                line={'color':C['verde'],'strokeWidth':2},
                color=alt.Gradient(gradient='linear',x1=0,x2=0,y1=1,y2=0,
                    stops=[alt.GradientStop(color='white',offset=0),
                           alt.GradientStop(color=C['verde']+'55',offset=1)])
            ).encode(x=alt.X('fecha:T'),y=alt.Y('saturacion:Q',title='%',scale=alt.Scale(domain=[0,100])),
                     tooltip=['fecha:T','saturacion:Q']).properties(height=150,title='Saturación del suelo')
            u80=alt.Chart(pd.DataFrame({'y':[80]})).mark_rule(color='red',strokeDash=[4,4]).encode(y='y:Q')
            st.altair_chart(dtw_ch, use_container_width=True)
            st.altair_chart(sat_ch+u80, use_container_width=True)

        with tab3:
            ds=df_sat[['susceptibilidad']].reset_index()
            ds.columns=['fecha','susceptibilidad']
            ds['alerta']=ds['susceptibilidad'].apply(lambda s:nivel_alerta(s)[0].split(' ',1)[1])
            sc=alt.Chart(ds).mark_area(
                line={'color':C['morado'],'strokeWidth':2.5},
                color=alt.Gradient(gradient='linear',x1=0,x2=0,y1=1,y2=0,
                    stops=[alt.GradientStop(color='white',offset=0),
                           alt.GradientStop(color=C['morado']+'55',offset=1)])
            ).encode(x=alt.X('fecha:T'),y=alt.Y('susceptibilidad:Q',scale=alt.Scale(domain=[0,1])),
                     tooltip=['fecha:T','susceptibilidad:Q','alerta:N'])
            u60=alt.Chart(pd.DataFrame({'y':[.6]})).mark_rule(color='orange',strokeDash=[4,4]).encode(y='y:Q')
            u75=alt.Chart(pd.DataFrame({'y':[.75]})).mark_rule(color='red',strokeDash=[6,3]).encode(y='y:Q')
            st.altair_chart((sc+u60+u75).properties(height=300), use_container_width=True)
            s_act=float(ult['susceptibilidad'])
            niv,chex,cls=nivel_alerta(s_act)
            fondo={"critico":"#FFEBEE","alto":"#FFF3E0","moderado":"#FFFDE7","bajo":"#E8F5E9"}
            st.markdown(f"""
            <div class='alerta-box' style='background:{fondo[cls]};border-left:5px solid {chex}'>
              <b style='font-size:1.1rem;color:{chex}'>Estado actual: {niv}</b><br>
              Susceptibilidad = {s_act:.4f} · DTW = {ult['dtw']:.2f}m · Sat = {ult['saturacion_suelo']:.1f}%
            </div>""", unsafe_allow_html=True)

        st.divider()
        csv=df_sat.reset_index().to_csv(index=False).encode('utf-8')
        col_d1,col_d2=st.columns(2)
        col_d1.download_button("⬇️ Descargar CSV satelital",csv,
                               f"satelital_{cuenca['region'].lower()}_{datetime.now().strftime('%Y%m%d')}.csv",
                               "text/csv",use_container_width=True)
        col_d2.info("💡 Ve a **🗺️ Mapa WAM 3D** para visualizar estos datos sobre el territorio")
    else:
        st.info("👆 Presiona **Descargar datos** para iniciar la ingesta de Open-Meteo ERA5")


# ════════════════════════════════════════════════════════════
#  PÁGINA 3 — MAPA WAM 3D
# ════════════════════════════════════════════════════════════
elif pagina == "🗺️ Mapa WAM 3D":
    st.markdown(f"## 🗺️ Mapa WAM 3D — {cuenca_nombre}")
    st.caption("Visualización geoespacial interactiva · pydeck · Hexágonos 3D · Heatmap · GeoJSON")
    st.divider()

    col_m1, col_m2, col_m3 = st.columns([1,1,2])
    with col_m1:
        resolucion = st.select_slider("Resolución grilla",
            options=[0.50,0.35,0.25,0.15], value=0.35,
            format_func=lambda x: f"{x}° (~{int(x*111)} km)")
        dias_grilla = st.slider("Días historia", 3, 14, 7)
    with col_m2:
        tipo_capa = st.radio("Tipo de visualización",[
            'hexagono_3d','scatter','heatmap'],
            format_func=lambda x:{
                'hexagono_3d':'📊 Hexágonos 3D',
                'scatter':    '⭕ Círculos',
                'heatmap':    '🌡️ Mapa de calor'}[x])
        variable_mapa = st.selectbox("Variable a colorear",[
            'susceptibilidad','precipitacion','dtw','saturacion_suelo'],
            format_func=lambda x:{
                'susceptibilidad':'⚠️ Susceptibilidad WAM',
                'precipitacion':  '🌧️ Precipitación (mm)',
                'dtw':            '💧 DTW (m)',
                'saturacion_suelo':'🌱 Saturación (%)'}[x])
    with col_m3:
        n_pts = int((cuenca['lat_max']-cuenca['lat_min'])/resolucion)*int((cuenca['lon_max']-cuenca['lon_min'])/resolucion)
        st.markdown(f"""
        <div class='info-card'>
          📍 <b>{cuenca_nombre}</b> · {cuenca['area_km2']:,} km²<br>
          🔢 Puntos a descargar: <b>~{n_pts}</b><br>
          📡 Fuente: Open-Meteo ERA5 (sin token)<br>
          🖱️ <b>Clic</b> en cualquier punto para ver datos<br>
          📐 <b>Ctrl+arrastrar</b> para rotar en 3D
        </div>""", unsafe_allow_html=True)

    col_btn1, col_btn2, _ = st.columns([1,1,2])
    btn_mapear  = col_btn1.button("🚀 Descargar + Mapear", type="primary", use_container_width=True)
    btn_limpiar = col_btn2.button("🗑️ Limpiar mapa", use_container_width=True)
    if btn_limpiar:
        for k in ['grilla_df','grilla_ts']: st.session_state.pop(k,None)
        st.rerun()

    if btn_mapear:
        st.session_state.pop('grilla_df',None)
        st.markdown(f"#### 📡 Descargando grilla sobre {cuenca_nombre}...")
        df_g=descargar_grilla(cuenca, resolucion, dias_grilla)
        st.session_state['grilla_df']=df_g
        st.session_state['grilla_ts']=datetime.now().strftime('%d/%m/%Y %H:%M')
        n_real=(df_g['fuente']=='Open-Meteo ERA5 (Real)').sum()
        if n_real > len(df_g)*0.5:
            st.success(f"✅ {len(df_g)} puntos · {n_real} datos reales ERA5")
        else:
            st.warning(f"⚠️ Sin conexión — {len(df_g)} puntos sintéticos calibrados")

    if 'grilla_df' in st.session_state:
        df_g = st.session_state['grilla_df'].copy()
        ts   = st.session_state.get('grilla_ts','')

        # Recalcular colores si cambia la variable
        if variable_mapa != 'susceptibilidad':
            vmin=df_g[variable_mapa].min(); vmax=df_g[variable_mapa].max()
            norm=(df_g[variable_mapa]-vmin)/(vmax-vmin+.001)
            colores=norm.apply(susc_a_rgb).tolist()
            df_g['r']=[c[0] for c in colores]; df_g['g']=[c[1] for c in colores]
            df_g['b']=[c[2] for c in colores]; df_g['a']=[c[3] for c in colores]
            df_g['altura']=norm*15000; df_g['radio_m']=8000+norm*12000
        else:
            colores=df_g['susceptibilidad'].apply(susc_a_rgb).tolist()
            df_g['r']=[c[0] for c in colores]; df_g['g']=[c[1] for c in colores]
            df_g['b']=[c[2] for c in colores]; df_g['a']=[c[3] for c in colores]
            df_g['altura']=df_g['susceptibilidad']*15000
            df_g['radio_m']=8000+df_g['susceptibilidad']*12000

        # KPIs del mapa
        st.markdown(f"**{cuenca_nombre}** · {ts}")
        mk1,mk2,mk3,mk4,mk5 = st.columns(5)
        mk1.metric("⚠️ Susc. máxima",  f"{df_g['susceptibilidad'].max():.3f}")
        mk2.metric("⚠️ Susc. media",   f"{df_g['susceptibilidad'].mean():.3f}")
        mk3.metric("🌧️ Precip. máx",   f"{df_g['precipitacion'].max():.1f} mm")
        mk4.metric("💧 DTW mínimo",    f"{df_g['dtw'].min():.2f} m")
        mk5.metric("📍 Puntos",        f"{len(df_g)}")

        # Tooltip HTML
        tooltip_html = """
        <div style='background:#1B3A6B;color:white;padding:10px;border-radius:8px;
                    font-size:12px;min-width:190px'>
          <b style='font-size:13px'>{nivel_alerta}</b><br>
          <hr style='border-color:#4A90D9;margin:5px 0'>
          📍 {lat:.3f}°S, {lon:.3f}°W<br>
          ⚠️ Susceptibilidad: <b>{susceptibilidad:.3f}</b><br>
          🌧️ Precip. hoy: <b>{precipitacion:.1f} mm</b><br>
          🌧️ Precip. 7d:  <b>{precip_7d:.1f} mm</b><br>
          💧 DTW: <b>{dtw:.2f} m</b><br>
          🌱 Saturación: <b>{saturacion_suelo:.1f}%</b><br>
          🌡️ Temperatura: <b>{temperatura:.1f}°C</b><br>
          <hr style='border-color:#4A90D9;margin:5px 0'>
          <span style='font-size:10px;opacity:.8'>📡 {fuente}</span>
        </div>"""

        view=pdk.ViewState(latitude=cuenca['lat'],longitude=cuenca['lon'],
                           zoom=cuenca['zoom'],pitch=40 if tipo_capa=='hexagono_3d' else 0)

        if tipo_capa=='hexagono_3d':
            capa=pdk.Layer('ColumnLayer',data=df_g,get_position='[lon,lat]',
                           get_elevation='altura',elevation_scale=1,radius=7000,
                           get_fill_color='[r,g,b,a]',pickable=True,auto_highlight=True)
        elif tipo_capa=='scatter':
            capa=pdk.Layer('ScatterplotLayer',data=df_g,get_position='[lon,lat]',
                           get_color='[r,g,b,a]',get_radius='radio_m',
                           pickable=True,filled=True,stroked=True,
                           get_line_color=[255,255,255,80],line_width_min_pixels=1)
        else:
            capa=pdk.Layer('HeatmapLayer',data=df_g,get_position='[lon,lat]',
                           get_weight='susceptibilidad',radiusPixels=90,intensity=2,
                           threshold=0.03,color_range=[
                               [0,100,200,100],[0,200,100,150],
                               [255,200,0,180],[255,100,0,200],[200,0,0,230]])

        mapa=pdk.Deck(layers=[capa],initial_view_state=view,
                      tooltip={"html":tooltip_html} if tipo_capa!='heatmap' else {},
                      map_style='dark')
        st.pydeck_chart(mapa, use_container_width=True)

        # Leyenda + distribución
        col_l1,col_l2,col_l3 = st.columns(3)
        with col_l1:
            st.markdown("""
            **Leyenda:**
            🔴 Crítico ≥0.75 · 🟠 Alto 0.60–0.75
            🟡 Moderado 0.40–0.60 · 🟢 Bajo <0.40
            """)
        with col_l2:
            nc=int((df_g['susceptibilidad']>=.75).sum())
            na=int(((df_g['susceptibilidad']>=.60)&(df_g['susceptibilidad']<.75)).sum())
            nm=int(((df_g['susceptibilidad']>=.40)&(df_g['susceptibilidad']<.60)).sum())
            nb=int((df_g['susceptibilidad']<.40).sum())
            st.markdown(f"🔴 {nc} · 🟠 {na} · 🟡 {nm} · 🟢 {nb} puntos")
        with col_l3:
            n_real_g=(df_g['fuente']=='Open-Meteo ERA5 (Real)').sum()
            badge="🟢 ERA5 Real" if n_real_g>len(df_g)*.5 else "🟡 Sintético"
            st.markdown(f"**Fuente:** {badge} · {n_real_g}/{len(df_g)} puntos reales")

        st.divider()
        # Mapa de calor de precipitación
        st.markdown("#### 🌧️ Heatmap — Precipitación acumulada 7 días")
        df_p=df_g.copy()
        df_p['_w']=df_p['precip_7d']/(df_p['precip_7d'].max()+.001)
        mapa2=pdk.Deck(
            layers=[pdk.Layer('HeatmapLayer',data=df_p,get_position='[lon,lat]',
                              get_weight='_w',radiusPixels=100,intensity=2.5,threshold=0.01,
                              color_range=[[33,150,243,80],[33,150,243,140],
                                           [76,175,80,160],[249,168,37,190],
                                           [245,124,0,210],[211,47,47,240]])],
            initial_view_state=pdk.ViewState(latitude=cuenca['lat'],longitude=cuenca['lon'],
                                             zoom=cuenca['zoom'],pitch=0),
            map_style='dark')
        st.pydeck_chart(mapa2, use_container_width=True)

        st.divider()
        with st.expander("📋 Tabla de datos completa"):
            df_t=df_g[['lat','lon','susceptibilidad','nivel_alerta',
                        'precipitacion','precip_7d','temperatura',
                        'dtw','saturacion_suelo','fuente']].copy()
            df_t=df_t.round(3).sort_values('susceptibilidad',ascending=False)
            st.dataframe(df_t,use_container_width=True,hide_index=True,
                         column_config={'susceptibilidad':st.column_config.ProgressColumn(
                             "Susceptibilidad",min_value=0,max_value=1)})

        col_e1,col_e2=st.columns(2)
        with col_e1:
            csv=df_g.drop(columns=['r','g','b','a','radio_m','altura'],errors='ignore'
                          ).to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Descargar CSV", csv,
                               f"grilla_wam_{cuenca['region'].lower()}_{datetime.now().strftime('%Y%m%d')}.csv",
                               "text/csv", use_container_width=True)
        with col_e2:
            geojson={"type":"FeatureCollection","features":[
                {"type":"Feature",
                 "geometry":{"type":"Point","coordinates":[row['lon'],row['lat']]},
                 "properties":{k:row[k] for k in
                   ['susceptibilidad','nivel_alerta','precipitacion','dtw','saturacion_suelo']}}
                for _,row in df_g.iterrows()]}
            st.download_button("⬇️ Descargar GeoJSON",
                               json.dumps(geojson).encode('utf-8'),
                               f"wam_{cuenca['region'].lower()}_{datetime.now().strftime('%Y%m%d')}.geojson",
                               "application/json", use_container_width=True)
    else:
        st.markdown("#### ▶️ Instrucciones")
        col_ia,col_ib=st.columns(2)
        with col_ia:
            for paso in ["1️⃣ Selecciona la **resolución** (0.25° = más detalle, más lento)",
                         "2️⃣ Elige el **tipo de visualización** (hexágonos, círculos o heatmap)",
                         "3️⃣ Selecciona la **variable** a colorear",
                         "4️⃣ Presiona **Descargar + Mapear**",
                         "5️⃣ **Clic en cualquier punto** del mapa para ver los datos",
                         "6️⃣ Exporta como **CSV o GeoJSON** para QGIS/ArcGIS"]:
                st.markdown(f"<div class='info-card'>{paso}</div>", unsafe_allow_html=True)
        with col_ib:
            df_cuencas=pd.DataFrame([{'Cuenca':k,'Región':v['region'],
                'Área km²':f"{v['area_km2']:,}",'Descripción':v['desc']}
                for k,v in CUENCAS.items()])
            st.dataframe(df_cuencas,use_container_width=True,hide_index=True)


# ════════════════════════════════════════════════════════════
#  PÁGINA 4 — DATOS SINTÉTICOS
# ════════════════════════════════════════════════════════════
elif pagina == "🧪 Datos Sintéticos":
    st.markdown("## 🧪 Generador de Datos Sintéticos Hidrológicos")
    st.caption("Simulación físicamente coherente · parámetros en la barra lateral")
    st.divider()

    variable=st.selectbox("Variable a visualizar",FEATURES+['susceptibilidad'])
    cmap={'precipitacion':C['celeste'],'temperatura':C['rojo'],
          'evapotranspiracion':C['gris'],'dtw':C['naranja'],
          'saturacion_suelo':C['verde'],'acumulacion_flujo':C['azul2'],
          'susceptibilidad':C['morado']}
    col1,col2=st.columns([3,1])
    with col1:
        dv=df[[variable]].reset_index(); dv.columns=['fecha',variable]
        ac=alt.Chart(dv).mark_area(
            line={'color':cmap[variable],'strokeWidth':1.5},
            color=alt.Gradient(gradient='linear',x1=0,x2=0,y1=1,y2=0,
                stops=[alt.GradientStop(color='white',offset=0),
                       alt.GradientStop(color=cmap[variable],offset=1)])
        ).encode(x=alt.X('fecha:T'),y=alt.Y(f'{variable}:Q'),
                 tooltip=['fecha:T',f'{variable}:Q']).properties(height=300)
        rules_vals=([.60,.75] if variable=='susceptibilidad' else
                    [80.0] if variable=='saturacion_suelo' else [])
        cf=ac
        for r in rules_vals:
            cf=cf+alt.Chart(pd.DataFrame({'y':[r]})).mark_rule(
                color='red',strokeDash=[4,4]).encode(y='y:Q')
        if variable=='dtw': cf=cf.properties().configure_scale()
        st.altair_chart(cf, use_container_width=True)
    with col2:
        st.markdown(f"**Estadísticas `{variable}`**")
        for k,v in df[variable].describe().items():
            st.metric(k,f"{v:.3f}")

    st.divider()
    st.markdown("#### 4 variables WAM simultáneas (semanal)")
    dfw=df.resample('W').mean().reset_index(); dfw.columns=['fecha']+list(df.columns)
    charts_p=[]
    for var,color,tipo in [('precipitacion',C['celeste'],'bar'),('dtw',C['naranja'],'area'),
                            ('saturacion_suelo',C['verde'],'area'),('susceptibilidad',C['morado'],'area')]:
        base=alt.Chart(dfw).encode(x=alt.X('fecha:T',title=''),
                                    y=alt.Y(f'{var}:Q',title=var),tooltip=['fecha:T',f'{var}:Q'])
        c=(base.mark_bar(color=color,opacity=.7) if tipo=='bar' else
           base.mark_area(line={'color':color,'strokeWidth':1.5},
               color=alt.Gradient(gradient='linear',x1=0,x2=0,y1=1,y2=0,
                   stops=[alt.GradientStop(color='white',offset=0),
                          alt.GradientStop(color=color,offset=1)])))
        charts_p.append(c.properties(height=125,title=var))
    st.altair_chart(alt.vconcat(*charts_p,spacing=5),use_container_width=True)

    st.divider()
    csv=df.reset_index().to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Descargar dataset CSV",csv,
                       "dataset_sintetico_itata.csv","text/csv",use_container_width=True)


# ════════════════════════════════════════════════════════════
#  PÁGINA 5 — ENTRENAMIENTO LSTM
# ════════════════════════════════════════════════════════════
elif pagina == "🏋️ Entrenamiento LSTM":
    st.markdown("## 🏋️ Entrenamiento LSTM en Vivo")
    st.caption("FloodLSTM · NumPy puro · Adam optimizer · pronóstico 7 días · Motor WAM-IA Fase 2")
    st.divider()

    col_cfg,col_info=st.columns([1,2])
    with col_cfg:
        st.markdown("**Hiperparámetros**")
        ventana  =st.slider("Ventana entrada (días)",      7,30,14)
        horizonte=st.slider("Horizonte pronóstico (días)", 3,14, 7)
        epochs   =st.slider("Épocas de entrenamiento",    10,60,25)
        hidden   =st.select_slider("Neuronas LSTM",  [16,32,64],value=32)
        lr       =st.select_slider("Learning rate",[0.0005,0.001,0.005],value=0.001)
    with col_info:
        # Fuente de entrenamiento
        fuente_tren="📡 Satelital (ERA5)" if 'sat_serie' in st.session_state else "🧪 Sintético"
        st.info(f"**Fuente de datos:** {fuente_tren}  \n"
                f"{'Serie satelital disponible ✅ — el modelo aprenderá de datos reales.' if 'sat_serie' in st.session_state else 'No hay datos satelitales. Ve a 🛰️ Ingesta para activarlos.'}")
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

            # Usar datos satelitales si están disponibles, si no los sintéticos
            df_train = (st.session_state['sat_serie']
                        if 'sat_serie' in st.session_state
                        and len(st.session_state['sat_serie']) > ventana+horizonte+20
                        else df)

            np.random.seed(SEED)
            X_seq,y_seq,scX,scY=preparar_secuencias(df_train,ventana,horizonte)
            n=len(X_seq); n_test=int(n*.2); n_val=int(n*.1); n_train=n-n_test-n_val
            X_tr,y_tr=X_seq[:n_train],y_seq[:n_train]
            X_va,y_va=X_seq[n_train:n_train+n_val],y_seq[n_train:n_train+n_val]
            X_te,y_te=X_seq[n_train+n_val:],y_seq[n_train+n_val:]
            model=NumpyLSTM(len(FEATURES),hidden,horizonte,lr)

            st.markdown("#### 📉 Curva de aprendizaje")
            chart_ph=st.empty(); status_ph=st.empty(); prog_ph=st.progress(0)
            tl_hist,vl_hist=[],[]

            for ep in range(1,epochs+1):
                idx=np.random.permutation(len(X_tr))
                tl_ep=0; nb=0
                for start in range(0,len(idx),32):
                    b=idx[start:start+32]; bl=0
                    for j in b:
                        yp,h,c,cache=model.forward(X_tr[j])
                        bl+=model.backward(X_tr[j],y_tr[j],yp,h,c,cache)
                    tl_ep+=bl/len(b); nb+=1
                tl=tl_ep/nb
                vl=sum(float(np.mean((model.predict(X_va[j])-y_va[j])**2))
                       for j in range(len(X_va)))/max(len(X_va),1)
                tl_hist.append(tl); vl_hist.append(vl)

                if ep%3==0 or ep==epochs:
                    df_live=pd.DataFrame({
                        'época':list(range(1,len(tl_hist)+1))*2,
                        'loss':tl_hist+vl_hist,
                        'tipo':['Train']*len(tl_hist)+['Validación']*len(vl_hist)})
                    live=alt.Chart(df_live).mark_line().encode(
                        x=alt.X('época:Q'),y=alt.Y('loss:Q',scale=alt.Scale(type='log')),
                        color=alt.Color('tipo:N',scale=alt.Scale(
                            domain=['Train','Validación'],range=[C['azul'],C['rojo']])),
                        strokeDash=alt.condition(alt.datum.tipo=='Validación',
                                                 alt.value([6,3]),alt.value([1,0]))
                    ).properties(height=220)
                    chart_ph.altair_chart(live,use_container_width=True)
                    status_ph.info(f"Época {ep}/{epochs} — Train: {tl:.5f} | Val: {vl:.5f}")
                    prog_ph.progress(ep/epochs)

            prog_ph.empty(); status_ph.success(f"✅ Completado · Val Loss: {vl:.5f}")

            y_pred_s=np.array([model.predict(X_te[j]) for j in range(len(X_te))])
            def desnorm(a):
                return np.clip(scY.inverse_transform(a.reshape(-1,1)).flatten().reshape(a.shape),0,1)
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
                        domain=['MAE','RMSE'],range=[C['celeste'],C['azul']])),
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
                        domain=['Real','Predicho'],range=[C['azul'],C['rojo']])),
                    strokeDash=alt.condition(alt.datum.serie=='Predicho',
                                             alt.value([6,3]),alt.value([1,0])))
                ul=alt.Chart(pd.DataFrame({'y':[.6]})).mark_rule(
                    color='orange',strokeDash=[4,4]).encode(y='y:Q')
                st.altair_chart((cc+ul).properties(height=260),use_container_width=True)
    else:
        st.info("👆 Configura los hiperparámetros y presiona **Iniciar entrenamiento**")


# ════════════════════════════════════════════════════════════
#  PÁGINA 6 — ALERTA TEMPRANA
# ════════════════════════════════════════════════════════════
elif pagina == "🚨 Alerta Temprana":
    st.markdown("## 🚨 Boletín de Alerta Temprana — 7 Días")
    st.caption("Pronóstico Basado en Impacto (PBI) · Motor WAM-IA · CORFO TRL-5")
    st.divider()

    if 'model' not in st.session_state:
        st.warning("⚠️ **No hay modelo entrenado.**  \nVe a **🏋️ Entrenamiento LSTM** y entrena el modelo primero.")
        st.stop()

    model=st.session_state['model']
    scX=st.session_state['scX']
    scY=st.session_state['scY']
    ventana=st.session_state['ventana']
    horizonte=st.session_state['horizonte']

    # Selector de fuente
    opciones=["🧪 Datos sintéticos (histórico 5 años)"]
    if 'sat_serie' in st.session_state:
        opciones=["📡 Datos satelitales ERA5 (tiempo real)"]+opciones
    fuente_sel=st.radio("Fuente de datos",opciones,horizontal=True)
    df_alerta=(st.session_state['sat_serie']
               if "satelitales" in fuente_sel and 'sat_serie' in st.session_state
               else df)

    if "satelitales" in fuente_sel:
        ts=st.session_state.get('sat_ts','')
        st.success(f"✅ Usando datos reales ERA5 · {ts} · {len(df_alerta)} días")

    col_f1,col_f2=st.columns([2,1])
    with col_f1:
        fecha_sel=st.date_input("Fecha base del pronóstico",
                                 value=df_alerta.index[-1].date(),
                                 min_value=df_alerta.index[ventana].date(),
                                 max_value=df_alerta.index[-1].date())
    with col_f2:
        st.markdown(" ")
        if st.button("🎲 Evento extremo aleatorio",use_container_width=True):
            cand=df_alerta.index[df_alerta['susceptibilidad']>0.65]
            if len(cand):
                st.session_state['fecha_sel']=cand[np.random.randint(len(cand))].date()
                st.rerun()

    if 'fecha_sel' in st.session_state:
        fecha_sel=st.session_state['fecha_sel']

    try:
        idx_base=df_alerta.index.get_loc(pd.Timestamp(fecha_sel))
        if idx_base<ventana: st.error("Selecciona una fecha más tardía."); st.stop()
        datos_vent=df_alerta[FEATURES].iloc[idx_base-ventana:idx_base].values
        pred_s=model.predict(scX.transform(datos_vent))
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
          Pronóstico {horizonte} días desde {str(fecha_sel)} · {fuente_sel}</span>
        </div>""", unsafe_allow_html=True)

        # Tarjetas diarias
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
            st.markdown("#### Serie histórica + pronóstico WAM-IA")
            hs=max(0,idx_base-45)
            hdf=pd.DataFrame({'fecha':df_alerta.index[hs:idx_base+1],
                               'susceptibilidad':df_alerta['susceptibilidad'].iloc[hs:idx_base+1].values})
            hist_line=alt.Chart(hdf).mark_area(
                line={'color':C['azul'],'strokeWidth':2},
                color=alt.Gradient(gradient='linear',x1=0,x2=0,y1=1,y2=0,
                    stops=[alt.GradientStop(color='white',offset=0),
                           alt.GradientStop(color=C['azul']+'22',offset=1)])
            ).encode(x=alt.X('fecha:T'),
                     y=alt.Y('susceptibilidad:Q',scale=alt.Scale(domain=[0,1.05])))
            pdf=pd.DataFrame({'fecha':fechas_pred,'susceptibilidad':pred_real,
                'color':[nivel_alerta(float(s))[1] for s in pred_real]})
            pred_bars=alt.Chart(pdf).mark_bar(opacity=.5,width=20).encode(
                x='fecha:T',y=alt.Y('susceptibilidad:Q',scale=alt.Scale(domain=[0,1.05])),
                color=alt.Color('color:N',scale=alt.Scale(
                    domain=pdf['color'].tolist(),range=pdf['color'].tolist()),legend=None))
            pred_line=alt.Chart(pdf).mark_line(
                color=C['rojo'],strokeWidth=2.5,strokeDash=[6,3],
                point=alt.OverlayMarkDef(color=C['rojo'],size=80)
            ).encode(x='fecha:T',y=alt.Y('susceptibilidad:Q',scale=alt.Scale(domain=[0,1.05])))
            u60=alt.Chart(pd.DataFrame({'y':[.6]})).mark_rule(color='orange',strokeDash=[4,4],strokeWidth=1.5).encode(y='y:Q')
            u75=alt.Chart(pd.DataFrame({'y':[.75]})).mark_rule(color='red',strokeDash=[6,3],strokeWidth=1.5).encode(y='y:Q')
            st.altair_chart(alt.layer(hist_line,pred_bars,pred_line,u60,u75)
                            .properties(height=340),use_container_width=True)

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
        st.markdown("#### Tabla resumen exportable")
        df_bol=pd.DataFrame({
            'Fecha':          [f.strftime('%d/%m/%Y') for f in fechas_pred],
            'Susceptibilidad':[f"{s:.4f}" for s in pred_real],
            'Nivel':          [nivel_alerta(float(s))[0] for s in pred_real],
            'Acción':         ['Monitoreo continuo'  if nivel_alerta(float(s))[2]=='critico' else
                               'Activar protocolos'  if nivel_alerta(float(s))[2]=='alto'    else
                               'Vigilancia estándar' if nivel_alerta(float(s))[2]=='moderado'else
                               'Sin acción inmediata' for s in pred_real]})
        st.dataframe(df_bol,use_container_width=True,hide_index=True)
        st.download_button("⬇️ Descargar boletín CSV",
                           df_bol.to_csv(index=False).encode('utf-8'),
                           f"boletin_wam_{fecha_sel}.csv","text/csv",use_container_width=True)

    except Exception as e:
        st.error(f"Error al calcular pronóstico: {e}")
