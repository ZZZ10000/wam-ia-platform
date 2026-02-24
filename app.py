# ============================================================
#  🌊 WAM-IA — Módulo de Mapa Satelital Interactivo
#  MAKEY × Integra Sur Norte × UNB
#  v5.0 — Descarga satelital + Visualización geoespacial
#  Open-Meteo ERA5 · pydeck 3D · Folium · Heatmap WAM
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import requests
import json
import pydeck as pdk
from datetime import datetime, timedelta

st.set_page_config(
    page_title="WAM-IA | Mapa Satelital",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  [data-testid="stSidebar"] { background: #0D1B2A; }
  [data-testid="stSidebar"] * { color: #E8F4FD !important; }
  .info-card {
    background: #F0F4FF;
    border-left: 4px solid #1B3A6B;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin: 6px 0;
    font-size: .88rem;
  }
  .sat-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: .75rem;
    font-weight: 700;
    margin: 2px;
  }
  div[data-testid="stMetricValue"] { color: #1B3A6B !important; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  CONFIGURACIÓN DE CUENCAS CHILENAS
# ════════════════════════════════════════════════════════════
CUENCAS = {
    '🌊 Itata — Ñuble': {
        'lat': -36.6, 'lon': -71.8,
        'lat_min': -37.2, 'lat_max': -36.0,
        'lon_min': -72.5, 'lon_max': -71.0,
        'zoom': 8, 'area_km2': 11315,
        'region': 'Ñuble',
        'descripcion': 'Cuenca piloto WAM-IA · validación histórica 2017'
    },
    '🌊 Biobío — Concepción': {
        'lat': -37.8, 'lon': -72.5,
        'lat_min': -38.5, 'lat_max': -37.0,
        'lon_min': -73.5, 'lon_max': -71.5,
        'zoom': 7, 'area_km2': 23695,
        'region': 'Biobío',
        'descripcion': 'Cuenca más grande de Chile central'
    },
    '🌊 Maule — Talca': {
        'lat': -35.8, 'lon': -71.5,
        'lat_min': -36.5, 'lat_max': -35.0,
        'lon_min': -72.0, 'lon_max': -70.5,
        'zoom': 8, 'area_km2': 20280,
        'region': 'Maule',
        'descripcion': 'Cuenca vitivinícola · alta demanda hídrica'
    },
    '🌊 Copiapó — Atacama': {
        'lat': -27.8, 'lon': -69.8,
        'lat_min': -28.5, 'lat_max': -27.0,
        'lon_min': -70.5, 'lon_max': -69.0,
        'zoom': 8, 'area_km2': 18705,
        'region': 'Atacama',
        'descripcion': 'Cuenca validación inundación 2015 · evento extremo'
    },
    '🌊 Aconcagua — Valparaíso': {
        'lat': -32.8, 'lon': -70.5,
        'lat_min': -33.2, 'lat_max': -32.3,
        'lon_min': -71.2, 'lon_max': -70.0,
        'zoom': 9, 'area_km2': 7340,
        'region': 'Valparaíso',
        'descripcion': 'Cuenca urbana · interfase Andes-Costa'
    },
}

# Paleta de colores WAM por nivel de susceptibilidad
def susc_a_color_rgb(s):
    """Convierte susceptibilidad 0-1 a color RGB."""
    if s >= 0.75:   return [211, 47,  47,  200]   # Rojo crítico
    elif s >= 0.60: return [245, 124,  0,  200]   # Naranja alto
    elif s >= 0.40: return [249, 168, 37,  180]   # Amarillo moderado
    elif s >= 0.20: return [ 76, 175, 80,  160]   # Verde bajo
    else:           return [ 33, 150, 243, 140]   # Azul muy bajo


# ════════════════════════════════════════════════════════════
#  DESCARGA OPEN-METEO ERA5 — GRILLA DE PUNTOS
# ════════════════════════════════════════════════════════════
def descargar_grilla_open_meteo(cuenca, resolucion=0.25, dias=7):
    """
    Descarga datos meteorológicos reales para una grilla de puntos
    sobre la cuenca usando Open-Meteo ERA5 (gratis, sin token).

    resolucion: grados entre puntos (0.25° ≈ 25 km)
    Retorna DataFrame con lat, lon y variables hidrológicas.
    """
    lats = np.arange(cuenca['lat_min'], cuenca['lat_max'], resolucion)
    lons = np.arange(cuenca['lon_min'], cuenca['lon_max'], resolucion)
    puntos = [(lat, lon) for lat in lats for lon in lons]

    resultados = []
    barra      = st.progress(0)
    status     = st.empty()
    total      = len(puntos)

    for idx, (lat, lon) in enumerate(puntos):
        barra.progress((idx + 1) / total)
        status.caption(f"📡 Descargando punto {idx+1}/{total} — ({lat:.2f}°, {lon:.2f}°)")

        try:
            url    = "https://api.open-meteo.com/v1/forecast"
            params = {
                'latitude':    lat,
                'longitude':   lon,
                'daily': [
                    'precipitation_sum',
                    'temperature_2m_mean',
                    'et0_fao_evapotranspiration',
                    'soil_moisture_0_to_7cm',
                    'precipitation_probability_max'
                ],
                'past_days':     dias,
                'forecast_days': 7,
                'timezone':      'America/Santiago'
            }
            r = requests.get(url, params=params, timeout=8)

            if r.status_code == 200:
                data   = r.json()['daily']
                n      = len(data['time'])
                precip = np.array(data['precipitation_sum'],     dtype=float)
                temp   = np.array(data['temperature_2m_mean'],   dtype=float)
                etp    = np.array(data['et0_fao_evapotranspiration'], dtype=float)
                humedad= np.array(data.get('soil_moisture_0_to_7cm',
                                           [0.3]*n),              dtype=float)

                # Llenar NaN con 0
                precip  = np.nan_to_num(precip,  nan=0.0)
                temp    = np.nan_to_num(temp,    nan=15.0)
                etp     = np.nan_to_num(etp,     nan=2.0)
                humedad = np.nan_to_num(humedad, nan=0.3)

                # Calcular DTW simplificado
                r7  = np.convolve(precip, np.ones(7)/7,   mode='same')
                dtw = np.clip(3.5 - r7 * 0.05, 0.1, 8.0)

                # Saturación desde humedad volumétrica
                sat  = np.clip(humedad * 250, 5, 100)

                # Acumulación de flujo proxy
                area_proxy = cuenca['area_km2'] * 0.1
                afl  = np.clip(area_proxy + 80 * r7, 200, 15000)

                # Susceptibilidad WAM
                susc = np.clip(
                    np.tanh(precip / 30) * 0.35 +
                    np.exp(-dtw / 2)     * 0.30 +
                    (sat / 100) ** 2     * 0.25 +
                    np.tanh(afl / 5000)  * 0.10, 0, 1)

                # Usar último día histórico (índice -8 = hoy)
                i_hoy = min(dias, n - 1)

                resultados.append({
                    'lat':               lat,
                    'lon':               lon,
                    'precipitacion':     float(precip[i_hoy]),
                    'temperatura':       float(temp[i_hoy]),
                    'evapotranspiracion':float(etp[i_hoy]),
                    'dtw':               float(dtw[i_hoy]),
                    'saturacion_suelo':  float(sat[i_hoy]),
                    'acumulacion_flujo': float(afl[i_hoy]),
                    'susceptibilidad':   float(susc[i_hoy]),
                    'precip_7d':         float(np.sum(precip[max(0,i_hoy-7):i_hoy+1])),
                    'prob_lluvia_max':   float(np.nanmax(
                        data.get('precipitation_probability_max', [0]*n)
                    )),
                    'fuente':            'Open-Meteo ERA5 (Real)',
                    'timestamp':         datetime.now().strftime('%Y-%m-%d %H:%M')
                })
            else:
                resultados.append(_punto_sintetico(lat, lon, cuenca))

        except Exception:
            resultados.append(_punto_sintetico(lat, lon, cuenca))

    barra.empty()
    status.empty()
    return pd.DataFrame(resultados)


def _punto_sintetico(lat, lon, cuenca):
    """Genera un punto sintético calibrado cuando falla la API."""
    np.random.seed(int(abs(lat * 100 + lon * 100)) % 9999)
    precip = max(0, np.random.exponential(5))
    dtw    = max(0.1, 3.5 - precip * 0.05 + np.random.normal(0, 0.3))
    sat    = min(100, max(5, 30 + precip * 2 - dtw * 5))
    susc   = np.clip(
        np.tanh(precip / 30) * 0.35 + np.exp(-dtw / 2) * 0.30 +
        (sat / 100) ** 2 * 0.25 + 0.05, 0, 1)
    return {
        'lat': lat, 'lon': lon,
        'precipitacion': round(precip, 2),
        'temperatura':   round(14 + np.random.normal(0, 3), 1),
        'evapotranspiracion': round(max(0, 2 + np.random.normal(0, 0.5)), 2),
        'dtw':               round(dtw, 3),
        'saturacion_suelo':  round(sat, 1),
        'acumulacion_flujo': round(1500 + precip * 80, 0),
        'susceptibilidad':   round(float(susc), 4),
        'precip_7d':         round(precip * 3, 1),
        'prob_lluvia_max':   round(min(100, precip * 5), 0),
        'fuente':            'Sintético calibrado',
        'timestamp':         datetime.now().strftime('%Y-%m-%d %H:%M')
    }


def enriquecer_colores(df):
    """Agrega columnas de color RGB para pydeck."""
    df = df.copy()
    colores = df['susceptibilidad'].apply(susc_a_color_rgb).tolist()
    df['r'] = [c[0] for c in colores]
    df['g'] = [c[1] for c in colores]
    df['b'] = [c[2] for c in colores]
    df['a'] = [c[3] for c in colores]
    df['nivel_alerta'] = df['susceptibilidad'].apply(lambda s:
        '🔴 CRÍTICO'   if s >= 0.75 else
        '🟠 ALTO'      if s >= 0.60 else
        '🟡 MODERADO'  if s >= 0.40 else
        '🟢 BAJO')
    df['radio_m'] = 8000 + df['susceptibilidad'] * 12000
    df['altura']  = df['susceptibilidad'] * 15000
    return df


# ════════════════════════════════════════════════════════════
#  VISUALIZACIÓN PYDECK
# ════════════════════════════════════════════════════════════
def crear_mapa_pydeck(df, cuenca, tipo_capa='hexagono_3d'):
    """
    Crea mapa interactivo con pydeck.
    Tipos disponibles: hexagono_3d, scatter, heatmap
    """
    view_state = pdk.ViewState(
        latitude=cuenca['lat'],
        longitude=cuenca['lon'],
        zoom=cuenca['zoom'],
        pitch=40 if tipo_capa == 'hexagono_3d' else 0,
        bearing=0
    )

    tooltip_html = """
    <div style='background:#1B3A6B;color:white;padding:10px;border-radius:8px;
                font-family:sans-serif;font-size:12px;min-width:200px'>
      <b style='font-size:14px'>{nivel_alerta}</b><br>
      <hr style='border-color:#4A90D9;margin:6px 0'>
      📍 {lat:.3f}°S, {lon:.3f}°W<br>
      ⚠️ Susceptibilidad: <b>{susceptibilidad:.3f}</b><br>
      🌧️ Precip. hoy: <b>{precipitacion:.1f} mm</b><br>
      🌧️ Precip. 7d:  <b>{precip_7d:.1f} mm</b><br>
      💧 DTW: <b>{dtw:.2f} m</b><br>
      🌱 Saturación: <b>{saturacion_suelo:.1f}%</b><br>
      🌡️ Temperatura: <b>{temperatura:.1f}°C</b><br>
      <hr style='border-color:#4A90D9;margin:6px 0'>
      <span style='font-size:10px;opacity:.8'>📡 {fuente}</span>
    </div>
    """

    if tipo_capa == 'hexagono_3d':
        capa = pdk.Layer(
            'ColumnLayer',
            data=df,
            get_position='[lon, lat]',
            get_elevation='altura',
            elevation_scale=1,
            radius=8000,
            get_fill_color='[r, g, b, a]',
            pickable=True,
            auto_highlight=True,
            coverage=0.9
        )

    elif tipo_capa == 'scatter':
        capa = pdk.Layer(
            'ScatterplotLayer',
            data=df,
            get_position='[lon, lat]',
            get_color='[r, g, b, a]',
            get_radius='radio_m',
            pickable=True,
            opacity=0.8,
            stroked=True,
            filled=True,
            radius_scale=1,
            radius_min_pixels=4,
            radius_max_pixels=60,
            line_width_min_pixels=1,
            get_line_color=[255, 255, 255, 80]
        )

    elif tipo_capa == 'heatmap':
        capa = pdk.Layer(
            'HeatmapLayer',
            data=df,
            get_position='[lon, lat]',
            get_weight='susceptibilidad',
            radiusPixels=80,
            intensity=1.5,
            threshold=0.05,
            color_range=[
                [33,  150, 243, 120],
                [76,  175,  80, 160],
                [249, 168,  37, 180],
                [245, 124,   0, 200],
                [211,  47,  47, 220],
                [183,  28,  28, 255],
            ]
        )

    mapa = pdk.Deck(
        layers=[capa],
        initial_view_state=view_state,
        tooltip={"html": tooltip_html} if tipo_capa != 'heatmap' else {},
        map_style='mapbox://styles/mapbox/satellite-streets-v12',
        map_provider='carto',
    )
    return mapa


def crear_mapa_satelite_base(df, cuenca, tipo_capa):
    """
    Versión con mapa base satelital usando carto dark matter.
    """
    view_state = pdk.ViewState(
        latitude=cuenca['lat'],
        longitude=cuenca['lon'],
        zoom=cuenca['zoom'],
        pitch=35 if tipo_capa == 'hexagono_3d' else 0,
    )

    tooltip = {
        "html": """
        <b>{nivel_alerta}</b><br>
        Susc: {susceptibilidad} | Precip: {precipitacion}mm<br>
        DTW: {dtw}m | Sat: {saturacion_suelo}%
        """
    }

    capas = []

    # Capa principal WAM
    if tipo_capa == 'hexagono_3d':
        capas.append(pdk.Layer(
            'ColumnLayer', data=df,
            get_position='[lon, lat]',
            get_elevation='altura',
            radius=7000,
            get_fill_color='[r, g, b, 200]',
            pickable=True, auto_highlight=True
        ))

    elif tipo_capa == 'scatter':
        capas.append(pdk.Layer(
            'ScatterplotLayer', data=df,
            get_position='[lon, lat]',
            get_color='[r, g, b, 180]',
            get_radius='radio_m',
            pickable=True, filled=True, stroked=True,
            get_line_color=[255,255,255,100],
            line_width_min_pixels=1
        ))

    elif tipo_capa == 'heatmap':
        capas.append(pdk.Layer(
            'HeatmapLayer', data=df,
            get_position='[lon, lat]',
            get_weight='susceptibilidad',
            radiusPixels=90, intensity=2, threshold=0.03,
            color_range=[
                [0,   100, 200, 100],
                [0,   200, 100, 150],
                [255, 200,   0, 180],
                [255, 100,   0, 200],
                [200,   0,   0, 230],
            ]
        ))

    # Capa de texto con susceptibilidad
    if tipo_capa != 'heatmap' and len(df) <= 50:
        capas.append(pdk.Layer(
            'TextLayer', data=df,
            get_position='[lon, lat]',
            get_text='nivel_alerta',
            get_size=12,
            get_color=[255,255,255,200],
            get_alignment_baseline="'bottom'"
        ))

    return pdk.Deck(
        layers=capas,
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style='dark',
    )


# ════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:16px 0 8px'>
      <div style='font-size:2.5rem'>🛰️</div>
      <div style='font-size:1.1rem;font-weight:800'>WAM-IA MAPS</div>
      <div style='font-size:.7rem;opacity:.75'>Visualización Satelital</div>
      <div style='font-size:.65rem;opacity:.6'>MAKEY × ISN × UNB</div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    st.markdown("**🗺️ Cuenca**")
    cuenca_nombre = st.selectbox("", list(CUENCAS.keys()),
                                  label_visibility="collapsed")
    cuenca = CUENCAS[cuenca_nombre]

    st.divider()
    st.markdown("**📡 Configuración**")
    resolucion = st.select_slider(
        "Resolución de la grilla",
        options=[0.50, 0.35, 0.25, 0.15],
        value=0.35,
        format_func=lambda x: f"{x}° (~{int(x*111)} km)"
    )
    dias_hist = st.slider("Días de historia ERA5", 3, 14, 7)

    st.divider()
    st.markdown("**🎨 Tipo de visualización**")
    tipo_capa = st.radio("", [
        'hexagono_3d',
        'scatter',
        'heatmap'
    ], format_func=lambda x: {
        'hexagono_3d': '📊 Hexágonos 3D (susceptibilidad)',
        'scatter':     '⭕ Círculos (radio = riesgo)',
        'heatmap':     '🌡️ Mapa de calor'
    }[x], label_visibility="collapsed")

    st.divider()
    st.markdown("**🔍 Variable a mapear**")
    variable_mapa = st.selectbox("", [
        'susceptibilidad',
        'precipitacion',
        'dtw',
        'saturacion_suelo',
        'temperatura'
    ], format_func=lambda x: {
        'susceptibilidad': '⚠️ Susceptibilidad WAM',
        'precipitacion':   '🌧️ Precipitación (mm)',
        'dtw':             '💧 Depth-to-Water (m)',
        'saturacion_suelo':'🌱 Saturación del suelo (%)',
        'temperatura':     '🌡️ Temperatura (°C)'
    }[x], label_visibility="collapsed")

    st.divider()
    btn_descargar = st.button("🚀 Descargar + Mapear",
                               type="primary", use_container_width=True)
    btn_limpiar   = st.button("🗑️ Limpiar", use_container_width=True)

    if btn_limpiar:
        for k in ['grilla_df','grilla_cuenca','grilla_timestamp']:
            st.session_state.pop(k, None)
        st.rerun()

    st.markdown(f"""
    <div style='margin-top:16px;font-size:.7rem;opacity:.7'>
      <b>Cuenca:</b> {cuenca['region']}<br>
      <b>Área:</b> {cuenca['area_km2']:,} km²<br>
      <b>Puntos grilla:</b> ~{int((cuenca['lat_max']-cuenca['lat_min'])/resolucion) * int((cuenca['lon_max']-cuenca['lon_min'])/resolucion)}<br>
      <b>Fuente:</b> Open-Meteo ERA5
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  PÁGINA PRINCIPAL
# ════════════════════════════════════════════════════════════
st.markdown(f"## 🛰️ Mapa Satelital WAM-IA — {cuenca_nombre}")
st.caption(f"{cuenca['descripcion']} · Open-Meteo ERA5 · pydeck · Tiempo real")
st.divider()


# ── Proceso de descarga ──
if btn_descargar:
    st.session_state.pop('grilla_df', None)

    col_prog, _ = st.columns([2, 1])
    with col_prog:
        st.markdown(f"#### 📡 Descargando datos para {cuenca_nombre}...")
        st.markdown(f"""
        <div class='info-card'>
          🌐 Conectando a <b>Open-Meteo ERA5</b><br>
          📍 Grilla: {resolucion}° × {resolucion}° sobre la cuenca<br>
          📅 Ventana: últimos {dias_hist} días + pronóstico 7 días
        </div>""", unsafe_allow_html=True)

    df_grilla = descargar_grilla_open_meteo(cuenca, resolucion, dias_hist)

    if df_grilla is not None and len(df_grilla) > 0:
        st.session_state['grilla_df']        = df_grilla
        st.session_state['grilla_cuenca']    = cuenca_nombre
        st.session_state['grilla_timestamp'] = datetime.now().strftime('%d/%m/%Y %H:%M')
        n_real = (df_grilla['fuente'] == 'Open-Meteo ERA5 (Real)').sum()
        if n_real > len(df_grilla) * 0.5:
            st.success(f"✅ {len(df_grilla)} puntos descargados — {n_real} datos reales ERA5")
        else:
            st.warning(f"⚠️ {len(df_grilla)} puntos · Sin conexión a internet — usando datos sintéticos calibrados")


# ── Mostrar mapa si hay datos ──
if 'grilla_df' in st.session_state:
    df_g   = st.session_state['grilla_df'].copy()
    ts     = st.session_state.get('grilla_timestamp', '')
    n_real = (df_g['fuente'] == 'Open-Meteo ERA5 (Real)').sum()

    # Recalcular colores según variable seleccionada
    if variable_mapa != 'susceptibilidad':
        # Normalizar la variable seleccionada para colorear
        vmin = df_g[variable_mapa].min()
        vmax = df_g[variable_mapa].max()
        if vmax > vmin:
            df_g['_norm'] = (df_g[variable_mapa] - vmin) / (vmax - vmin)
        else:
            df_g['_norm'] = 0.5
        df_g['altura'] = df_g['_norm'] * 15000
        colores = df_g['_norm'].apply(lambda s: susc_a_color_rgb(s)).tolist()
        df_g['r'] = [c[0] for c in colores]
        df_g['g'] = [c[1] for c in colores]
        df_g['b'] = [c[2] for c in colores]
        df_g['a'] = [c[3] for c in colores]
    else:
        df_g = enriquecer_colores(df_g)

    # ── KPIs ──
    st.markdown(f"**📊 Estado actual — {cuenca_nombre}**  ·  _{ts}_")
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    susc_max  = df_g['susceptibilidad'].max()
    susc_mean = df_g['susceptibilidad'].mean()
    niv_max   = ('🔴 CRÍTICO' if susc_max>=0.75 else '🟠 ALTO' if susc_max>=0.60
                 else '🟡 MODERADO' if susc_max>=0.40 else '🟢 BAJO')

    c1.metric("⚠️ Susc. máxima",  f"{susc_max:.3f}",  niv_max)
    c2.metric("⚠️ Susc. media",   f"{susc_mean:.3f}")
    c3.metric("🌧️ Precip. max",   f"{df_g['precipitacion'].max():.1f} mm")
    c4.metric("💧 DTW mínimo",    f"{df_g['dtw'].min():.2f} m",
              delta="Mayor riesgo", delta_color="inverse")
    c5.metric("🌱 Sat. máxima",   f"{df_g['saturacion_suelo'].max():.1f}%")
    c6.metric("📡 Puntos reales", f"{n_real}/{len(df_g)}")

    st.divider()

    # ── Mapa principal pydeck ──
    st.markdown(f"#### 🗺️ Mapa interactivo — {variable_mapa.replace('_',' ').title()}")

    mapa = crear_mapa_satelite_base(df_g, cuenca, tipo_capa)
    st.pydeck_chart(mapa, use_container_width=True)

    # Leyenda
    col_ley1, col_ley2, col_ley3 = st.columns([1,1,2])
    with col_ley1:
        st.markdown("""
        **Leyenda de susceptibilidad:**
        <div style='font-size:.85rem;line-height:1.8'>
        🔴 <b>Crítico</b>  ≥ 0.75<br>
        🟠 <b>Alto</b>     0.60–0.75<br>
        🟡 <b>Moderado</b> 0.40–0.60<br>
        🟢 <b>Bajo</b>     < 0.40
        </div>
        """, unsafe_allow_html=True)
    with col_ley2:
        n_crit = int((df_g['susceptibilidad']>=0.75).sum())
        n_alto = int(((df_g['susceptibilidad']>=0.60)&(df_g['susceptibilidad']<0.75)).sum())
        n_mod  = int(((df_g['susceptibilidad']>=0.40)&(df_g['susceptibilidad']<0.60)).sum())
        n_bajo = int((df_g['susceptibilidad']<0.40).sum())
        st.markdown(f"""
        **Distribución de puntos:**
        <div style='font-size:.85rem;line-height:1.8'>
        🔴 {n_crit} puntos críticos<br>
        🟠 {n_alto} puntos alto<br>
        🟡 {n_mod} puntos moderado<br>
        🟢 {n_bajo} puntos bajo
        </div>
        """, unsafe_allow_html=True)
    with col_ley3:
        st.markdown("""
        **Controles del mapa:**
        <div style='font-size:.82rem;line-height:2'>
        🖱️ <b>Clic</b> en un punto → ver datos detallados<br>
        🔄 <b>Arrastar</b> → mover el mapa<br>
        🔍 <b>Scroll</b> → zoom in/out<br>
        📐 <b>Ctrl + arrastar</b> → rotar en 3D (hexágonos)<br>
        🔧 Cambia el tipo de capa en la barra lateral
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Mapa de calor de precipitación ──
    st.markdown("#### 🌧️ Mapa de calor — Precipitación acumulada 7 días")
    df_precip = df_g.copy()
    df_precip['_norm'] = df_precip['precip_7d'] / (df_precip['precip_7d'].max() + 0.001)
    colores_p = df_precip['_norm'].apply(lambda s: susc_a_color_rgb(s)).tolist()
    df_precip['r'] = [c[0] for c in colores_p]
    df_precip['g'] = [c[1] for c in colores_p]
    df_precip['b'] = [c[2] for c in colores_p]

    mapa_precip = pdk.Deck(
        layers=[
            pdk.Layer(
                'HeatmapLayer', data=df_precip,
                get_position='[lon, lat]',
                get_weight='_norm',
                radiusPixels=100, intensity=2, threshold=0.01,
                color_range=[
                    [33,  150, 243, 80],
                    [33,  150, 243, 140],
                    [76,  175,  80, 160],
                    [249, 168,  37, 190],
                    [245, 124,   0, 210],
                    [211,  47,  47, 240],
                ]
            )
        ],
        initial_view_state=pdk.ViewState(
            latitude=cuenca['lat'], longitude=cuenca['lon'],
            zoom=cuenca['zoom'], pitch=0
        ),
        map_style='dark',
        tooltip={"text": "Precip 7d: {precip_7d} mm"}
    )
    st.pydeck_chart(mapa_precip, use_container_width=True)

    st.divider()

    # ── Tabla de datos ──
    with st.expander("📋 Ver tabla de datos completa"):
        df_tabla = df_g[[
            'lat','lon','susceptibilidad','nivel_alerta',
            'precipitacion','precip_7d','temperatura',
            'dtw','saturacion_suelo','fuente'
        ]].copy()
        df_tabla = df_tabla.round(3).sort_values('susceptibilidad', ascending=False)
        st.dataframe(df_tabla, use_container_width=True, hide_index=True,
                     column_config={
                         'susceptibilidad': st.column_config.ProgressColumn(
                             "Susceptibilidad", min_value=0, max_value=1),
                         'saturacion_suelo': st.column_config.ProgressColumn(
                             "Saturación %", min_value=0, max_value=100),
                     })

    # ── Descarga ──
    col_d1, col_d2, _ = st.columns([1, 1, 2])
    with col_d1:
        csv = df_g.drop(columns=['r','g','b','a','radio_m','altura'],
                         errors='ignore').to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Descargar CSV completo", csv,
                           f"grilla_satelital_{cuenca['region'].lower()}_{datetime.now().strftime('%Y%m%d')}.csv",
                           "text/csv", use_container_width=True)
    with col_d2:
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [row['lon'], row['lat']]},
                    "properties": {
                        "susceptibilidad": row['susceptibilidad'],
                        "nivel_alerta":    row['nivel_alerta'],
                        "precipitacion":   row['precipitacion'],
                        "dtw":             row['dtw'],
                        "saturacion":      row['saturacion_suelo']
                    }
                }
                for _, row in df_g.iterrows()
            ]
        }
        st.download_button("⬇️ Descargar GeoJSON", json.dumps(geojson).encode('utf-8'),
                           f"wam_ia_{cuenca['region'].lower()}_{datetime.now().strftime('%Y%m%d')}.geojson",
                           "application/json", use_container_width=True)


# ── Estado inicial ──
else:
    st.markdown("### Cómo usar este módulo")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        #### 📋 Pasos
        <div class='info-card'>1️⃣ Selecciona la <b>cuenca</b> en la barra lateral</div>
        <div class='info-card'>2️⃣ Elige la <b>resolución</b> de la grilla (0.25° = ~25 km entre puntos)</div>
        <div class='info-card'>3️⃣ Selecciona el <b>tipo de visualización</b> (hexágonos 3D, círculos, heatmap)</div>
        <div class='info-card'>4️⃣ Presiona <b>"Descargar + Mapear"</b></div>
        <div class='info-card'>5️⃣ Haz <b>clic en cualquier punto</b> del mapa para ver los datos</div>
        <div class='info-card'>6️⃣ Descarga los datos como <b>CSV o GeoJSON</b> para QGIS/ArcGIS</div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        #### 📡 Fuentes de datos

        | Fuente | Variables | Auth |
        |--------|-----------|------|
        | Open-Meteo ERA5 | Precip, Temp, ETP, Humedad | ❌ No requiere |
        | NASA GPM IMERG | Precipitación | ✅ NASA Earthdata |
        | ESA Sentinel-1 | Humedad SAR | ✅ Copernicus |
        | SNSAT FASat | Imagen 70cm | ✅ AEXA |

        #### 🗺️ Formatos de exportación
        - **CSV** — análisis en Python/R/Excel
        - **GeoJSON** — importar en QGIS, ArcGIS, Google Maps
        - **Compatibles con** Copernicus EMS, DGA-Chile

        #### ⚙️ Variables WAM calculadas
        Las variables **DTW, saturación y susceptibilidad**
        se calculan automáticamente aplicando la
        física del modelo WAM sobre los datos meteorológicos descargados.
        """)

    st.divider()
    st.markdown("#### 🌍 Cuencas disponibles")
    df_cuencas = pd.DataFrame([
        {'Cuenca': k,
         'Región': v['region'],
         'Área km²': f"{v['area_km2']:,}",
         'Descripción': v['descripcion']}
        for k, v in CUENCAS.items()
    ])
    st.dataframe(df_cuencas, use_container_width=True, hide_index=True)
