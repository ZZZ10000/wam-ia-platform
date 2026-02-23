# 🌊 WAM-IA — Plataforma Híbrida de Inteligencia Hídrica

**Motor Híbrido WAM-IA** · MAKEY × Integra Sur Norte × UNB  
CORFO Innova Alta Tecnología 2025 — Fase 2 MVP

---

## 🚀 Deploy en Streamlit Cloud (5 minutos)

### Paso 1 — Subir a GitHub
1. Crea un repositorio nuevo en GitHub (puede ser privado)
2. Sube los dos archivos:
   - `app.py`
   - `requirements.txt`

```bash
git init
git add app.py requirements.txt README.md
git commit -m "WAM-IA Platform v1.0"
git remote add origin https://github.com/TU_USUARIO/wam-ia-platform.git
git push -u origin main
```

### Paso 2 — Deploy en Streamlit Cloud
1. Ve a [share.streamlit.io](https://share.streamlit.io)
2. Inicia sesión con GitHub
3. Clic en **"New app"**
4. Selecciona tu repositorio y rama `main`
5. Archivo principal: `app.py`
6. Clic en **"Deploy!"**

✅ En ~3 minutos tienes la URL pública lista para compartir con Gustavo.

---

## 📋 Módulos de la plataforma

| Módulo | Descripción |
|--------|-------------|
| 📊 **Dashboard** | KPIs, serie temporal, correlaciones, estacionalidad |
| 🧪 **Datos Sintéticos** | Generador hidrológico configurable + descarga CSV |
| 🏋️ **Entrenamiento LSTM** | Training en vivo con curva de aprendizaje + métricas |
| 🚨 **Alerta Temprana** | Boletín interactivo PBI 7 días + exportación |

## ⚙️ Parámetros configurables (barra lateral)
- Días a simular (365–3650)
- Variabilidad climática
- Frecuencia de eventos extremos
- Hiperparámetros del modelo LSTM

## 🔄 Flujo de uso recomendado
```
1. Ajustar parámetros en barra lateral
2. Explorar datos en 🧪 Datos Sintéticos
3. Entrenar modelo en 🏋️ Entrenamiento LSTM
4. Generar boletín en 🚨 Alerta Temprana
```

---

## 🏗️ Stack tecnológico

- **Frontend:** Streamlit 1.35
- **Modelo:** PyTorch 2.2 (FloodLSTM 2 capas)
- **Visualización:** Plotly 5.22
- **Datos:** NumPy + Pandas

---

*Plataforma desarrollada por MAKEY como MVP de la Fase 2 del proyecto Motor Híbrido WAM-IA.*
