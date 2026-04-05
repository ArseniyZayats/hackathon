import streamlit as st
import tempfile
import os
import sys
import importlib
import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai

# =============================================================================
# 1. БЕЗПЕЧНИЙ ІМПОРТ МОДУЛІВ КОМАНДИ
# =============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from parser import parse_bin 
    from CoreAnalytics.coreAnalytics import merge_data, compute_all_metrics
    vis_module = importlib.import_module("Visual.visual")
    generate_3d_model = getattr(vis_module, "generate_3d_model")
except Exception as e:
    st.error(f"⚠️ Помилка імпорту модулів: {e}")

# =============================================================================
# 2. НАЛАШТУВАННЯ СТОРІНКИ
# =============================================================================
st.set_page_config(page_title="BEST UAV Analyzer", page_icon="🚁", layout="wide")
st.title("🚁 Система аналізу телеметрії БПЛА")
st.markdown("Інтерактивний дашборд для розшифровки \"чорних скриньок\" Ardupilot.")

# =============================================================================
# 3. ФУНКЦІЇ ОБРОБКИ
# =============================================================================
@st.cache_data
def process_flight_data(file_path):
    gps_df, imu_df, att_df, mode_df, curr_df = parse_bin(file_path)
    df = merge_data(gps_df, imu_df)
    metrics = compute_all_metrics(df)
    return df, metrics, att_df, mode_df, curr_df

# =============================================================================
# 4. БІЧНА ПАНЕЛЬ
# =============================================================================
with st.sidebar:
    st.header("⚙️ Завантаження даних")
    # Дозволяємо мультизавантаження
    uploaded_files = st.file_uploader("Завантажте лог-файли (.BIN)", type=['bin'], accept_multiple_files=True)
    
    st.divider()
    st.info("💡 **Порада:** Завантажте ДВА файли одночасно, щоб увімкнути режим порівняння місій!")

# =============================================================================
# 5. ГОЛОВНИЙ ІНТЕРФЕЙС
# =============================================================================
if uploaded_files:
    if len(uploaded_files) > 2:
        st.warning("⚠️ Для коректного порівняння завантажте не більше 2-х файлів.")
    else:
        with tempfile.TemporaryDirectory() as tmp_dir:
            
            # Зберігаємо файли
            for f in uploaded_files:
                with open(os.path.join(tmp_dir, f.name), "wb") as out:
                    out.write(f.getbuffer())
            
            with st.spinner("Аналіз місій, пошук критичних точок та рендеринг..."):
                try:
                    # Обробка даних алгоритмами
                    flights_data = []
                    for f in uploaded_files:
                        path = os.path.join(tmp_dir, f.name)
                        df, metrics, att_df, mode_df, curr_df = process_flight_data(path)
                        flights_data.append({
                            "name": f.name, "df": df, "metrics": metrics, 
                            "mode_df": mode_df, "curr_df": curr_df
                        })
                    
                    # Генерація 3D (спіймає всі файли в папці)
                    fig_3d = generate_3d_model(target_folder=tmp_dir)
                    st.success("Дані успішно оброблено!")
                    
                    # --- БЛОК 1: МЕТРИКИ ---
                    if len(flights_data) == 1:
                        st.subheader(f"📊 Показники місії: {flights_data[0]['name']}")
                        m1 = flights_data[0]['metrics']
                        cols = st.columns(4)
                        for i, (k, v) in enumerate(m1.items()):
                            val_str = f"{v:.2f}" if isinstance(v, float) else str(v)
                            cols[i % 4].metric(k.replace("_", " ").title(), val_str)
                            
                    elif len(flights_data) == 2:
                        st.subheader("⚖️ Порівняльний аналіз місій")
                        m1, m2 = flights_data[0]['metrics'], flights_data[1]['metrics']
                        name1, name2 = flights_data[0]['name'], flights_data[1]['name']
                        st.markdown(f"**Базовий:** `{name1}` | **Порівнюється з:** `{name2}`")
                        
                        cols = st.columns(4)
                        for i, key in enumerate(m1.keys()):
                            if key in m2:
                                val1, val2 = m1[key], m2[key]
                                if isinstance(val1, (int, float)):
                                    diff = val2 - val1
                                    cols[i % 4].metric(
                                        label=key.replace("_", " ").title(), 
                                        value=f"{val2:.2f}", 
                                        delta=f"{diff:.2f} (відносно {name1})",
                                        delta_color="normal"
                                    )
                    st.divider()
                    
                    # --- БЛОК 2: ПРОСТОРОВА ТРАЄКТОРІЯ (2D + 3D) ---
                    st.subheader("🗺️ Просторова траєкторія")
                    tab_2d, tab_3d = st.tabs(["🗺️ 2D Карта (GPS) з критичними точками", "🧊 3D Анімація польоту"])
                    
                    with tab_2d:
                        fig_map = go.Figure()
                        colors = ['#ff0055', '#00e5ff'] # Палітра для різних польотів
                        start_lat, start_lon = None, None
                        
                        for idx, flight in enumerate(flights_data):
                            df_map = flight['df']
                            if 'lat_deg' in df_map.columns and 'lon_deg' in df_map.columns and not df_map.empty:
                                if start_lat is None:
                                    start_lat, start_lon = df_map['lat_deg'].iloc[0], df_map['lon_deg'].iloc[0]
                                
                                # 1. Малюємо основну лінію маршруту
                                fig_map.add_trace(go.Scattermapbox(
                                    lat=df_map['lat_deg'], lon=df_map['lon_deg'],
                                    mode='lines', line=dict(width=4, color=colors[idx % 2]), 
                                    name=f"Маршрут {flight['name']}"
                                ))
                                
                                # 2. Шукаємо координати КРИТИЧНИХ ТОЧОК
                                c_lats, c_lons, c_texts = [], [], []
                                
                                # Старт і Фініш
                                c_lats.extend([df_map['lat_deg'].iloc[0], df_map['lat_deg'].iloc[-1]])
                                c_lons.extend([df_map['lon_deg'].iloc[0], df_map['lon_deg'].iloc[-1]])
                                c_texts.extend(["🛫 Старт", "🛬 Фініш"])
                                
                                # Максимальна висота
                                if 'alt_m' in df_map.columns:
                                    idx_max_alt = df_map['alt_m'].idxmax()
                                    c_lats.append(df_map['lat_deg'].loc[idx_max_alt])
                                    c_lons.append(df_map['lon_deg'].loc[idx_max_alt])
                                    c_texts.append(f"⛰️ Макс. висота ({df_map['alt_m'].max():.1f}м)")
                                    
                                # Максимальна швидкість
                                if 'gps_speed' in df_map.columns:
                                    idx_max_spd = df_map['gps_speed'].idxmax()
                                    c_lats.append(df_map['lat_deg'].loc[idx_max_spd])
                                    c_lons.append(df_map['lon_deg'].loc[idx_max_spd])
                                    c_texts.append(f"⚡ Макс. швидкість ({df_map['gps_speed'].max():.1f}м/с)")
                                
                                # 3. Малюємо виколоті крапки (білі маркери поверх лінії)
                                fig_map.add_trace(go.Scattermapbox(
                                    lat=c_lats, lon=c_lons,
                                    mode='markers+text',
                                    marker=dict(size=12, color='white', opacity=0.9), # "Виколота" крапка
                                    text=c_texts,
                                    textposition="top right",
                                    textfont=dict(color='white', size=12),
                                    name=f"Точки {flight['name']}",
                                    showlegend=False
                                ))

                        if start_lat is not None:
                            fig_map.update_layout(
                                mapbox_style="open-street-map",
                                mapbox=dict(center=dict(lat=start_lat, lon=start_lon), zoom=15),
                                margin=dict(l=0, r=0, b=0, t=0), height=500
                            )
                            st.plotly_chart(fig_map, use_container_width=True)
                        else:
                            st.warning("GPS координати відсутні.")

                    with tab_3d:
                        if fig_3d:
                            st.plotly_chart(fig_3d, use_container_width=True)
                        else:
                            st.error("Помилка генерації 3D-анімації.")
                    
                    st.divider()
                    
                    # --- БЛОК 3: БАТАРЕЯ ТА РЕЖИМИ ---
                    # Якщо файлів кілька, створюємо вкладки для кожного
                    flight_tabs = st.tabs([f"Дані: {f['name']}" for f in flights_data])
                    for idx, f_tab in enumerate(flight_tabs):
                        with f_tab:
                            c_bat, c_mode = st.columns([2, 1])
                            with c_bat:
                                st.subheader("🔋 Аналіз живлення")
                                if not flights_data[idx]['curr_df'].empty and 'voltage' in flights_data[idx]['curr_df'].columns:
                                    st.line_chart(flights_data[idx]['curr_df'], x='timestamp', y='voltage', color="#ff4b4b")
                                else:
                                    st.info("Датчик живлення не був підключений.")
                            with c_mode:
                                st.subheader("🕹️ Режими польоту")
                                if not flights_data[idx]['mode_df'].empty:
                                    st.dataframe(flights_data[idx]['mode_df'][['timestamp', 'mode_name']], use_container_width=True, hide_index=True)
                                else:
                                    st.info("Дані про режими відсутні.")
                                    
                    st.divider()
                    
                    # --- БЛОК 4: AI АСИСТЕНТ ---
                    st.subheader("🤖 AI Аналітик")
                    if st.button("Згенерувати автоматичний звіт", type="primary"):
                        with st.spinner("AI працює..."):
                            api_key = st.secrets.get("GEMINI_API_KEY", "")
                            if not api_key:
                                st.error("⚠️ API-ключ не знайдено.")
                            else:
                                genai.configure(api_key=api_key)
                                model = genai.GenerativeModel('gemini-1.5-flash')
                                
                                if len(flights_data) == 1:
                                    prompt = f"Проаналізуй політ БПЛА. Метрики: {flights_data[0]['metrics']}. Напиши висновок українською (3 речення)."
                                else:
                                    prompt = f"Ти інженер. Порівняй два польоти БПЛА. Політ 1: {flights_data[0]['metrics']}. Політ 2: {flights_data[1]['metrics']}. Напиши стислий висновок українською (4-5 речень): який був агресивнішим, ефективнішим."
                                
                                try:
                                    st.success(model.generate_content(prompt).text)
                                except Exception as e:
                                    st.error(f"Помилка AI: {e}")

                except Exception as e:
                    st.error(f"🛑 Сталася помилка обробки: {e}")
else:
    st.info("👈 Завантажте один або два лог-файли у бічній панелі.")