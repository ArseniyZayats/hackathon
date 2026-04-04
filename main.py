import streamlit as st
import tempfile
import os
import sys
import importlib
import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai

GEMINI_API_KEY = "AIzaSyA-uQrpw2pLwsAak6xp4FnCNOw_dnD8j6o"

# =============================================================================
# 1. БЕЗПЕЧНИЙ ІМПОРТ МОДУЛІВ КОМАНДИ
# =============================================================================
# Додаємо поточну папку в шляхи Python, щоб він бачив сусідні файли
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    # 1. Парсер
    # Якщо ви перейменували файл на log_parser.py, змініть тут:
    from parser import parse_bin 
    
    # 2. Аналітика
    from CoreAnalytics.coreAnalytics import merge_data, compute_all_metrics
    
    # 3. 3D Візуалізатор (безпечний імпорт папки, що починається з цифри)
    vis_module = importlib.import_module("Visual.visual")
    generate_3d_model = getattr(vis_module, "generate_3d_model")
    
except Exception as e:
    st.error(f"⚠️ Помилка імпорту модулів: {e}. Переконайтеся, що main.py лежить у головній папці.")

# =============================================================================
# 2. НАЛАШТУВАННЯ СТОРІНКИ
# =============================================================================
st.set_page_config(page_title="BEST UAV Analyzer", page_icon="🚁", layout="wide")
st.title("🚁 Система аналізу телеметрії БПЛА")
st.markdown("Інтерактивний дашборд для розшифровки 'чорних скриньок' Ardupilot.")

# =============================================================================
# 3. ФУНКЦІЇ ОБРОБКИ (КЕШОВАНІ)
# =============================================================================
@st.cache_data
def process_flight_data(file_path):
    """Розпаковує дані та рахує метрики через алгоритми команди"""
    # Парсимо 5 таблиць
    gps_df, imu_df, att_df, mode_df, curr_df = parse_bin(file_path)
    
    # Зливаємо GPS та IMU для базової аналітики
    df = merge_data(gps_df, imu_df)
    
    # Рахуємо всі метрики команди
    metrics = compute_all_metrics(df)
    
    return df, metrics, att_df, mode_df, curr_df

# =============================================================================
# 4. БІЧНА ПАНЕЛЬ
# =============================================================================
with st.sidebar:
    st.header("⚙️ Завантаження даних")
    uploaded_file = st.file_uploader("Завантажте лог-файл (.BIN)", type=['bin'])
    
    st.divider()
    st.info("💡 **Порада:** Цей дашборд автоматично розраховує кінематику, відображає напругу батареї та генерує 3D-анімацію місії.")

# =============================================================================
# 5. ГОЛОВНИЙ ІНТЕРФЕЙС
# =============================================================================
if uploaded_file is not None:
    # Створюємо тимчасову ізольовану директорію для роботи 3D-рушія
    with tempfile.TemporaryDirectory() as tmp_dir:
        
        # Зберігаємо завантажений файл із його оригінальною назвою
        temp_file_path = os.path.join(tmp_dir, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        with st.spinner("Аналіз лог-файлу, розрахунок метрик та генерація 3D-анімації..."):
            try:
                # --- ВИКЛИК АЛГОРИТМІВ ---
                df, metrics, att_df, mode_df, curr_df = process_flight_data(temp_file_path)
                
                # Виклик 3D-моделі (передаємо папку, як цього вимагає рушій команди)
                fig_3d = generate_3d_model(target_folder=tmp_dir)
                
                st.success(f"Файл {uploaded_file.name} успішно розпарсено та оброблено!")
                
                # --- БЛОК 1: МЕТРИКИ КІНЕМАТИКИ ---
                st.subheader("📊 Підсумкові показники місії")
                if metrics:
                    num_cols = min(len(metrics), 4)
                    cols = st.columns(max(num_cols, 1))
                    for i, (key, value) in enumerate(metrics.items()):
                        # Форматуємо значення залежно від типу
                        val_str = f"{value:.2f}" if isinstance(value, float) else str(value)
                        cols[i % num_cols].metric(label=str(key).replace("_", " ").title(), value=val_str)
                else:
                    st.warning("Алгоритм аналітики не повернув метрик.")
                
                st.divider()
                
                # --- БЛОК 2: ПРОСТОРОВА ТРАЄКТОРІЯ (2D + 3D) ---
                st.subheader("🗺️ Просторова траєкторія")
                tab_2d, tab_3d = st.tabs(["🗺️ 2D Карта (GPS WGS-84)", "🧊 3D Анімація польоту"])
                
                with tab_2d:
                    if 'lat_deg' in df.columns and 'lon_deg' in df.columns and not df.empty:
                        start_lat, start_lon = df['lat_deg'].iloc[0], df['lon_deg'].iloc[0]
                        fig_map = go.Figure(go.Scattermapbox(
                            lat=df['lat_deg'], lon=df['lon_deg'],
                            mode='lines', line=dict(width=4, color='#ff0055'), name='Маршрут'
                        ))
                        fig_map.update_layout(
                            mapbox_style="open-street-map",
                            mapbox=dict(center=dict(lat=start_lat, lon=start_lon), zoom=15),
                            margin=dict(l=0, r=0, b=0, t=0), height=500
                        )
                        st.plotly_chart(fig_map, use_container_width=True)
                    else:
                        st.error("GPS дані (lat_deg, lon_deg) відсутні у цьому файлі.")
                        
                with tab_3d:
                    if fig_3d:
                        st.plotly_chart(fig_3d, use_container_width=True)
                    else:
                        st.error("Не вдалося згенерувати 3D-анімацію. Перевірте логіку у 3Dvisual.py")
                
                st.divider()
                
                # --- БЛОК 3: БАТАРЕЯ ТА РЕЖИМИ ---
                col_bat, col_mode = st.columns([2, 1])
                
                with col_bat:
                    st.subheader("🔋 Аналіз живлення (Напруга)")
                    if not curr_df.empty and 'voltage' in curr_df.columns:
                        st.line_chart(curr_df, x='timestamp', y='voltage', color="#ff4b4b")
                    else:
                        st.info("Датчик живлення не був підключений.")
                        
                with col_mode:
                    st.subheader("🕹️ Режими польоту")
                    if not mode_df.empty and 'mode_name' in mode_df.columns:
                        st.dataframe(mode_df[['timestamp', 'mode_name']], use_container_width=True, hide_index=True)
                    else:
                        st.info("Дані про режими відсутні.")
                
                st.divider()
                
                # --- БЛОК 4: AI АСИСТЕНТ ---
                st.subheader("🤖 AI Висновок")
                if "ai_response" not in st.session_state:
                    st.session_state.ai_response = None
                    
                if st.button("Згенерувати автоматичний звіт", type="primary"):
                    with st.spinner("AI аналізує показники..."):
                        try:
                            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
                            model = genai.GenerativeModel('gemini-2.5-flash')
                              
                            metrics_text = ", ".join([f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()])
                            prompt = f"Ти досвідчений інженер телеметрії БПЛА. Проаналізуй ці метрики: {metrics_text}. Напиши короткий технічний висновок українською мовою (3-4 речення). Вкажи на можливі аномалії."
                                
                            response = model.generate_content(prompt)
                            if response.text:
                                st.session_state.ai_response = response.text
                        except Exception as e:
                            st.error(f"⚠️ Помилка AI: {e}")
                            
                if st.session_state.ai_response:
                    st.success(st.session_state.ai_response)

            except Exception as e:
                st.error(f"🛑 Сталася критична помилка під час обробки: {e}")

else:
    # Заглушка, коли файл ще не завантажено
    st.info("👈 Будь ласка, завантажте лог-файл (наприклад `00000001.BIN`) у бічній панелі, щоб розпочати роботу.")