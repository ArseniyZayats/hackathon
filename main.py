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
    # ЗМІНА: Додаємо accept_multiple_files=True
    uploaded_files = st.file_uploader("Завантажте лог-файли (.BIN)", type=['bin'], accept_multiple_files=True)
    
    st.divider()
    st.info("💡 Завантажте ДВА файли одночасно, щоб увімкнути режим порівняння!")

# =============================================================================
# 5. ГОЛОВНИЙ ІНТЕРФЕЙС
# =============================================================================
if uploaded_files:
    if len(uploaded_files) > 2:
        st.warning("⚠️ Для коректного порівняння завантажте не більше 2-х файлів.")
    else:
        with tempfile.TemporaryDirectory() as tmp_dir:
            
            # 1. Зберігаємо ВСІ завантажені файли у тимчасову папку
            for f in uploaded_files:
                path = os.path.join(tmp_dir, f.name)
                with open(path, "wb") as out:
                    out.write(f.getbuffer())
            
            with st.spinner("Аналіз місій та рендеринг..."):
                try:
                    # 2. Обробляємо кожен файл алгоритмами команди
                    flights_data = []
                    for f in uploaded_files:
                        path = os.path.join(tmp_dir, f.name)
                        df, metrics, att_df, mode_df, curr_df = process_flight_data(path)
                        flights_data.append({
                            "name": f.name, "df": df, "metrics": metrics, 
                            "mode_df": mode_df, "curr_df": curr_df
                        })
                    
                    # 3. Викликаємо 3D-модель (вона сама знайде всі файли в tmp_dir!)
                    fig_3d = generate_3d_model(target_folder=tmp_dir)
                    
                    st.success("Дані успішно оброблено!")
                    
                    # --- БЛОК 1: МЕТРИКИ (ОДИН ФАЙЛ АБО ПОРІВНЯННЯ) ---
                    if len(flights_data) == 1:
                        st.subheader(f"📊 Показники місії: {flights_data[0]['name']}")
                        m1 = flights_data[0]['metrics']
                        cols = st.columns(4)
                        for i, (k, v) in enumerate(m1.items()):
                            cols[i % 4].metric(k.replace("_", " ").title(), f"{v:.2f}" if isinstance(v, float) else str(v))
                    
                    elif len(flights_data) == 2:
                        st.subheader("⚖️ Порівняльний аналіз місій")
                        m1 = flights_data[0]['metrics']
                        m2 = flights_data[1]['metrics']
                        name1, name2 = flights_data[0]['name'], flights_data[1]['name']
                        
                        st.markdown(f"**Базовий:** `{name1}` | **Порівнюється з:** `{name2}`")
                        
                        cols = st.columns(4)
                        # Виводимо метрики з автоматичним розрахунком різниці (delta)
                        for i, key in enumerate(m1.keys()):
                            if key in m2:
                                val1, val2 = m1[key], m2[key]
                                if isinstance(val1, (int, float)):
                                    diff = val2 - val1 # Різниця (наскільки 2-й політ більший/менший за 1-й)
                                    cols[i % 4].metric(
                                        label=key.replace("_", " ").title(), 
                                        value=f"{val2:.2f}", 
                                        delta=f"{diff:.2f} (відносно {name1})",
                                        delta_color="normal"
                                    )
                    st.divider()
                    
                    # --- БЛОК 2: 3D АНІМАЦІЯ ---
                    st.subheader("🗺️ Просторова траєкторія")
                    if fig_3d:
                        st.plotly_chart(fig_3d, use_container_width=True)
                        if len(flights_data) == 2:
                            st.caption("✨ Натисніть кнопку Play під графіком, щоб побачити одночасну симуляцію двох польотів!")
                    else:
                        st.error("Не вдалося згенерувати 3D-анімацію.")
                        
                    st.divider()
                    
                    # --- БЛОК 3: AI АСИСТЕНТ (АДАПТОВАНИЙ ДЛЯ ПОРІВНЯННЯ) ---
                    st.subheader("🤖 AI Аналітик")
                    if st.button("Згенерувати автоматичний звіт", type="primary"):
                        with st.spinner("AI працює..."):
                            api_key = st.secrets.get("GEMINI_API_KEY", "")
                            if not api_key:
                                st.error("⚠️ API-ключ не знайдено.")
                            else:
                                genai.configure(api_key=api_key)
                                model = genai.GenerativeModel('gemini-2.5-flash')
                                
                                if len(flights_data) == 1:
                                    prompt = f"Проаналізуй політ. Метрики: {flights_data[0]['metrics']}. Напиши висновок українською (3 речення)."
                                else:
                                    prompt = f"Ти інженер. Порівняй два польоти БПЛА. Політ 1: {flights_data[0]['metrics']}. Політ 2: {flights_data[1]['metrics']}. Напиши стислий висновок українською (4 речення): який був агресивнішим, ефективнішим тощо."
                                
                                try:
                                    st.success(model.generate_content(prompt).text)
                                except Exception as e:
                                    st.error(f"Помилка AI: {e}")

                except Exception as e:
                    st.error(f"🛑 Сталася помилка обробки: {e}")
else:
    st.info("👈 Завантажте один або два лог-файли у бічній панелі.")