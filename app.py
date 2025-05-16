import streamlit as st
import cv2
import tempfile
import os
from detection import detect_blue_cards

THRESHOLDS = list(range(250, 30, -10))


st.title("Детекция карточек и фигур на них на изображении")
uploaded_file = st.file_uploader(
    "Загрузите изображение", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    original_img = cv2.imread(tmp_path)
    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    with st.spinner("Ищем синие карты..."):
        processed_img = detect_blue_cards(tmp_path)

    os.unlink(tmp_path)

    if processed_img is not None:
        processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.image(original_img_rgb, caption="Исходное изображение",
                     use_container_width=True)
        with col2:
            st.image(processed_img_rgb, caption="Результат обработки",
                     use_container_width=True)
    else:
        st.error("Не удалось обработать изображение")
else:
    st.info("Пожалуйста, загрузите изображение в формате JPG, PNG или JPEG")
