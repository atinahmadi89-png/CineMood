import streamlit as st
import pandas as pd
import numpy as np
import joblib
from deepface import DeepFace
from PIL import Image


st.set_page_config(
page_title="سینمود",
page_icon="🎥",
layout="wide"
)

st.title("🎥سینمود")
st.subheader("فیلم متناسب با حالو هوای تو")

st.write(
" به سینمود خوش اومدی"
" فیلم مورد علاقتو انتخاب کن"
" تا پنج فیلم مشابه بهت پیشنهاد بدیم"
)

st.divider()


@st.cache_data
def load_movies():
    return pd.read_json("movies.json", encoding="utf-8")

@st.cache_data
def load_similarity():
    return np.load("similarity_matrix.npy")

@st.cache_resource
def load_sentiment_model():
    return joblib.load("sentiment_model.joblib")

sentiment_model = load_sentiment_model()

data = load_movies()
similarity_matrix = load_similarity()


if similarity_matrix.shape != (len(data), len(data)):
    st.error("تعداد فیلم ها با ابعاد ماتریس شباهت مطابقت ندارد")
    st.stop()


st.header("پیشنهاد فیلم های مشابه")

movie_options = {}

for index, row in data.iterrows():
    movie_name = str(row["title_fa"])
    original_name = str(row["title"])

    movie_options[f"{movie_name} ({original_name})"] = index

selected_movie = st.selectbox(
"فیلم موردعلاقتو انتخاب کن",
list(movie_options.keys())
)


if st.button("پیدا کردن فیلم‌های مشابه"):
    movie_index = movie_options[selected_movie]
    similarities = similarity_matrix[movie_index]
    similar_movies_indices = similarities.argsort()[::-1]
    similar_movies_indices = [
        i for i in similar_movies_indices if i != movie_index
    ][:5]
    recommended_movies = data.iloc[similar_movies_indices]
    st.subheader("فیلم‌های پیشنهادی برای تو")
    for _, movie in recommended_movies.iterrows():
        st.markdown(f"### 🎥 {movie['title_fa']}")
        st.write(f"نام انگلیسی: {movie['title']}")
        st.write(f"سال ساخت: {movie['year']}")
        st.write(f"⭐ امتیاز: {movie['rating']}")
        st.divider()


st.header("🎬 تحلیل احساسات نقد فیلم")

review = st.text_area(
    "نظرت درباره فیلم رو بنویس:"
)

if st.button("تحلیل نقد"):
    if review.strip():
        prediction = sentiment_model.predict([review])[0]
        probabilities = sentiment_model.predict_proba([review])[0]
        confidence = max(probabilities) * 100

        if prediction == "positive":
            st.success("احساس نقد: مثبت 😊")
        else:
            st.error("احساس نقد: منفی 😞")

        st.write(f"اطمینان مدل: {confidence:.2f}%")
    else:
        st.warning("لطفاً ابتدا نقد فیلم رو بنویس.")
st.caption("سینمود| دنیای فیلم ها متناسب با احساس تو")

st.divider()

st.header("🎭 تشخیص احساس چهره")
st.write(
    "یک عکس از چهره‌ات آپلود کن تا احساس تو رو تشخیص بدیم "
    "و فیلم‌های متناسب با حالت رو پیشنهاد کنیم!"
)

uploaded_image = st.file_uploader(
    "عکس چهره‌ات رو انتخاب کن",
    type=["jpg", "jpeg", "png"]
)

emotion_names = {
    "happy": "شاد 😊",
    "sad": "غمگین 😢",
    "angry": "عصبانی 😠",
    "surprise": "متعجب 😲",
    "neutral": "خنثی 😐"
}


def parse_moods(value):
    """تبدیل فهرست احساسات هر فیلم به لیست قابل جست‌وجو."""
    if isinstance(value, list):
        items = value
    else:
        text = str(value).strip().strip("[]()")
        items = text.split(",")

    return [
        str(item).strip().strip("'\"").lower()
        for item in items
        if str(item).strip()
    ]


if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="عکس انتخاب‌شده", width=300)

    if st.button("تشخیص احساس من 🎯"):
        try:
            with st.spinner("داریم احساس چهره‌ات رو بررسی می‌کنیم..."):
                image_array = np.array(image)

                result = DeepFace.analyze(
                    img_path=image_array,
                    actions=["emotion"],
                    enforce_detection=True
                )

                if isinstance(result, list):
                    result = result[0]

                detected_emotion = result["dominant_emotion"]

            if detected_emotion not in emotion_names:
                st.warning("احساس تشخیص‌داده‌شده پشتیبانی نمی‌شود.")
            else:
                st.success(
                    f"احساس تشخیص‌داده‌شده: "
                    f"{emotion_names[detected_emotion]}"
                )

                matching_movies = data[
                    data["moods"].apply(
                        lambda moods: detected_emotion
                        in parse_moods(moods)
                    )
                ]

                matching_movies = matching_movies.sort_values(
                    by="rating",
                    ascending=False
                ).head(5)

                st.subheader("🎬 فیلم‌های پیشنهادی برای تو")

                if matching_movies.empty:
                    st.info(
                        "فیلمی با این احساس در دیتاست پیدا نشد."
                    )
                else:
                    for _, movie in matching_movies.iterrows():
                        st.markdown(
                            f"### 🎥 {movie['title_fa']}"
                        )
                        st.write(
                            f"نام انگلیسی: {movie['title']}"
                        )
                        st.write(
                            f"سال ساخت: {movie['year']}"
                        )
                        st.write(
                            f"⭐ امتیاز: {movie['rating']}"
                        )
                        st.divider()

        except Exception as e:
            st.error(
                "تشخیص احساس انجام نشد. لطفاً یک عکس واضح "
                "که چهره در آن مشخص است انتخاب کن."
            )
            st.caption(f"جزئیات خطا: {e}")