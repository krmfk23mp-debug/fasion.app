import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import requests
import pandas as pd
import csv
import os
from datetime import datetime
import numpy as np
import random
import uuid

# 設定
CLASSES = ['ブラウス', 'トップス', 'Tシャツ', 'タンクトップ', 'ワンピース', 'スカート', 'ショートパンツ', 'パンツ', 'アウター', 'ブラ', 'スーツ', 'バッグ', 'シューズ']
NUM_CLASSES = len(CLASSES)

# モデルパス
MODEL_PATH = 'my_fashion_model.pth'

# OpenWeatherMap APIキー
API_KEY = "b4cb05f139a5da5e3e434764df89de3d"

# ページ設定
st.set_page_config(page_title="Coordinate AI", page_icon="👚", layout="wide")

# CSS設定
st.markdown("""
    <style>
    /* 全体の背景色を変更 */
    .stApp {
        background-color: #ffffff !important;
    }

    /* サイドバーの背景色を変更 */
    section[data-testid="stSidebar"] {
        background-color: #FFF0F5 !important; 
    }
    
    /* フォント変更 */
    .stApp {
        font-family: 'Helvetica Neue', 'Helvetica', 'Arial', 'Hiragino Kaku Gothic ProN', 'Hiragino Sans', 'Meiryo', sans-serif !important;
        color: #424242 !important;
    }

    /* ポイント色 #EB5EA0 */
    div[role="radiogroup"] {
        color: #EB5EA0 !important;
    }
    
    /* ボタンのデザイン */
    div.stButton > button {
        background-color: #FFF0F5 !important;
        color: #EB5EA0 !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: normal !important;
        padding: 10px 20px !important;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #EB5EA0 !important;
        color: #FFFFFF !important; /* 全角＃を半角#に修正 */
    }
    div.stButton > button:hover p {
        color: #ffffff !important;
    }

    /* ★ボタンのグレーアウト */
    div.stButton > button:disabled {
        background-color: #f0f2f6 !important; 
        color: #a3a8b4 !important; 
        border: 1px solid #dce0e6 !important; 
        cursor: not-allowed !important;
        border-radius: 8px !important;
    }
    
    div.stButton > button:disabled:hover {
        background-color: #f0f2f6 !important;
        color: #a3a8b4 !important;
    }
    div.stButton > button:disabled:hover p {
        color: #a3a8b4 !important;
    }

    /* （st.divider）の調整（使用しないため無視されますが念のため残します） */
    hr {
        margin-top: 5px !important;     
        margin-bottom: 5px !important; 
        border-color: #FFFFFF !important; 
    }
    
    /* 背景色変更*/
    .stAlert {
        background-color: #FFF0F5 !important; 
        border: 1px solid #F8CEDE !important;
    }
    .stAlert > div {
        color: #424242 !important;
    }
    
    img {
        border-radius: 4px; 
    }
    
    hr {
        border-color: #eee !important;
    }
    </style>
    """, unsafe_allow_html=True)

# モデル読込み
@st.cache_resource
def load_model():
    model = models.mobilenet_v3_large(pretrained=False)
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, NUM_CLASSES)
    device = torch.device('cpu')
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        return model
    except FileNotFoundError:
        return None

# 画像処理
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# 天気取得
def get_weather(city_name):
    # 1. 現在の天気を取得
    current_url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name}&units=metric&lang=ja&appid={API_KEY}"
    # 2. 天気予報を取得（最高・最低気温用）
    forecast_url = f"http://api.openweathermap.org/data/2.5/forecast?q={city_name}&units=metric&lang=ja&appid={API_KEY}"
    
    try:
        # 現在の天気を取得
        current_res = requests.get(current_url).json()
        if current_res["cod"] != 200:
            return None, None, None, None, None, None
        
        temp = current_res["main"]["temp"]
        desc = current_res["weather"][0]["description"]
        icon = current_res["weather"][0]["icon"]
        name = current_res["name"]
        
        # 予報データを取得して、今日の最高・最低気温を探す
        forecast_res = requests.get(forecast_url).json()
        if forecast_res["cod"] == "200":
            # 今日（24時間以内）のデータを抽出して、その中からMax/Minを探す
            temps = []
            for item in forecast_res['list'][:8]: # 直近8個（24時間分）のデータを見る
                temps.append(item['main']['temp_max'])
                temps.append(item['main']['temp_min'])
            
            # 現在気温も含めて比較する
            temps.append(temp)
            
            temp_max = max(temps)
            temp_min = min(temps)
        else:
            # 予報が取れなかった場合は現在の気温を入れる（フォールバック）
            temp_max = temp
            temp_min = temp
        
        return temp, temp_min, temp_max, desc, icon, name
    except:
        return None, None, None, None, None, None

# アドバイス生成
def get_fashion_advice(item_name, temp, weather_desc):
    advice = ""
    if temp >= 30:
        advice += "☀️ かなり暑いです！熱中症対策を忘れずに、涼しい素材を選びましょう。\n"
        if item_name in ['outer', 'suit']:
            advice += "💦 そのアイテムは暑すぎるかもしれません。手持ちにした方が良いかも？\n"
    elif temp >= 25:
        advice += "🏖️ 暑い日です。半袖やノースリーブが快適です。\n"
    elif temp >= 20:
        advice += "🍀 過ごしやすい気温です！ファッションを一番楽しめます✨\n"
    elif temp >= 15:
        advice += "☁️ 少し肌寒いです。カーディガンやジャケットがあると安心です。\n"
        if item_name in ['t-shirt', 'tank', 'top']:
            advice += "🌨️ そのままだと寒いかも。重ね着を検討しましょう。\n"
    elif temp >= 10:
        advice += "❄️ 寒いです！コートやジャケットが必要です。\n"
        if item_name in ['skirt', 'shorts', 'dress']:
            advice += "🧣 足元が冷えないように、タイツやブーツを合わせましょう。\n"
    else:
        advice += "真冬の寒さです！❄️マフラーや手袋でしっかり防寒してください。\n"

    if "雨" in weather_desc:
        advice += "雨予報です☔ ️濡れても大丈夫な靴や、防水スプレーを使いましょう。\n"
        if item_name == 'shoes':
            advice += "👠 白い靴や布製の靴は避けたほうが無難かもしれません。\n"
    
    return advice

# 評価保存
def save_feedback(predicted_label, user_feedback, comment):
    file_path = 'feedback_log.csv'
    if not os.path.exists(file_path):
        with open(file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["日時", "AIの判定", "ユーザー評価", "コメント"])
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), predicted_label, user_feedback, comment])

# 画像保存
def save_image_for_retraining(image, true_label):
    save_dir = os.path.join("retrain_data", true_label)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = f"{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}.jpg"
    save_path = os.path.join(save_dir, file_name)
    image.save(save_path)
    return save_path

# ランダム画像取得
def get_random_image(folder_name):
    base_dir = "suggest_images"
    target_dir = os.path.join(base_dir, folder_name)
    if not os.path.exists(target_dir):
        return None
    files = [f for f in os.listdir(target_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not files:
        return None
    chosen_file = random.choice(files)
    return os.path.join(target_dir, chosen_file)

# 明るさ判定
def get_color_tone(image):
    small_img = image.resize((50, 50))
    gray_img = small_img.convert('L')
    img_array = np.array(gray_img)
    avg_brightness = img_array.mean()
    if avg_brightness < 100:
        return "dark"
    else:
        return "light"

# 空白を入れる関数
def add_space(px):
    st.markdown(f'<div style="height: {px}px;"></div>', unsafe_allow_html=True)

# ボックス（背景）の色を統一
def show_pink_box(text, icon=None):
    icon_html = f'<div style="font-size: 20px; margin-right: 15px;">{icon}</div>' if icon else ""
    
    style = "background-color: #FFF0F5; border: none; border-radius: 8px; padding: 15px; color: #424242; display: flex; align_items: flex-start; justify-content: flex-start; font-weight: normal; font-size: 14px; line-height: 1.6; margin-bottom: 15px;"
    
    html_content = f'<div style="{style}">{icon_html}<div style="margin: 0; white-space: pre-wrap;">{text}</div></div>'
    
    st.markdown(html_content, unsafe_allow_html=True)

# 提案ロジック
def show_coordinate_suggestions(label, image):
    tone = get_color_tone(image)
    
    st.markdown("""
    <h4 style='text-align: left; color: #424242; font-weight: bold; font-size: 24px; letter-spacing: 0.1em; margin-bottom: 10px; margin-top: 20px;'>
        スタイリングの提案
    </h4>
    """, unsafe_allow_html=True)

    if tone == "dark":
        suggest_tone = "light"
        msg = "シックで落ち着いた色味ですね！\n<b>明るいアイテム</b>を合わせて軽さを出しましょう。"
        show_pink_box(msg, icon="🌑")
    else:
        suggest_tone = "dark"
        msg = "明るく爽やかな色味ですね！\n<b>引き締めカラー（暗め）</b>を合わせるとバランスが良いです。"
        show_pink_box(msg, icon="🍒")

    st.write("▼ 相性がいいアイテム✨")
    
    suggestion_plan = []
    if label in ['Tシャツ', 'トップス', 'ブラウス', 'タンクトップ', 'ブラ']:
        suggestion_plan = [(f"bottoms_{suggest_tone}", "相性の良いボトムス"), (f"outer_{suggest_tone}", "羽織るならこちら")]
    elif label in ['パンツ', 'スカート', 'ショートパンツ']:
        suggestion_plan = [(f"tops_{suggest_tone}", "バランスの良いトップス"), (f"shoes_{suggest_tone}", "足元も色味を合わせて")]
    elif label in ['アウター', 'スーツ']:
        suggestion_plan = [(f"tops_{suggest_tone}", "インナーの提案"), (f"bottoms_{suggest_tone}", "ボトムスの提案")]
    elif label == 'ワンピース':
        suggestion_plan = [(f"bag_{suggest_tone}", "小物をアクセントに"), (f"shoes_{suggest_tone}", "足元の合わせ")]
    else:
        suggestion_plan = [(f"tops_{suggest_tone}", "トップスの提案"), (f"bottoms_{suggest_tone}", "ボトムスの提案")]

    c1, c2 = st.columns(2)
    columns = [c1, c2]

    for i, (folder_name, caption) in enumerate(suggestion_plan):
        img_path = get_random_image(folder_name)
        with columns[i]:
            if img_path:
                st.image(img_path, caption=caption)
            else:
                st.warning(f"画像準備中... ({folder_name})")

# リボンの画像
def render_sidebar_header(text):
    c1, c2 = st.sidebar.columns([0.1, 0.9])
    ribbon_path = "ribbon.png"
    
    with c1:
        if os.path.exists(ribbon_path):
            st.image(ribbon_path, width=40)
            
    with c2:
        st.markdown(f"<h3 style='margin: 0; padding-top: 0px; font-size: 18px; color: #424242;'>{text}</h3>", unsafe_allow_html=True)

# UI構築
st.markdown("""
<h4 style='text-align: left; color: #424242; font-weight: bold; font-size: 30px; letter-spacing: 0.1em; margin-bottom: 10px;'>
    ˙⟡ コーディネート提案AI ⟡⁺.
</h4>
""", unsafe_allow_html=True)

st.write("◎ AIが服を判定し、おすすめコーデを提案します！✨")
st.caption(" カメラで撮影するか、画像をアップロードしてください。")

# 変数の初期化
image_content = None

# サイドバー設定
with st.sidebar:
    
    # 画像アップロード
    render_sidebar_header("画像アップロード")
    input_method = st.radio("入力方法を選んでください", ("ライブラリから選択", "カメラで撮影"))

    if input_method == "ライブラリから選択":
        uploaded_file = st.file_uploader("フォルダから画像を選択", type=["jpg", "png"])
        if uploaded_file is not None:
            image_content = uploaded_file
    else:
        camera_file = st.camera_input("カメラで撮影")
        if camera_file is not None:
            image_content = camera_file
    
    add_space(10) 

    # 現在地
    st.write("")
    render_sidebar_header("現在地")
    city = st.text_input("都市名", "Tokyo", label_visibility="collapsed")
    
    add_space(30) 

    st.caption(f"使用モデル: {MODEL_PATH}")

# メイン処理
if image_content is not None:
    analyze_btn = st.button('☆ 判定＆アドバイスを見る', key='btn_active', use_container_width=True)
    col1, col2 = st.columns([1, 1])
    
    with col1:
        image = Image.open(image_content).convert('RGB')
        st.image(image, use_container_width=True)
        
    if analyze_btn:
        model = load_model()
        if model is None:
            st.error("モデルファイルが見つかりません。")
        else:
            with st.spinner('Thinking...'):
                input_tensor = transform_image(image)
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, predicted_idx = torch.max(probabilities, 1)
                
                predicted_label = CLASSES[predicted_idx.item()]
                confidence_score = confidence.item() * 100
                
            temp, temp_min, temp_max, weather_desc, icon_code, real_name = get_weather(city)
            display_city_name = real_name if real_name else city

            with col2:
                # 天気情報
                st.markdown(f"""
                <h3 style='margin: 0 0 10px 0; padding: 0; font-size: 20px; color: #424242; font-weight: bold;'>
                    {display_city_name} の天気
                </h3>
                """, unsafe_allow_html=True)
                
                if temp is not None:
                    w_col1, w_col2, w_col3 = st.columns([1.5, 1.2, 1.2])
                    
                    with w_col1: 
                        st.markdown(f'<img src="http://openweathermap.org/img/wn/{icon_code}@4x.png" style="width: 120px;">', unsafe_allow_html=True)
                    
                    with w_col2:
                        st.markdown(f"""
                        <div style="display: flex; flex-direction: column; justify-content: center; height: 100%; padding-top: 20px;">
                            <span style="font-size: 14px; color: #888;">現在の気温</span>
                            <span style="font-size: 36px; font-weight: midium; color: #424242;">{int(temp)}℃</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with w_col3: 
                        st.markdown(f"""
                        <div style="font-size: 13px; line-height: 1.5; color: #666; margin-top: 35px;">
                            <span style="color: #DE1738; font-weight: midium;">最 高： {int(temp_max)}℃</span><br>
                            <span style="color: #476FBF; font-weight: midium;">最 低：  {int(temp_min)}℃</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("天気情報取得失敗")
                    st.metric(label="晴れ", value="20℃")

                add_space(30) # dividerを空白に置換

                # 判定結果
                st.markdown(f"""
                <div style="text-align: left; margin-bottom: 20px;">
                    <p style="margin:0; color: #888; font-size: 12px; letter-spacing: 0.1em;">RESULT</p>
                    <h2 style="margin:10px 0; font-size: 32px; color: #EB5EA0; letter-spacing: 1px; font-weight:bold;">{predicted_label.upper()}</h2>
                    <p style="margin:0; color: #888; font-size: 12px;">Confidence: {int(confidence_score)}%</p>
                </div>
                """, unsafe_allow_html=True)

            if analyze_btn:
                # アドバイス
                add_space(30) # dividerを空白に置換
                st.markdown("""
                <h4 style='text-align: left; color: #424242; font-weight: bold; font-size: 24px; letter-spacing: 0.1em; margin-bottom: 10px; margin-top: 20px;'>
                    アドバイス
                </h4>
                """, unsafe_allow_html=True)
                advice_text = get_fashion_advice(predicted_label, temp, weather_desc)
                show_pink_box(advice_text)

                # グラフ
                with st.expander("📊 分析データ"):
                    probs_np = probabilities[0].numpy() * 100
                    df = pd.DataFrame({'アイテム': CLASSES, 'スコア': probs_np})
                    st.bar_chart(df.set_index('アイテム'))

                # 提案機能
                add_space(30) # dividerを空白に置換
                try:
                    show_coordinate_suggestions(predicted_label, image)
                except Exception as e:
                    st.error(f"Image Error: {e}")
                    st.info("※フォルダ名やファイル構成を確認してください")

                # 評価フォーム
                add_space(30) # dividerを空白に置換
                st.markdown("""
                <h4 style='text-align: left; color: #424242; font-weight: bold; font-size: 24px; letter-spacing: 0.1em; margin-bottom: 10px; margin-top: 20px;'>
                    フィードバック
                </h4>
                """, unsafe_allow_html=True)
                with st.form(key='feedback_form'):
                    st.caption("AIの判定は合っていましたか？")
                    feedback_options = ("合っていた！🙆‍♀️", "違っていた...🙅‍♀️")
                    user_feedback = st.radio("評価", feedback_options, horizontal=True, label_visibility="collapsed")

                    correct_label = st.selectbox(
                        "もし違っていたら、正解を教えてください▼", 
                        options=["(選択)"] + CLASSES,
                        index=0
                    )

                    comment = st.text_input("コメント（任意）")
                    submit_btn = st.form_submit_button("送信")
                    
                    if submit_btn:
                        save_feedback(predicted_label, user_feedback, comment)
                        
                        if user_feedback == "違っていた...🙅‍♀️" and correct_label != "(選択)":
                            save_path = save_image_for_retraining(image, correct_label)
                            st.success(f"「{correct_label}」として保存しました。ありがとうございます！")
                        elif user_feedback == "合っていた！🙆‍♀️":
                            save_image_for_retraining(image, predicted_label)
                            st.success("フィードバックありがとうございます！")
                        else:
                            st.success("フィードバックを受け付けました。")

else:
    # 画像がない時の案内
    st.markdown("""
    <div style="
        background-color: #FFF0F5; 
        border: none;
        border-radius: 8px; 
        padding: 10px 15px; 
        color: #424242; 
        display: flex;
        align_items: center;
        justify-content: center;
        font-weight: normal; 
        font-size: 14px;
        margin-bottom: 20px;
    ">
        <span style="font-size: 20px; margin-right: 10px;">⬅️</span>
        <span>左のメニューから画像をアップロードしてください</span>
    </div>
    """, unsafe_allow_html=True)

    st.button('☆ 判定＆アドバイスを見る', disabled=True, key='btn_disabled', use_container_width=True)
