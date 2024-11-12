import streamlit as st
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import joblib
import time
import os

# 모델 및 토크나이저 로드 (질문 분류)
question_model_path = '/Users/jeon-yewon/Desktop/project/숙소문의모델'
tokenizer = BertTokenizer.from_pretrained(question_model_path)
question_model = BertForSequenceClassification.from_pretrained(question_model_path)

# 모델 및 토크나이저 로드 (감정 분류)
emotion_model_path = '/Users/jeon-yewon/Desktop/project/감정분류모델'
emotion_model = BertForSequenceClassification.from_pretrained(emotion_model_path)

# 라벨 인코더 로드
label_encoder = joblib.load('/Users/jeon-yewon/Desktop/project/숙소문의모델/label_encoder.pkl')

# CSV 파일 경로
question_csv_path = '/Users/jeon-yewon/Desktop/project/data/숙소문의.csv'
emotion_csv_path = '/Users/jeon-yewon/Desktop/project/data/감정분류.csv'
guest_reviews_csv_path = '/Users/jeon-yewon/Desktop/project/data/guest_reviews.csv'

# 카테고리 배열 (질문 분류용)
categories = label_encoder.classes_

# 질문 분류 함수
def classify_question(question):
    inputs = tokenizer(question, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = question_model(**inputs)
        logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    return predicted_label

# 감정 예측 함수
def predict_emotion(review):
    inputs = tokenizer(review, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = emotion_model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()

    label_map = {0: '감사한', 1: '만족한', 2: '힐링되는', 3: '신나는', 4: '경험적인', 5: '불만족', 6: '양가감정'}
    return label_map[predicted_class]

# 호스트 설문 데이터를 CSV 파일에 저장하는 함수 (질문 분류)
def save_host_data(host_name, property_name, answers):
    data = pd.read_csv(question_csv_path)
    
    rows_to_add = []
    for category, answer in answers.items():
        question = f"{category} 관련 질문입니다."
        new_row = {'호스트명': host_name, '숙소명': property_name, '카테고리': category, '질문': question, '답변': answer}
        rows_to_add.append(new_row)

    data = pd.concat([data, pd.DataFrame(rows_to_add)], ignore_index=True)
    data.to_csv(question_csv_path, index=False)

# 호스트 설문 데이터를 CSV 파일에 저장하는 함수 (감정 분류)
def save_guest_review(host_name, property_name, review, emotion):
    review_data = pd.DataFrame({'호스트명': [host_name], '숙소명': [property_name], '리뷰': [review], '감정': [emotion]})
    review_data.to_csv(guest_reviews_csv_path, mode='a', header=False, index=False)

# 호스트 데이터 기반으로 게스트 질문에 대한 답변 반환
def get_answer(host_name, property_name, guest_question):
    data = pd.read_csv(question_csv_path)
    host_data = data[(data['호스트명'] == host_name) & (data['숙소명'] == property_name)]
    
    if host_data.empty:
        return "해당 호스트 또는 숙소 정보를 찾을 수 없습니다."
    
    category = classify_question(guest_question)
    matched_row = host_data[host_data['카테고리'] == category]
    
    if not matched_row.empty:
        return matched_row['답변'].values[0]
    
    return "해당 질문에 대한 답변을 찾을 수 없습니다."

# 챗봇 페이지 (게스트용)
def chatbot_page():
    st.title("실시간 챗봇 🏘️🗯️")
    host_name = st.text_input("⭐️ 호스트명")
    property_name = st.text_input("⭐️ 숙소명")
    
    if host_name and property_name:
        st.write(f"안녕하세요, {property_name}의 챗봇입니다!")
        guest_query = st.text_input("메시지를 입력하세요...")
        
        if guest_query:
            response = get_answer(host_name, property_name, guest_query)
            st.write(f"챗봇: {response}")

# 호스트 설문 페이지 (질문 분류)
def host_survey_page_question():
    st.title("📝 호스트 설문 페이지 (숙소 문의)")
    host_name = st.text_input("호스트명을 입력하세요")
    property_name = st.text_input("숙소명을 입력하세요")
    
    answers = {}
    for category in categories:
        answers[category] = st.text_input(f"{category} 관련 답변을 입력하세요")
    
    if st.button('저장'):
        save_host_data(host_name, property_name, answers)
        st.success("설문이 저장되었습니다.")

# 감정 분류 설문 페이지
def host_survey_page_emotion():
    st.title("📝 호스트 설문 페이지 (감정 분류)")
    host_name = st.text_input("호스트명을 입력하세요")
    property_name = st.text_input("숙소명을 입력하세요")
    
    answers = {}
    emotions = ['감사한', '만족한', '힐링되는', '신나는', '경험적인', '불만족', '양가감정']
    for emotion in emotions:
        answers[emotion] = st.text_input(f"'{emotion}' 감정을 느낀 게스트에게 제공할 답변을 입력하세요")
    
    if st.button('저장'):
        # 사용자가 입력한 데이터를 바로 저장
        data = pd.DataFrame({
            '호스트명': [host_name],
            '숙소명': [property_name],
            **answers  # 입력한 감정 관련 데이터를 저장
        })
        
        # 이미 지정한 경로 사용하여 CSV 파일에 저장
        file_path = emotion_csv_path
        if not os.path.exists(file_path):
            data.to_csv(file_path, mode='w', index=False)
        else:
            data.to_csv(file_path, mode='a', header=False, index=False)

        st.success("감정 분류 설문이 성공적으로 저장되었습니다.")



# 감정 분류 페이지 (게스트용)
def emotion_classification_page():
    st.title("감정 분류 📊")
    host_name = st.text_input("⭐️ 호스트명")
    property_name = st.text_input("⭐️ 숙소명")
    review = st.text_area("리뷰 작성")

    if st.button('감정 분류'):
        emotion = predict_emotion(review)
        st.write(f"예측된 감정: {emotion}")
        
        save_guest_review(host_name, property_name, review, emotion)
        st.success(f"리뷰가 성공적으로 저장되었습니다. 예측된 감정: {emotion}")

        # 호스트 답변 출력 (감정 분류에서는 '호스트' 컬럼 사용)
        data = pd.read_csv(emotion_csv_path)
        host_data = data[(data['호스트'] == host_name) & (data['숙소명'] == property_name)]
        if not host_data.empty:
            st.write(f"호스트의 {emotion} 답변: {host_data[emotion].values[0]}")
        else:
            st.write("해당 호스트에 대한 정보가 없습니다.")


# 프롤로그 페이지
def prologue_page():
    st.markdown("<h1 style='text-align: center; color: #FF5A5F;'>에어비앤비 챗봇 시스템에 오신 것을</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: #FF5A5F;'>환영합니다 🙌</h1>", unsafe_allow_html=True)
    
    time.sleep(1)
    st.write("데이터 분석가 부트캠프 17회차 최종 프로젝트")
    time.sleep(1)
    st.write("by. 김솔, 이예진, 전예원")
    
    if st.button("서비스 이용하기"):
        st.session_state.page = "main"

# 메인 페이지
def main_page():
    st.sidebar.title("에어비앤비 챗봇 🏘️")
    
    service = st.sidebar.radio("1️⃣ 서비스 선택", ["감정 분류", "숙소 문의"])
    
    if service == "숙소 문의":
        user_type = st.sidebar.radio("2️⃣ 사용자 선택", ["호스트", "게스트"])
        if user_type == "게스트":
            chatbot_page()  # 챗봇
        elif user_type == "호스트":
            host_survey_page_question()  # 숙소 문의 설문
    
    elif service == "감정 분류":
        user_type = st.sidebar.radio("2️⃣ 사용자 선택", ["호스트", "게스트"])
        if user_type == "게스트":
            emotion_classification_page()  # 감정 분류 (게스트용)
        elif user_type == "호스트":
            host_survey_page_emotion()  # 감정 분류 설문 (호스트용)

# 메인 실행 함수
def main():
    if "page" not in st.session_state:
        prologue_page()  # 첫 화면: 프롤로그 페이지
    else:
        main_page()  # 메인 페이지

if __name__ == "__main__":
    main()
