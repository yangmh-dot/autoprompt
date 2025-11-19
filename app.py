import streamlit as st
import cv2
import base64
import tempfile
import os
from openai import OpenAI

# --- 設定頁面 ---
st.set_page_config(page_title="Sora 2 Video Remix Agent", layout="centered")

st.title("🎬 Sora 2 Video Remix Generator")
st.markdown("上傳影片，輸入修改需求，自動生成 9:16 英文 Prompt。")

# --- API Key 輸入 ---
api_key = st.text_input("請輸入 OpenAI API Key", type="password")

# --- 輔助函式：處理影片 ---
def extract_frames(video_path, num_frames=5):
    """從影片中提取關鍵影格"""
    video = cv2.VideoCapture(video_path)
    base64Frames = []
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    step = total_frames // num_frames
    
    for i in range(0, total_frames, step):
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, frame = video.read()
        if success:
            _, buffer = cv2.imencode(".jpg", frame)
            base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
            if len(base64Frames) >= num_frames:
                break
    video.release()
    return base64Frames

# --- 主程式邏輯 ---
if api_key:
    client = OpenAI(api_key=api_key)
    
    uploaded_file = st.file_uploader("上傳參考影片 (MP4/MOV)", type=["mp4", "mov"])
    modification = st.text_input("你想要如何修改內容？ (例如：把黑貓換成白貓，背景變成雪地)", value="Keep the same style, but change the subject to...")

    if uploaded_file and st.button("生成 Sora Prompt"):
        with st.spinner("正在分析影片並生成 Prompt..."):
            # 1. 儲存暫存影片
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_file.read())
            
            # 2. 提取影格
            frames = extract_frames(tfile.name)
            
            # 3. 構建 Prompt 給 GPT-4o
            prompt_messages = [
                {
                    "role": "system",
                    "content": """
                    You are an expert AI Video Prompt Engineer specializing in Sora 2.
                    Your task is to analyze video frames and generate a high-fidelity text-to-video prompt.
                    
                    Follow these rules strictly:
                    1. Analyze the camera movement, lighting, style, aesthetic, and action in the frames.
                    2. Apply the user's requested MODIFICATION to the content.
                    3. Output ONLY the prompt in English.
                    4. The output must be highly descriptive, photorealistic (unless specified otherwise), and detailed.
                    5. End the prompt with specific technical parameters: "--ar 9:16 --v 2"
                    """
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"User Modification Request: {modification}. \n\nAnalyze these frames and generate the Sora prompt:"},
                        *map(lambda x: {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{x}"}}, frames),
                    ],
                }
            ]

            # 4. 呼叫 API
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=prompt_messages,
                    max_tokens=500,
                )
                
                result_prompt = response.choices[0].message.content
                
                # 5. 顯示結果
                st.success("生成成功！")
                st.subheader("Sora 2 Prompt (English):")
                st.code(result_prompt, language="text")
                st.caption("請複製上方文字貼上至 Sora。")
                
            except Exception as e:
                st.error(f"發生錯誤: {e}")
            
            finally:
                os.remove(tfile.name) # 清理暫存檔

else:
    st.warning("請先輸入 API Key 才能開始使用。")