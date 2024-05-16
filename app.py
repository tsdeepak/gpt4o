import streamlit as st
import openai
from openai import OpenAI
import os
import base64
import cv2
from moviepy.editor import VideoFileClip
from dotenv import load_dotenv

load_dotenv()


# documentation
# 1. Cookbook:  https://cookbook.openai.com/examples/gpt4o/introduction_to_gpt4o
# 2. Configure your Project and Orgs to limit/allow Models:  https://platform.openai.com/settings/organization/general
# 3. Watch your Billing!  https://platform.openai.com/settings/organization/billing/overview


# Set API key and organization ID from environment variables
openai.api_key = os.getenv('OPENAI_API_KEY')
openai.organization = os.getenv('OPENAI_ORG_ID')
client = OpenAI(api_key= os.getenv('OPENAI_API_KEY'), organization=os.getenv('OPENAI_ORG_ID'))

# Define the model to be used
#MODEL = "gpt-4o"
MODEL = "gpt-4o-2024-05-13"

text_prompt = None
img_prompt = None
audio_prompt = None
video_prompt = None

def process_text(text_prompt):
    text_input = st.text_input("Enter your text:")
    print(text_prompt)
    if text_input:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": text_prompt},
                {"role": "user", "content": f"Hello! Could you solve {text_input}?"}
            ]
        )
        st.write("Assistant: " + completion.choices[0].message.content)

def process_image(image_input,img_prompt):
    print(img_prompt)
    if image_input:
        base64_image = base64.b64encode(image_input.read()).decode("utf-8")
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that responds in Markdown."},
                {"role": "user", "content": [
                    {"type": "text", "text": img_prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"}
                    }
                ]}
            ],
            temperature=0.0,
        )
        st.markdown(response.choices[0].message.content)

def process_audio(audio_input,audio_prompt):
    print(audio_prompt)
    if audio_input:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_input,
        )
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
            {"role": "system", "content":audio_prompt},
            {"role": "user", "content": [{"type": "text", "text": f"The audio transcription is: {transcription.text}"}],}
            ],
            temperature=0,
        )
        st.markdown(response.choices[0].message.content)

def process_audio_for_video(video_input,audio_prompt):
    print(video_prompt)
    if video_input:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=video_input,
        )
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
            {"role": "system", "content":audio_prompt },
            {"role": "user", "content": [{"type": "text", "text": f"The audio transcription is: {transcription}"}],}
            ],
            temperature=0,
        )
        st.markdown(response.choices[0].message.content)
        return response.choices[0].message.content

def save_video(video_file):
    # Save the uploaded video file
    with open(video_file.name, "wb") as f:
        f.write(video_file.getbuffer())
    return video_file.name

def process_video(video_path, seconds_per_frame=2):
    base64Frames = []
    base_video_path, _ = os.path.splitext(video_path)
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frames_to_skip = int(fps * seconds_per_frame)
    curr_frame = 0

    # Loop through the video and extract frames at specified sampling rate
    while curr_frame < total_frames - 1:
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        curr_frame += frames_to_skip

    video.release()

    # Extract audio from video
    audio_path = f"{base_video_path}.mp3"
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, bitrate="32k")
    clip.audio.close()
    clip.close()

    print(f"Extracted {len(base64Frames)} frames")
    print(f"Extracted audio to {audio_path}")

    return base64Frames, audio_path

def process_audio_and_video(video_input,video_prompt,audio_prompt):
    if video_input is not None:
        # Save the uploaded video file
        video_path = save_video(video_input )
    
        # Process the saved video
        base64Frames, audio_path = process_video(video_path, seconds_per_frame=1)

        # Get the transcript for the video model call
        transcript = process_audio_for_video(video_input,audio_prompt)
        
        # Generate a summary with visual and audio
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": video_prompt },
                {"role": "user", "content": [
                    "These are the frames from the video.",
                    *map(lambda x: {"type": "image_url",
                                    "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}, base64Frames)
                    
                ]},
            ],
            temperature=0,
        )
    
        st.markdown(response.choices[0].message.content)


def main():
    st.markdown("### OpenAI GPT-4o Model")
    st.markdown("#### The Omni Model with Text, Audio, Image, and Video")
    option = st.selectbox("Select an option", ("Text", "Image", "Audio", "Video"))
    if option == "Text":
        text_prompt = st.text_area("Enter a prompt:",value="You are a helpful assistant. Help me with my math homework!")
        process_text(text_prompt)
    elif option == "Image":
        image_input = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        img_prompt = st.text_input("Enter a prompt:",value="Help me understand what is in this picture and list ten facts as markdown outline with appropriate emojis that describes what you see.")
        process_image(image_input,img_prompt)
    elif option == "Audio":
        audio_input = st.file_uploader("Upload an audio file", type=["mp3", "wav"]) 
        audio_prompt = st.text_input("Enter a prompt:",value="You are generating a transcript summary. Create a summary of the provided transcription. Respond in Markdown.")
        process_audio(audio_input,audio_prompt)
    elif option == "Video":
        video_input = st.file_uploader("Upload a video file", type=["mp4"])
        audio_prompt = "You are generating a transcript summary. Create a summary of the provided transcription. Respond in Markdown."
        video_prompt = st.text_input("Enter a prompt:",value="You are generating a video summary. Create a summary of the provided video and its transcript. Respond in Markdown")
        process_audio_and_video(video_input,video_prompt,audio_prompt)

if __name__ == "__main__":
    main()

