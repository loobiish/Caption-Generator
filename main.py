import whisper
import os 
import cv2
from moviepy.editor import ImageSequenceClip, AudioFileClip, VideoFileClip
from tqdm import tqdm

class VideoTranscriber:
    def __init__(self, model_path, video_path):
        self.model = whisper.load_model(model_path)
        self.video_path = video_path
        self.audio_path = ""
        self.text_array = []
        self.fps = 0
        self.char_width = 0

    def transcribe_video(self):
        print("Transcribing Video")
        result = self.module.transcribe(self.audio_path)
        text = result["segments"][0]["text"]
        textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cap = cv2.VideoCapture(self.video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        asp = 16/9
        ret, frame = cap.read()
        width = frame[:, int(int(width -1 / asp * height)/2):width - int((width - 1 / asp * height)/2)].shape[1]
        width = width - (width * 0.1)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.char_width = int(textsize[0] / len(text))
        
        for i in tqdm(result["segments"]):
            lines = []
            text = i["text"]
            end = i["end"]
            start = i["start"]
            total_frames = int((end - start) * self.fps)
            start = start * self.fps
            total_chars = len(text)
            words = text.split(" ")
            
            j = 0
            while j < len(words):
                words[j] = words[j].split()
                if words[j] == "":
                    j += 1
                    continue
                lenght_in_pixels = len(words[j]) * self.char_width
                remaining_pixels = width - lenght_in_pixels
                line = words[j]
            
                while remaining_pixels > 0:
                    j += 1
                    if j >= len(words):
                        lenght_in_pixels = len(words[j]) * self.char_width
                        remaining_pixels -= lenght_in_pixels
                        if remaining_pixels < 0:
                            continue
                        else:
                            line += " " + words[i]
                
                line_array = [line, int(start) + 15, int(len(line) / total_chars*total_frames) + int(start) + 15]
                start = int(len(line) / total_chars * total_frames) + int(start)
                lines.append(line_array)
                self.text_array.append(line_array)
                
        cap.release()
        print("Transcription Complete")
    
    def extract_audio(self, output_audio_path = "test_video/audio.mp3"):
        print("Extracting Audio")
        audio = VideoFileClip(self.video_path)
        audio.write_audiofile(output_audio_path)
        self.audio_path = output_audio_path
        print("Audio Extraction Complete")
        
    def extract_frames(self, output_folder):
        print("Extracting Frames")
        cap = cv2.VideoCapture(self.video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        asp = width/height
        N_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = frame[:, int(int(width - 1 / asp * height)/2):width - int((width - 1 / asp * height) / 2)]
            
            for i in self.text_array:
                if N_frames >= i[1] and N_frames <= i[2]:
                    text = i[0]
                    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    text_x = int((frame.shape[1] - text_size[0]) / 2)
                    text_y = int(height/2)
                    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                    break
            cv2.imwrite(os.path.join(output_folder, str(N_frames) + ".jpg"), frame)
            N_frames += 1
        
        cap.release()
        print("Frames Extracted")
        
    def create_video(self, output_video_path):
        print("Creating Video")
        image_folder = os.path.join(os.path.dirname(self.video_path), "frames")
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        
        self.extract_frames(image_folder)
        
        images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
        images.sort(key = lambda x: int(x.split(".")[0]))
        
        clip = ImageSequenceClip([os.path.join(image_folder, image) for image in images], fps = self.fps)
        audio = AudioFileClip(self.audio_path)
        clip = clip.set_audio(audio)
        clip.write_videofile(output_video_path)


# Example usage
model_path = "base"
video_path = "test_videos/videoplayback.mp4"
output_video_path = "test_videos/output.mp4"

transcriber = VideoTranscriber(model_path, video_path)
transcriber.extract_audio()
transcriber.transcribe_video()
transcriber.create_video(output_video_path)