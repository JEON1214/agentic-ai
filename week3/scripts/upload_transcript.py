import requests
from pathlib import Path

SERVER = "http://localhost:8000"
TRANSCRIPT = Path(__file__).resolve().parents[1] / 'data' / 'transcript_user.txt'

def upload(collection='video_transcripts', user_id='user1', session_id=''):
    files = {'file': ('transcript_user.txt', TRANSCRIPT.read_text(encoding='utf-8'))}
    data = {'collection': collection, 'user_id': user_id, 'session_id': session_id}
    resp = requests.post(SERVER + '/upload', files=files, data=data)
    print('Status:', resp.status_code)
    print(resp.text)

if __name__ == '__main__':
    upload()
