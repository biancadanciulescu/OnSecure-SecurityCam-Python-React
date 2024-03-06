from flask import Flask, render_template, Response, jsonify
import cv2
import threading
import winsound
import atexit

app = Flask(__name__)

cam = cv2.VideoCapture(0)
camera_lock = threading.Lock()
motion_detected = False
prev_frame = None

def detect_motion():

    video_source = 1
    cap = cv2.VideoCapture(video_source)
    threshold = 100
    

    ret, prev_frame = cap.read()
    if not ret:
        print("Error reading video source")
        return

    while True:

        ret, curr_frame = cap.read()
        if not ret:
            print("Error reading video source")
            break

        
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        
        frame_diff = cv2.absdiff(prev_gray, curr_gray)

        
        _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        
        motion_detected2 = any(cv2.contourArea(contour) > threshold for contour in contours)

        
        prev_frame = curr_frame.copy()

        if motion_detected2:
            winsound.PlaySound('alert.wav', winsound.SND_ASYNC) 
            return True      
        else:
            return False
            

    cap.release()
    cv2.destroyAllWindows()


def generate_frames():
    while True:
        with camera_lock:
            _, frame = cam.read()
            if not _:
                break
            _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data')
def api_data():
    global motion_detected
    print('API Data accessed!')
    data = {
        'motion_detected': motion_detected
    }
    motion_detected = detect_motion() 
    return jsonify(data)

def cleanup():
    print('Cleaning up...')
    cam.release()


atexit.register(cleanup)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
