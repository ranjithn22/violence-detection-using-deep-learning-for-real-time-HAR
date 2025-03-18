from .models import EmergencyAlertSystem, GRUModel
import json
from django.http import JsonResponse
from django.utils import timezone
from datetime import timedelta
from django.contrib.auth.hashers import make_password, check_password
from .models import AdminData  
import os
from django.conf import settings
from .models import UsersTable
import time
from django.contrib import messages
from django.shortcuts import redirect, render
from django.contrib.auth.hashers import make_password
from .models import RegistrationsTable
from twilio.rest import Client
import cv2
import numpy as np
import torch
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from ultralytics import YOLO
import pytz

input_size = 6
# Initialize GRU model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load GRU model
def load_gru_model(model_path, device):
    # Hyperparameters
    input_size = 6  # Bounding box coordinates (x1, y1, x2, y2) + confidence score
    hidden_size = 128  # Hidden size of the GRU (adjustable)
    num_layers = 2  # Number of GRU layers
    num_classes = 2  # Fight or no fight
    learning_rate = 0.001
    num_epochs = 50  # Number of epochs for training
    max_seq_len = 90  # Max sequence length (frames)
    batch_size = 16  # Batch size for training
    max_detections_per_frame = 10  # Max detections per frame

    model = GRUModel(input_size, hidden_size, num_layers, num_classes).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    return model


# Step 5: Function to extract spatial features using YOLOv8
def extract_spatial_features(video_path, yolo_model):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    spatial_features = []
    for idx in range(total_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break

        # Perform inference using YOLOv8 (only predict persons)
        results = yolo_model.predict(frame, verbose=False)

        results = yolo_model(frame, classes=[0])


        # Extract bounding boxes and confidence scores for people (class 0)
        detections = results[0].boxes.data.cpu().numpy()

        if len(detections) > 0:
            spatial_features.append(detections[:, :6])
        else:
            spatial_features.append(np.zeros((1, 6)))

    cap.release()
    return spatial_features


# Step 6: Preprocess the video features
def preprocess_video(features, max_seq_len, max_detections_per_frame):
    padded_features = np.zeros(
        (max_seq_len, max_detections_per_frame, input_size))
    seq_len = min(len(features), max_seq_len)

    for i in range(seq_len):
        frame_features = features[i]
        num_detections = min(frame_features.shape[0], max_detections_per_frame)
        padded_features[i, :num_detections, :] = frame_features[:num_detections, :]

    aggregated_features = np.max(padded_features, axis=1)
    return torch.tensor(aggregated_features, dtype=torch.float32).unsqueeze(0).to(device)


# Step 7: Make predictions
def predict_video(video_path, gru_model, yolo_model, max_seq_len, max_detections_per_frame, device):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Prepare a temporary file to hold the processed video
    temp_output_path = os.path.join(settings.MEDIA_ROOT, 'temp_processed_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for H264 codec
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        raise ValueError("Failed to open VideoWriter. Check codec and file path.")

    fight_probabilities = []
    no_fight_probabilities = []

    # Extract spatial features
    spatial_features = extract_spatial_features(video_path, yolo_model)
    print(f"Extracted spatial features for {len(spatial_features)} frames.")

    processed_features = preprocess_video(spatial_features, max_seq_len, max_detections_per_frame)
    print(f"Processed features shape: {processed_features.shape}")

    # Make predictions
    with torch.no_grad():
        outputs = gru_model(processed_features)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        fight_prob = probabilities[1] * 100
        no_fight_prob = probabilities[0] * 100
        fight_probabilities.append(fight_prob)
        no_fight_probabilities.append(no_fight_prob)


        prediction = "Fight" if probabilities[1] > probabilities[0] else "No Fight"

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    print(f"Processed {frame_count} frames and saved video to: {temp_output_path}")

    if not os.path.exists(temp_output_path):
        raise FileNotFoundError(f"Processed video file not found at: {temp_output_path}")

    with open(temp_output_path, 'rb') as f:
        video_binary = f.read()

    return prediction, probabilities, video_binary, fight_probabilities, no_fight_probabilities


# Function to upload a file to Google Drive
def upload_to_google_drive(file_path, folder_id):

    credentials = service_account.Credentials.from_service_account_file(
        'detection_app/credentials.json',
        scopes=['https://www.googleapis.com/auth/drive']
    )

    drive_service = build('drive', 'v3', credentials=credentials)

    file_name = os.path.basename(file_path)
    file_metadata = {
        'name': file_name,
        'parents': [folder_id]
    }

    media = MediaFileUpload(file_path, resumable=True)
    try:
        file = drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()

        file_id = file.get('id')

        drive_service.permissions().create(
            fileId=file_id,
            body={'type': 'anyone', 'role': 'reader'}
        ).execute()

        shareable_url = f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"
        return shareable_url
    except Exception as e:
        print(f"Error uploading to Google Drive: {e}")
        raise Exception("Failed to upload to Google Drive")


# Function to send SMS using Twilio
def send_sms(to_phone_number, message):

    try:

        client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)


        message = client.messages.create(
            body=message,
            from_=settings.TWILIO_PHONE_NUMBER,
            to=to_phone_number
        )

        print(f"SMS sent successfully to {to_phone_number}. Message SID: {message.sid}")
    except Exception as e:
        print(f"Error sending SMS: {e}")


def index(request):
    if request.method == 'POST':
        print("POST request received")

        video_file = request.FILES.get('video_file')
        camera_name = request.POST.get('camera_name')
        if not video_file:
            print("No file uploaded")
            return render(request, 'index.html', {'error': 'No file uploaded'})

        print(f"Uploaded file: {video_file.name}, Camera Name: {camera_name}")

        if not os.path.exists(settings.MEDIA_ROOT):
            os.makedirs(settings.MEDIA_ROOT)

        video_path = os.path.join(settings.MEDIA_ROOT, video_file.name)
        try:
            with open(video_path, 'wb+') as destination:
                for chunk in video_file.chunks():
                    destination.write(chunk)
            print(f"File saved to: {video_path}")
        except Exception as e:
            print(f"Error saving file: {e}")
            return render(request, 'index.html', {'error': 'Error saving file'})

        # Load models
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            yolo_model = YOLO(os.path.join(settings.BASE_DIR, 'models/yolo11n.pt'))
            gru_model = load_gru_model(os.path.join(settings.BASE_DIR, 'models/GRUmodelperson.pth'), device)
            print("Models loaded successfully")  # Debugging
        except Exception as e:
            print(f"Error loading models: {e}")  # Debugging
            return render(request, 'index.html', {'error': 'Error loading models'})


        try:

            prediction, probabilities, _, fight_probabilities, no_fight_probabilities = predict_video(
                video_path, gru_model, yolo_model, max_seq_len=90, max_detections_per_frame=10, device=device)
            print(f"Prediction: {prediction}")  # Debugging
        except Exception as e:
            print(f"Error during prediction: {e}")  # Debugging
            return render(request, 'index.html', {'error': 'Error during prediction'})

        if prediction == "Fight":
            temp_output_path = os.path.join(settings.MEDIA_ROOT, 'temp_processed_video.mp4')

            try:
                if not os.path.exists(temp_output_path):
                    raise FileNotFoundError(f"Processed video file not found at: {temp_output_path}")

                if os.path.getsize(temp_output_path) < 1024:
                    raise ValueError("Processed video file is too small or empty.")

                fight_videos_folder_id = "enter the folder where vedio should be uploaded"

                google_drive_url = upload_to_google_drive(temp_output_path, fight_videos_folder_id)
                print(f"Processed video uploaded to Google Drive: {google_drive_url}")

                user = UsersTable.objects.filter(camera_details=camera_name).first()
                if user:
                    # Append the new URL and timestamp
                    existing_urls = user.v_video.split(',') if user.v_video else []
                    existing_timestamps = user.v_timestamp.split(',') if user.v_timestamp else []

                    existing_urls.append(google_drive_url)

                    current_timestamp = timezone.now().astimezone(pytz.timezone('Asia/Kolkata'))
                    naive_timestamp = current_timestamp.replace(tzinfo=None)

                    existing_timestamps.append(
                        naive_timestamp.strftime('%Y-%m-%d %H:%M:%S'))

                    user.v_video = ','.join(existing_urls)
                    user.v_timestamp = ','.join(existing_timestamps)
                    user.service_creation_date = timezone.now()
                    user.save()
                    print(f"Google Drive URL stored for user with camera: {camera_name}")
                    phone_number = user.phoneno
                    if phone_number:
                        sms_message = (
                            f"Alert: A fight was detected in your area. "
                            f"You can view the video here: {google_drive_url}"
                        )
                        send_sms(phone_number, sms_message)
                    else:
                        print("User  does not have a phone number registered.")
                else:
                    print(f"No user found for the camera name: {camera_name}")
            except Exception as e:
                print(f"Error uploading to Google Drive: {e}")

            try:
                if os.path.exists(temp_output_path):
                    time.sleep(1)
                    os.remove(temp_output_path)
            except Exception as e:
                print(f"Error deleting file: {e}")
        else:
            print("No fight detected, processed video not uploaded to Google Drive.")

        return render(request, 'index.html', {

        })
    return render(request, 'index.html')


def register(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        phoneno = request.POST.get('phoneno')
        password = request.POST.get('password')
        address = request.POST.get('address')
        details = request.POST.get('details')

        if not phoneno.startswith("+91"):
            phoneno = "+91" + phoneno

        hashed_password = make_password(password)

        registration = RegistrationsTable(
            name=name,
            email=email,
            phoneno=phoneno,
            password=hashed_password,
            address=address,
            details=details
        )
        registration.save()

        alert_system = EmergencyAlertSystem(
            sender_email="enter your mail",
            sender_password="enter your password",
            receiver_email=email
        )
        subject = "Welcome to Violence Detection System – Stay Alert, Stay Safe!"
        message = f"""
        <html>
        <head>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background-color: #f4f4f4;
                    text-align: center;
                    padding: 20px;
                }}
                .container {{
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                    max-width: 450px;
                    margin: auto;
                }}
                h2 {{
                    color: #222;
                }}
                p {{
                    color: #555;
                    font-size: 16px;
                }}
                .highlight {{
                    font-weight: bold;
                    color: #007BFF;
                }}
                .btn {{
                    display: inline-block;
                    padding: 12px 25px;
                    font-size: 16px;
                    font-weight: bold;
                    color: white;
                    background: #007BFF;
                    text-decoration: none;
                    border-radius: 5px;
                    margin-top: 15px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                }}
                .btn:hover {{
                    background: #0056b3;
                }}
                .footer {{
                    margin-top: 20px;
                    font-size: 14px;
                    color: #777;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h2>Welcome to Violence Detection System App</h2>
                <p>Dear <span class="highlight">{name}</span>,</p>
                <p>We are excited to have you on board! Your registration is now complete, and you are ready to experience the power of <strong>AI-driven real-time violence detection.</strong></p>
                <p>With <strong>state-of-the-art AI models (YOLO11 + GRU)</strong>, our system ensures real-time detection and instant alerts, making both public and private spaces safer.</p>
                <a href="http://127.0.0.1:8000/" class="btn">🔍 Explore the Application</a>

                <p class="footer">If you have any questions, feel free to reach out to our support team.<br> Stay safe, stay secure! 🚀</p>
            </div>
        </body>
        </html>

        """

        alert_system.send_email(subject, message)
        messages.success(request, 'Successfully registered!')

        return redirect('home')

    return render_home(request)


def render_home(request):
    return render(request, 'home.html')


def login(request):
    if request.method == 'POST':
        email = request.POST.get('username')
        password = request.POST.get('password')

        try:
            user = UsersTable.objects.get(email=email)

            if check_password(password, user.password):
                request.session['email'] = user.email
                request.session['user_id'] = user.user_id
                return redirect('user_dashboard')
            else:
                messages.error(request, 'Invalid email or password.')

        except UsersTable.DoesNotExist:
            messages.error(request, 'Invalid email or password.')

    return render(request, 'home.html')


def user_dashboard(request):

    email = request.session.get('email')
    user = UsersTable.objects.filter(email=email).first()

    if not user:
        return redirect('login')

    video_urls = user.v_video.split(',') if user.v_video else []
    timestamps = user.v_timestamp.split(',') if user.v_timestamp else []

    if video_urls:
        processed_video = video_urls[-1].replace("/view?usp=sharing","/preview")
        video_available = True
    else:
        processed_video = ''
        video_available = False

    previous_results = list(zip(timestamps, video_urls, range(1, len(timestamps) + 1)))
    previous_results.reverse()
    context = {
        'user': user,
        'processed_video': processed_video,
        'video_available': video_available,
        'previous_results': previous_results,
    }
    return render(request, 'user_dashboard.html', context)


def admin_register(request):
    if request.method == 'POST':
        username = request.POST.get('admin_name')
        email = request.POST.get('email')
        password = request.POST.get('password')

        if AdminData.objects.filter(username=username).exists():
            messages.error(request, "Admin name already taken!")
            return redirect('admin_register')

        hashed_password = make_password(password)

        AdminData.objects.create(username=username, email=email, password=hashed_password)

        messages.success(request, "Admin registered successfully!")
        return redirect('admin_login')

    return render(request, 'admin_registration.html')


def admin_login(request):
    if request.method == 'POST':
        username = request.POST.get('admin_name')
        password = request.POST.get('password')

        try:
            admin_data = AdminData.objects.get(username=username)
            # Validate the password
            if check_password(password, admin_data.password):

                return redirect('admin_dashboard')
            else:
                messages.error(request, "Invalid admin name or password.")
        except AdminData.DoesNotExist:
            messages.error(request, "Invalid admin name or password.")

        return redirect('admin_login')

    return render(request, 'admin_login.html')


def admin_dashboard(request):
    # Fetch data from the RegistrationsTable
    registered_users = RegistrationsTable.objects.all()

    # Fetch data from the UsersTable (if needed)
    users_on_service = UsersTable.objects.all()

    context = {
        'registered_users': registered_users,
        'users_on_service': users_on_service,
    }
    return render(request, 'admin_dashboard.html', context)


def approve_user(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            slno = data.get('slno')
            camera_name = data.get('camera_name')
            if not slno or not camera_name:
                return JsonResponse({'status': 'error', 'message': 'Missing SLNO or Camera Name.'})

            registration = RegistrationsTable.objects.get(slno=slno)

            now_utc = timezone.now()
            ist_time = now_utc + timedelta(hours=5, minutes=30)


            user = UsersTable(
                name=registration.name,
                email=registration.email,
                phoneno=registration.phoneno,
                password=registration.password,
                details=registration.details,
                service_creation_date=ist_time.replace(tzinfo=None),
                camera_details=camera_name
            )
            user.save()

            # Optionally, delete the user from RegistrationsTable
            registration.delete()
            # Initialize Alert System
            alert_system = EmergencyAlertSystem(
                sender_email="enter your mail",
                sender_password="enter your password",
                receiver_email=registration.email
            )
            user_name=registration.name
            subject = "Welcome to Violence Detection System – Stay Alert, Stay Safe!"
            message = f"""
            <html>
            <head>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        background-color: #f4f4f4;
                        text-align: center;
                        padding: 20px;
                    }}
                    .container {{
                        background: white;
                        padding: 20px;
                        border-radius: 10px;
                        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                        max-width: 450px;
                        margin: auto;
                    }}
                    h2 {{
                        color: #222;
                    }}
                    p {{
                        color: #555;
                        font-size: 16px;
                    }}
                    .highlight {{
                        font-weight: bold;
                        color: #007BFF;
                    }}
                    .btn {{
                        display: inline-block;
                        padding: 12px 25px;
                        font-size: 16px;
                        font-weight: bold;
                        color: white;
                        background: #007BFF;
                        text-decoration: none;
                        border-radius: 5px;
                        margin-top: 15px;
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    }}
                    .btn:hover {{
                        background: #0056b3;
                    }}
                    .footer {{
                        margin-top: 20px;
                        font-size: 14px;
                        color: #777;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h2>🎉 Camera Approved: {camera_name} is Now Active!</h2>
                    <p>Dear <span class="highlight">{user_name}</span>,</p>
                    <p>Your camera <strong>"{camera_name}"</strong> has been successfully approved and is now actively monitoring your selected area using our <strong>AI-driven Violence Detection System.</strong></p>

                    <p><strong>🔍 Ensuring Safety in Your Selected Area</strong></p>
                    <p>Our advanced system is now operational, offering:</p>
                    <p>✔ <strong>Instant violence detection</strong> powered by <strong>YOLO11 + GRU</strong>.<br>  
                    ✔ <strong>Real-time alerts</strong> for quick response and intervention.<br>  
                    ✔ <strong>Enhanced security</strong> to ensure a safer environment.</p>

                    <a href="http://127.0.0.1:8000/" class="btn">🔍 Access Your Dashboard</a>

                    <p class="footer">If you have any questions, feel free to reach out to our support team.<br>  
                    Stay safe, and thank you for choosing our AI-powered security solution! 🚀</p>
                </div>

            </body>
            </html>

            """

            # Send the alert
            alert_system.send_email(subject, message)
            return JsonResponse({'status': 'success'})
        except RegistrationsTable.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'User not found.'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})

    return JsonResponse({'status': 'error', 'message': 'Invalid request.'})