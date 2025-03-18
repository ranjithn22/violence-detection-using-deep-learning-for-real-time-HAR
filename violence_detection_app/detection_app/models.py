import torch
import torch.nn as nn


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.5):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)

        # Final fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Initial hidden state (h0) for GRU
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate through GRU
        out, _ = self.gru(x, h0)

        # Apply dropout to the GRU output (use the last hidden state)
        out = self.dropout(out)

        # We use the last time step for classification
        out = self.fc(out[:, -1, :])

        return out


# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.db import models

# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.db import models


class RegistrationsTable(models.Model):
    slno = models.AutoField(primary_key=True)
    name = models.CharField()
    email = models.CharField(db_column='email')  # Field renamed to remove unsuitable characters. Field renamed because it started with '_'.
    phoneno = models.CharField(max_length=10)
    password = models.CharField()
    address = models.CharField()
    details = models.TextField()

    class Meta:
        managed = False
        db_table = 'registrations_table'


class UsersTable(models.Model):
    user_id = models.AutoField(primary_key=True)
    name = models.CharField(blank=True, null=True)
    email = models.CharField(blank=True, null=True)
    phoneno = models.CharField(max_length=10, blank=True, null=True)
    password = models.CharField(blank=True, null=True)
    service_creation_date = models.DateTimeField(blank=True, null=True)
    details = models.TextField()
    v_video = models.CharField(blank=True, null=True)
    camera_details = models.CharField(blank=True, null=True)
    v_timestamp = models.CharField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'users_table'


class AdminData(models.Model):
    slno = models.AutoField(primary_key=True)
    username = models.CharField(blank=True, null=True)
    email = models.CharField(blank=True, null=True)
    password = models.CharField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'admin_data'


import smtplib
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


class EmergencyAlertSystem:
    def __init__(self, sender_email, sender_password, receiver_email):
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.receiver_email = receiver_email

    def send_email(self, subject, message):
        # Set up the email server (Gmail SMTP)
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()  # Secure the connection
        server.login(self.sender_email, self.sender_password)

        # Create the email
        msg = MIMEMultipart()
        msg["From"] = self.sender_email
        msg["To"] = self.receiver_email
        msg["Subject"] = subject
        msg.attach(MIMEText(message, "html"))

        # Send the email
        server.sendmail(self.sender_email, self.receiver_email, msg.as_string())
        server.quit()

        print("✅ Email sent successfully!")

