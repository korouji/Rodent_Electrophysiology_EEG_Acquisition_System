import socket
import struct
import csv
import signal
import time
import sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import os
os.environ["QT_QPA_PLATFORM"] = "wayland"

import socket
import struct
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

 
from collections import deque

LOCAL_PORT = 5006
ESP_PORT   = 5005
ESP_IP     = "10.209.193.172"   

BATCH_SIZE = 20

PACKET_FORMAT = "<BBB4h3hB"
SINGLE_PACKET_SIZE = struct.calcsize(PACKET_FORMAT)
BATCH_PACKET_SIZE = SINGLE_PACKET_SIZE * BATCH_SIZE

running = True

recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
recv_sock.bind(("", LOCAL_PORT))
recv_sock.settimeout(0.5)

send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_cmd(cmd: bytes):
    send_sock.sendto(cmd, (ESP_IP, ESP_PORT))

def signal_handler(sig, frame):
    global running
    running = False

signal.signal(signal.SIGINT, signal_handler)

print(f"Packet size: {BATCH_PACKET_SIZE} bytes")

cmd_to_send = b"START"
csv_filename = "eeg_imu_data.csv"
csv_headers = ["counter", "EEG1", "EEG2", "EEG3", "EEG4", "IMU_X", "IMU_Y", "IMU_Z"]

csv_file = open(csv_filename, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(csv_headers)

# ==========================================
# ==========================================
MAX_POINTS = 500 
x_data = deque(maxlen=MAX_POINTS)
eeg1_data = deque(maxlen=MAX_POINTS)
eeg2_data = deque(maxlen=MAX_POINTS)
eeg3_data = deque(maxlen=MAX_POINTS)
eeg4_data = deque(maxlen=MAX_POINTS)
imux_data = deque(maxlen=MAX_POINTS)
imuy_data = deque(maxlen=MAX_POINTS)
imuz_data = deque(maxlen=MAX_POINTS)

plt.ion()
fig, (ax_eeg, ax_imu) = plt.subplots(2, 1, figsize=(10, 8))

line_eeg1, = ax_eeg.plot([], [], label="EEG1")
line_eeg2, = ax_eeg.plot([], [], label="EEG2")
line_eeg3, = ax_eeg.plot([], [], label="EEG3")
line_eeg4, = ax_eeg.plot([], [], label="EEG4")
ax_eeg.set_title("Real-time EEG Data")
ax_eeg.legend(loc="upper right")

line_imux, = ax_imu.plot([], [], label="IMU X")
line_imuy, = ax_imu.plot([], [], label="IMU Y")
line_imuz, = ax_imu.plot([], [], label="IMU Z")
ax_imu.set_title("Real-time IMU Data")
ax_imu.legend(loc="upper right")
# ==========================================

time.sleep(1)
send_cmd(cmd_to_send)
print(f"{cmd_to_send.decode()} sent to ESP32")
print("Waiting for data... Press Ctrl+C to stop.")

last_counter = -1
row_buffer = [] 
global_x = 0 

while running:
    try:
        data, addr = recv_sock.recvfrom(1024 * BATCH_SIZE)  

        if len(data) != BATCH_PACKET_SIZE:
            print(f"Wrong size: {len(data)}")
            continue

        for i in range(BATCH_SIZE):
            offset = i * SINGLE_PACKET_SIZE
            unpacked = struct.unpack(PACKET_FORMAT, data[offset:offset + SINGLE_PACKET_SIZE])

            header = unpacked[0]
            packetType = unpacked[1]
            counter = unpacked[2]
            eeg = unpacked[3:7] # eeg1, eeg2, eeg3, eeg4
            imu = unpacked[7:10] # imu_x, imu_y, imu_z
            footer = unpacked[10]

            if header != 0xAA or footer != 0x55:
                print("Corrupted packet")
                continue

            if last_counter != -1:
                if (counter - last_counter) % 256 != 1:
                    pass 
                    # print("Packet loss detected")
                    
            last_counter = counter

            row_buffer.append([counter, *eeg, *imu])

            x_data.append(global_x)
            eeg1_data.append(eeg[0])
            eeg2_data.append(eeg[1])
            eeg3_data.append(eeg[2])
            eeg4_data.append(eeg[3])
            imux_data.append(imu[0])
            imuy_data.append(imu[1])
            imuz_data.append(imu[2])
            global_x += 1

            if len(row_buffer) >= 100:
                csv_writer.writerows(row_buffer)
                row_buffer = []
                
                line_eeg1.set_data(x_data, eeg1_data)
                line_eeg2.set_data(x_data, eeg2_data)
                line_eeg3.set_data(x_data, eeg3_data)
                line_eeg4.set_data(x_data, eeg4_data)
                
                line_imux.set_data(x_data, imux_data)
                line_imuy.set_data(x_data, imuy_data)
                line_imuz.set_data(x_data, imuz_data)
                
                ax_eeg.relim()
                ax_eeg.autoscale_view()
                ax_imu.relim()
                ax_imu.autoscale_view()
                
                plt.pause(0.001)

    except socket.timeout:
        continue

if row_buffer:
    csv_writer.writerows(row_buffer)

send_cmd(b"STOP")
csv_file.close()
recv_sock.close()
send_sock.close()

plt.ioff()
print(f"\nStopped cleanly. Data saved to {csv_filename}")
plt.show()
