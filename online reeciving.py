

import asyncio
import struct
import csv
import sys
import queue
import threading
import platform
import time
import signal
from collections import deque
from datetime import datetime

from bleak import BleakClient, BleakScanner


# ─── BLE identifiers ────────────────────────────────────────────────────────────
DEVICE_NAME   = "EEG_Sensor"
SERVICE_UUID  = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
DATA_UUID     = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
CMD_UUID      = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"

# ─── Protocol constants ───────────────────────────────────────────────────────
SAMPLE_FORMAT  = "<BBB4h3hB"
SAMPLE_SIZE    = struct.calcsize(SAMPLE_FORMAT)
BATCH_SIZE     = 10
BATCH_BYTES    = SAMPLE_SIZE * BATCH_SIZE

assert SAMPLE_SIZE == 18, f"Sample size is {SAMPLE_SIZE}, expected 18"

HEADER_BYTE = 0xAA
FOOTER_BYTE = 0x55

DISCOVERY_TIMEOUT = 8.0
PLOT_WINDOW_SAMPLES = 500
PLOT_UPDATE_EVERY = 50

import platform
if platform.system() == "Windows":
    import matplotlib
    matplotlib.use("Agg")
    print("[plot] Windows: Using Agg backend (no GUI)")

if platform.system() == "Windows":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        print("[windows] Using ProactorEventLoopPolicy")
    except Exception as e:
        print(f"[windows] Could not set policy: {e}")

@staticmethod
async def find_device_fast(name: str, timeout: float = 8.0):
    if not WindowsBLEHelper.is_windows():
        return await BleakScanner.find_device_by_name(name, timeout=timeout)
    
    try:
        devices = await BleakScanner.discover(timeout=min(timeout, 5.0))
        for device in devices:
            if device.name and name.lower() in device.name.lower():
                return device
    except Exception as e:
        print(f"[windows] Scan error: {e}")
    
    return await BleakScanner.find_device_by_name(name, timeout=timeout)

# ─── Matplotlib backend ──────────────────────────────────────────────────────
import matplotlib
_BACKENDS = ["TkAgg", "Qt5Agg", "WxAgg", "Agg"]
_INTERACTIVE = False
for _b in _BACKENDS:
    try:
        matplotlib.use(_b)
        import matplotlib.pyplot as plt
        _test = plt.figure()
        plt.close(_test)
        _INTERACTIVE = (_b != "Agg")
        print(f"[plot] Using backend: {_b}")
        break
    except Exception:
        continue
else:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    print("[plot] WARNING: No interactive backend found.")

# ═══════════════════════════════════════════════════════════════════════════════
#  CSV Writer
# ═══════════════════════════════════════════════════════════════════════════════

class CSVWriter(threading.Thread):
    HEADERS = ["counter", "EEG1", "EEG2", "EEG3", "EEG4",
                "IMU_X", "IMU_Y", "IMU_Z"]

    def __init__(self, filename: str):
        super().__init__(daemon=True, name="CSVWriter")
        self._filename = filename
        self._q: queue.Queue = queue.Queue(maxsize=200_000)
        self._stop_event = threading.Event()
        self._rows_written = 0

    def put(self, row: list):
        try:
            self._q.put_nowait(row)
        except queue.Full:
            print("[csv] WARNING: queue full — sample dropped!")

    def stop(self):
        self._stop_event.set()

    @property
    def rows_written(self) -> int:
        return self._rows_written

    def run(self):
        try:
            with open(self._filename, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.HEADERS)
                while not (self._stop_event.is_set() and self._q.empty()):
                    try:
                        row = self._q.get(timeout=0.1)
                        writer.writerow(row)
                        self._q.task_done()
                        self._rows_written += 1
                    except queue.Empty:
                        continue
                f.flush()
            print(f"[csv] Closed — {self._rows_written} rows → {self._filename}")
        except Exception as e:
            print(f"[csv] ERROR: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  LivePlot
# ═══════════════════════════════════════════════════════════════════════════════

class LivePlot:
    EEG_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    IMU_COLORS = ["#9b59b6", "#1abc9c", "#e67e22"]

    def __init__(self):
        self.x = deque(maxlen=PLOT_WINDOW_SAMPLES)
        self.eeg = [deque(maxlen=PLOT_WINDOW_SAMPLES) for _ in range(4)]
        self.imu = [deque(maxlen=PLOT_WINDOW_SAMPLES) for _ in range(3)]
        self.gx = 0
        self.count = 0
        self.loss = 0
        self.total = 0
        self._lock = threading.Lock()

        if not _INTERACTIVE:
            self.fig = None
            return

        try:
            self.fig, (self.ax_eeg, self.ax_imu) = plt.subplots(2, 1, figsize=(11, 7))
            self.fig.suptitle("EEG Live Monitor", fontsize=13)

            self.lines_eeg = [
                self.ax_eeg.plot([], [], lw=0.8, color=c, label=f"EEG{i+1}")[0]
                for i, c in enumerate(self.EEG_COLORS)
            ]
            self.ax_eeg.set_ylabel("ADC counts")
            self.ax_eeg.legend(loc="upper right", fontsize=8)
            self.ax_eeg.grid(True, alpha=0.3)

            self.lines_imu = [
                self.ax_imu.plot([], [], lw=0.8, color=c, label=l)[0]
                for c, l in zip(self.IMU_COLORS, ["IMU X", "IMU Y", "IMU Z"])
            ]
            self.ax_imu.set_ylabel("IMU Sensor data (raw)")
            #self.ax_imu.set_xlabel("Sample")
            self.ax_imu.legend(loc="upper right", fontsize=8)
            self.ax_imu.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.ion()
            plt.show(block=False)
        except Exception as e:
            print(f"[plot] Could not initialize plot: {e}")
            self.fig = None

    def push(self, eeg_vals: tuple, imu_vals: tuple):
        with self._lock:
            self.x.append(self.gx)
            self.gx += 1
            for i, v in enumerate(eeg_vals):
                self.eeg[i].append(v)
            for i, v in enumerate(imu_vals):
                self.imu[i].append(v)
            self.total += 1
            self.count += 1

            if self.count >= PLOT_UPDATE_EVERY:
                self._redraw()
                self.count = 0

    def _redraw(self):
        if not _INTERACTIVE or self.fig is None:
            return
        try:
            xs = list(self.x)
            for i, line in enumerate(self.lines_eeg):
                line.set_data(xs, list(self.eeg[i]))
            for i, line in enumerate(self.lines_imu):
                line.set_data(xs, list(self.imu[i]))
            self.ax_eeg.relim()
            self.ax_eeg.autoscale_view()
            self.ax_imu.relim()
            self.ax_imu.autoscale_view()
            
            loss_pct = (self.loss / max(self.total, 1)) * 100
            try:
                self.fig.canvas.manager.set_window_title(
                    f"EEG BLE Monitor — {self.total} samples | loss: {self.loss} ({loss_pct:.1f}%)"
                )
            except:
                pass
            self.fig.canvas.flush_events()
            plt.pause(0.001)
        except Exception:
            pass

    def save_final(self, path: str):
        if self.fig:
            try:
                self.fig.savefig(path, dpi=150)
                print(f"[plot] Final plot saved → {path}")
            except Exception:
                pass

    def close(self):
        if _INTERACTIVE and self.fig:
            try:
                plt.ioff()
                plt.close(self.fig)
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════════════
#  Packet parser
# ═══════════════════════════════════════════════════════════════════════════════

def parse_notification(data: bytearray,
                       csv_writer: CSVWriter,
                       plot: LivePlot,
                       last_counter: list,
                       stats: dict) -> None:
    if len(data) != BATCH_BYTES:
        if len(data) > 0:
            stats["fragments"] = stats.get("fragments", 0) + 1
        return

    for i in range(BATCH_SIZE):
        offset = i * SAMPLE_SIZE
        try:
            fields = struct.unpack_from(SAMPLE_FORMAT, data, offset)
        except struct.error:
            continue

        header, pkt_type, counter = fields[0], fields[1], fields[2]
        eeg = fields[3:7]
        imu = fields[7:10]
        footer = fields[10]

        if header != HEADER_BYTE or footer != FOOTER_BYTE:
            stats["corrupt"] += 1
            continue

        lc = last_counter[0]
        if lc >= 0:
            expected = (lc + 1) % 256
            if counter != expected:
                lost = (counter - lc) % 256 - 1
                if lost > 0:
                    stats["loss"] += lost
                    plot.loss += lost
        last_counter[0] = counter

        csv_writer.put([counter, *eeg, *imu])
        plot.push(eeg, imu)
        stats["received"] += 1


# ═══════════════════════════════════════════════════════════════════════════════
#  Windows BLE Helper 
# ═══════════════════════════════════════════════════════════════════════════════

class WindowsBLEHelper:
    """Helper class for Windows BLE optimization"""
    
    @staticmethod
    def is_windows():
        return platform.system() == "Windows"
    
    @staticmethod
    def get_adapter():
        """Get the best BLE adapter on Windows"""
        return None
    
    @staticmethod
    async def find_device_fast(name: str, timeout: float = 8.0):
        """Fast device discovery for Windows"""
        if not WindowsBLEHelper.is_windows():
            return await BleakScanner.find_device_by_name(name, timeout=timeout)
        
        # Windows-specific fast discovery
        try:
            devices = await BleakScanner.discover(
                timeout=timeout,
                return_adv=False
            )
            for device in devices:
                if device.name and name in device.name:
                    return device
        except Exception as e:
            print(f"[windows] Discovery error: {e}")
            
        return await BleakScanner.find_device_by_name(name, timeout=timeout)


# ═══════════════════════════════════════════════════════════════════════════════
#  BLE async main 
# ═══════════════════════════════════════════════════════════════════════════════

async def run(csv_writer: CSVWriter, plot: LivePlot,
              stats: dict, stop_event: threading.Event) -> None:
    
    print(f"[ble] Scanning for \"{DEVICE_NAME}\"...")
    
    device = await WindowsBLEHelper.find_device_fast(DEVICE_NAME, DISCOVERY_TIMEOUT)

    if device is None:
        print(f"\n[ERROR] Device \"{DEVICE_NAME}\" not found.")
        print("  💡 Make sure ESP32 is powered on and advertising.")
        if platform.system() == "Windows":
            print("  💡 Windows tips:")
            print("     1. Open Bluetooth settings and check if device appears")
            print("     2. Try: 'python -m bleak.examples.discover'")
            print("     3. Restart Bluetooth service")
            print("     4. Try a USB BLE dongle if built-in fails")
        return

    print(f"[ble] Found: {device.name}  addr={device.address}")

    last_counter = [-1]

    def on_notification(characteristic, data: bytearray):
        if not stop_event.is_set():
            parse_notification(data, csv_writer, plot, last_counter, stats)

    try:
        client_timeout = 10.0 if platform.system() == "Windows" else 20.0
        
        async with BleakClient(device, timeout=client_timeout) as client:
            if not client.is_connected:
                print("[ERROR] BleakClient.connect() failed.")
                return

            print(f"[ble] Connected.  MTU = {client.mtu_size} bytes")
            
            if platform.system() == "Windows":
                print("[ble] Windows: Connection established successfully!")
                await asyncio.sleep(0.2)
            else:
                await asyncio.sleep(0.5)

            try:
                await client.start_notify(DATA_UUID, on_notification)
                print(f"[ble] Subscribed to DATA")
            except Exception as e:
                print(f"[ble] Subscribe error: {e}")
                if platform.system() == "Windows":
                    print("[ble] Windows: Retrying subscribe...")
                    await asyncio.sleep(0.5)
                    await client.start_notify(DATA_UUID, on_notification)
                    print("[ble] Subscribe retry successful")

            await asyncio.sleep(0.3)
            
            try:
                if platform.system() == "Windows":
                    await client.write_gatt_char(CMD_UUID, b"START", response=True)
                else:
                    await client.write_gatt_char(CMD_UUID, b"START", response=False)
                print("[ble] START sent")
            except Exception as e:
                print(f"[ble] START error: {e}")
                return

            stats["t_start"] = time.time()
            print("[ble] Receiving data... Press Ctrl+C to stop.\n")

            try:
                while client.is_connected and not stop_event.is_set():
                    await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                pass

            print("\n[ble] Stopping...")
            if client.is_connected:
                try:
                    await client.write_gatt_char(CMD_UUID, b"STOP", response=False)
                    print("[ble] STOP sent.")
                except Exception:
                    pass
                try:
                    await client.stop_notify(DATA_UUID)
                except Exception:
                    pass
                    
    except asyncio.TimeoutError:
        print("[ble] Connection timeout!")
        if platform.system() == "Windows":
            print("  💡 Windows: Try the following:")
            print("     1. Press EN button on ESP32")
            print("     2. Close and reopen Bluetooth settings")
            print("     3. Run as Administrator")
    except Exception as e:
        print(f"[ble] Connection error: {e}")
        if platform.system() == "Windows":
            print("  💡 Windows: Try disabling and re-enabling Bluetooth")


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 58)
    print("  EEG BLE Receiver — Optimized for Windows/Linux")
    print(f"  Platform : {platform.system()} {platform.release()}")
    print(f"  Sample   : {SAMPLE_SIZE} B × {BATCH_SIZE} = {BATCH_BYTES} B/notification")
    
    if platform.system() == "Windows":
        print("   Windows Optimizations: Fast discovery + Response=True")
    print("=" * 58)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"eeg_ble_{ts}.csv"
    plot_filename = f"eeg_ble_{ts}.png"

    stats = {"received": 0, "loss": 0, "corrupt": 0, "t_start": None, "fragments": 0}
    
    stop_event = threading.Event()

    csv_writer = CSVWriter(csv_filename)
    csv_writer.start()

    plot = LivePlot()

    def signal_handler(sig, frame):
        print("\n[main] Stopping...")
        stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if platform.system() == "Windows":
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            print("[windows] Using SelectorEventLoopPolicy for better performance")
        except Exception as e:
            print(f"[windows] Could not set policy: {e}")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    main_task = loop.create_task(run(csv_writer, plot, stats, stop_event))

    try:
        loop.run_until_complete(main_task)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[main] Error: {e}")
    finally:
        csv_writer.stop()
        csv_writer.join(timeout=15)

        elapsed = (time.time() - stats["t_start"]) if stats["t_start"] else 0
        print(f"\n{'=' * 58}")
        print(f"  Total samples  : {stats['received']}")
        print(f"  Packet loss    : {stats['loss']}")
        print(f"  Corrupt frames : {stats['corrupt']}")
        print(f"  Fragments      : {stats.get('fragments', 0)}")
        print(f"  Duration       : {elapsed:.1f} s")
        if elapsed > 0:
            print(f"  Effective rate : {stats['received'] / elapsed:.1f} samples/s")
        print(f"  CSV file       : {csv_filename}")
        print(f"{'=' * 58}")

        plot.save_final(plot_filename)
        plot.close()
        
        loop.close()
        print("[main] Done.")


if __name__ == "__main__":
    main()
