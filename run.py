import queue
import threading
from dataclasses import dataclass

import cv2
import tkinter as tk
from tkinter import ttk

from perception import GestureColorMapping, GESTURES, COLOR_NAMES
from perception import GesturePerception
from simulation import RobotArmSimulation


@dataclass
class AppConfig:
    cfg_path: str = 'cfg.yml'
    stable_frames: int = 8


def get_user_mapping():
    result = {'mapping': None}

    root = tk.Tk()
    root.title('Gesture to Color Mapping')
    root.geometry('430x270')
    root.resizable(False, False)

    tk.Label(root, text='Assign a different gesture to each color', font=('Arial', 13, 'bold')).pack(pady=12)
    frame = tk.Frame(root)
    frame.pack(pady=8)

    defaults = {'red': 'Open_Palm', 'green': 'Closed_Fist', 'blue': 'Thumb_Up'}
    vars_map = {}

    for i, color in enumerate(COLOR_NAMES):
        tk.Label(frame, text=f'{color.capitalize()}:', width=10, anchor='e', font=('Arial', 11)).grid(row=i, column=0, padx=8, pady=8)
        var = tk.StringVar(value=defaults[color])
        combo = ttk.Combobox(frame, textvariable=var, values=GESTURES, state='readonly', width=18)
        combo.grid(row=i, column=1, padx=8, pady=8)
        vars_map[color] = var

    error_label = tk.Label(root, text='', fg='red', font=('Arial', 10, 'bold'))
    error_label.pack(pady=6)

    def validate_selection(*args):
        chosen = [vars_map[c].get() for c in COLOR_NAMES]
        if len(set(chosen)) != len(chosen):
            error_label.config(text='Error: one gesture cannot be assigned to multiple colors.')
            return False
        error_label.config(text='')
        return True

    for color in COLOR_NAMES:
        vars_map[color].trace_add('write', validate_selection)

    def on_start():
        chosen = [vars_map[c].get() for c in COLOR_NAMES]
        if len(set(chosen)) != len(chosen):
            error_label.config(text='Error: one gesture cannot be assigned to multiple colors.')
            return
        result['mapping'] = GestureColorMapping(
            red=vars_map['red'].get(),
            green=vars_map['green'].get(),
            blue=vars_map['blue'].get(),
        )
        root.destroy()

    tk.Button(root, text='Start', command=on_start, font=('Arial', 11), width=18).pack(pady=12)
    root.mainloop()
    return result['mapping']


def main():
    app_cfg = AppConfig()
    mapping = get_user_mapping()
    if mapping is None:
        print('No mapping selected. Exiting.')
        return

    perception = GesturePerception(mapping=mapping, stable_frames=app_cfg.stable_frames)
    simulation = RobotArmSimulation(cfg_path=app_cfg.cfg_path, use_gui=True)
    task_queue = queue.Queue()
    stop_event = threading.Event()

    def robot_worker():
        while not stop_event.is_set():
            try:
                color = task_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if color is None:
                break
            if not simulation.busy:
                simulation.pick_and_place(color)

    worker = threading.Thread(target=robot_worker, daemon=True)
    worker.start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        perception.close()
        simulation.close()
        raise RuntimeError('Could not open webcam.')

    print('Running full pipeline. Press q or ESC to quit.')
    print(f'  red   -> {mapping.red}')
    print(f'  green -> {mapping.green}')
    print(f'  blue  -> {mapping.blue}')

    try:
        while True:
            success, frame = cap.read()
            if not success:
                continue

            frame, detected_gesture, detected_color, triggered_color = perception.process_frame(frame)

            if simulation.busy:
                cv2.putText(frame, 'Robot status: BUSY', (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, 'Robot status: READY', (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            if triggered_color is not None and not simulation.busy:
                task_queue.put(triggered_color)
                cv2.putText(frame, f'Triggered task: {triggered_color}', (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow('Gesture Perception Pipeline', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    finally:
        stop_event.set()
        task_queue.put(None)
        cap.release()
        cv2.destroyAllWindows()
        perception.close()
        simulation.close()


if __name__ == '__main__':
    main()
