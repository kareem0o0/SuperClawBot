# minimal_audio_classifier.py
import numpy as np, sounddevice as sd, tflite_runtime.interpreter as tflite, time, os

# === CONFIG ===
MODEL = "sound_classifier/soundclassifier_with_metadata.tflite"
LABELS = "sound_classifier/labels.txt"
RATE = 44100
OVERLAP = 0.5

# === LOAD MODEL & LABELS ===
interp = tflite.Interpreter(MODEL)
interp.allocate_tensors()
inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]
samples = inp['shape'][1]  # e.g., 44032

with open(LABELS) as f:
    labels = [l.strip() for l in f]

print(f"Model expects {samples} samples → {samples/RATE:.2f}s")
print(f"Labels: {', '.join(labels[:5])}{'...' if len(labels)>5 else ''}\n")

# === AUDIO CALLBACK ===
buf = np.zeros(samples, dtype=np.float32)
pos = 0

def callback(indata, frames, time, status):
    global pos, buf
    chunk = indata[:, 0].copy()
    buf[pos:pos+len(chunk)] = chunk[:samples-pos]
    pos += len(chunk)
    if pos >= samples:
        # Normalize
        audio = buf.copy()
        maxv = np.max(np.abs(audio))
        if maxv > 0: audio /= maxv
        
        # Reshape & run
        x = audio.reshape(1, samples).astype(np.float32)
        interp.set_tensor(inp['index'], x)
        interp.invoke()
        scores = interp.get_tensor(out['index'])[0]
        i = np.argmax(scores)
        print(f"→ {labels[i]:12} ({scores[i]:.1%})")
        
        # Slide with overlap
        shift = int(samples * (1 - OVERLAP))
        buf[:shift] = buf[-shift:]
        pos = shift

# === START STREAM ===
with sd.InputStream(samplerate=RATE, channels=1, dtype='float32', blocksize=int(RATE*0.1), callback=callback):
    print("LISTENING... (Ctrl+C to stop)\n")
    try: time.sleep(999999)
    except: print("\nStopped.")