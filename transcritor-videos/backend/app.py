import os
import uuid
import subprocess
import tempfile
import zipfile
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from faster_whisper import WhisperModel
import imageio_ffmpeg

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = tempfile.gettempdir()
MODEL_NAME = os.environ.get("WHISPER_MODEL", "tiny")

print(f"Carregando modelo Whisper '{MODEL_NAME}'...")
model = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")
print("Modelo carregado!")

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def extract_audio(video_path: str, audio_path: str) -> bool:
    try:
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        result = subprocess.run(
            [ffmpeg_bin, "-y", "-i", video_path, "-vn", "-ar", "16000",
             "-ac", "1", "-f", "wav", audio_path],
            capture_output=True, timeout=300
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Erro ao extrair áudio: {e}")
        return False


def format_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def segments_to_srt(segments) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        start = format_srt_time(seg.start)
        end = format_srt_time(seg.end)
        lines.append(f"{i}\n{start} --> {end}\n{seg.text.strip()}\n")
    return "\n".join(lines)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME})


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado"}), 400

    file = request.files["file"]
    language = request.form.get("language", "pt")
    generate_srt = request.form.get("srt", "false").lower() == "true"

    if not file.filename:
        return jsonify({"error": "Nome de arquivo inválido"}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Formato não suportado: {ext}"}), 400

    job_id = str(uuid.uuid4())
    video_path = os.path.join(UPLOAD_FOLDER, f"{job_id}{ext}")
    audio_path = os.path.join(UPLOAD_FOLDER, f"{job_id}.wav")
    txt_path = os.path.join(UPLOAD_FOLDER, f"{job_id}.txt")
    srt_path = os.path.join(UPLOAD_FOLDER, f"{job_id}.srt")
    zip_path = os.path.join(UPLOAD_FOLDER, f"{job_id}.zip")

    try:
        file.save(video_path)

        if not extract_audio(video_path, audio_path):
            return jsonify({"error": "Falha ao extrair áudio do vídeo"}), 500

        lang_param = None if language == "auto" else language
        segments, info = model.transcribe(audio_path, language=lang_param)
        segments = list(segments)

        full_text = " ".join(seg.text.strip() for seg in segments)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(full_text)

        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(txt_path, f"{Path(file.filename).stem}.txt")
            if generate_srt:
                srt_content = segments_to_srt(segments)
                with open(srt_path, "w", encoding="utf-8") as f:
                    f.write(srt_content)
                zf.write(srt_path, f"{Path(file.filename).stem}.srt")

        return send_file(
            zip_path,
            as_attachment=True,
            download_name=f"{Path(file.filename).stem}_transcricao.zip",
            mimetype="application/zip"
        )

    except Exception as e:
        print(f"Erro na transcrição: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        for path in [video_path, audio_path, txt_path, srt_path]:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except:
                pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
