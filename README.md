# Pool Guard (Edge AI Pool Safety)

Offline çalışan, RTSP/MP4 üzerinden havuz çevresinde **tehlikeli zone içine giren çocuk** durumunda alarm üreten Edge AI pipeline.

## Hızlı Başlangıç (PC)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements/dev.txt

# stub mode (modelsiz çalışır)
python -m pool_guard --config configs/default.yaml --stub
