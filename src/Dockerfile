# Five Blind Seconds - Samsung PRISM XR Hackathon 2026, Track 03 (IMU-only trajectory estimation)
# Python + NumPy only, one CPU thread. Build:  docker build -t fiveblindseconds .
# Run on an input in the current folder:      docker run --rm -v "$PWD":/data fiveblindseconds input.zip estimated_trajectory.txt
FROM python:3.12-slim
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY main.py /app/main.py
WORKDIR /data
ENTRYPOINT ["python", "/app/main.py"]
