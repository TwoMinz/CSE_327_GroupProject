# PyTorch 공식 이미지(CUDA 12.1 지원) 사용
FROM  pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필요한 도구 설치
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt 파일 복사
COPY requirements.txt .

# Python 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# 권한 설정
RUN mkdir -p /app/data /app/models /app/outputs
RUN chmod -R 777 /app

# 컨테이너 시작 시 실행할 명령 (기본적으로 bash를 실행)
CMD ["/bin/bash"]