FROM heat-env
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends python3-venv python3-dev libgtest-dev xvfb xauth squashfs-tools curl libxcb-xinerama0 libxcb-cursor0 libxkbcommon-x11-0 libegl1 libopengl0 libxcb-icccm4 libxcb-keysyms1 libxcb-shape0 libdbus-1-3 libcups2t64 libwayland-client0 libwayland-cursor0 libwayland-egl1 libxcomposite1 libxrender1 libxi6 libxrandr2 libxcursor1 && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/heat-python
ENV PATH="/opt/heat-python/bin:${PATH}"
COPY packaging/linux/requirements.txt /tmp/heat-requirements.txt
RUN pip install --no-cache-dir -r /tmp/heat-requirements.txt
WORKDIR /workspace
