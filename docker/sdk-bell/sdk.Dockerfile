# Production-style image of one Python SDK: only the interpreter, the SDK with its dependencies and the task, in a
# scratch image, run as a non-root user. Build from this directory:
#   docker build -f sdk.Dockerfile --build-arg PKGS=qiskit==2.6.1 --build-arg TASK=task_qiskit.py -t sdk-bell-qiskit .
FROM python:3.12-slim AS build
ARG PKGS
ARG TASK
RUN pip install --no-cache-dir ${PKGS}
# drop what a running container never needs: installers, tests, type stubs, the stdlib's test suite and GUI parts
RUN pip uninstall -y pip setuptools wheel >/dev/null 2>&1; \
    cd /usr/local/lib/python3.12 && rm -rf test idlelib tkinter turtledemo ensurepip pydoc_data turtle.py \
        lib-dynload/_tkinter* lib-dynload/_test* lib-dynload/xxlimited* \
 && find site-packages \( -name tests -o -name test \) -type d -prune -exec rm -rf {} + \
 && find site-packages -name '*.pyi' -delete
# matplotlib (a Cirq dependency) writes a font cache on first import: build it now, ship it read-only
ENV MPLCONFIGDIR=/mpl
RUN mkdir /mpl && (python -c "import matplotlib.pyplot" 2>/dev/null || true)
# root filesystem: python, its libraries, and the shared libraries they link
RUN mkdir -p /rootfs/usr/local/bin /rootfs/usr/local/lib /rootfs/tmp && chmod 1777 /rootfs/tmp \
 && cp -a /usr/local/lib/python3.12 /usr/local/lib/libpython3.12.so* /rootfs/usr/local/lib/ \
 && cp -a /usr/local/bin/python3.12 /rootfs/usr/local/bin/python && ln -s python /rootfs/usr/local/bin/python3 \
 && cp -a /mpl /rootfs/mpl && chown -R 65532:65532 /rootfs/mpl \
 && find /usr/local -type f \( -name '*.so' -o -name '*.so.*' -o -name python3.12 \) -exec ldd {} \; 2>/dev/null \
    | awk '$3 ~ /^\// {print $3} $1 ~ /^\/lib/ {print $1}' | sort -u | grep -v '^/usr/local' \
    | while read l; do cp -L --parents "$l" /rootfs; done \
 && chmod -R a+rX /rootfs

FROM scratch
COPY --from=build /rootfs/ /
ARG TASK
COPY ${TASK} /app/task.py
ENV PATH=/usr/local/bin LD_LIBRARY_PATH=/usr/local/lib PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/mpl HOME=/tmp
USER 65532:65532
WORKDIR /app
CMD ["python", "task.py"]
