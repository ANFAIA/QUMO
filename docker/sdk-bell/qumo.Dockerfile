# Production-style QUMO image: build/qumo compiled with the pinned Mojo toolchain, then only the binary and the
# shared libraries it links in a scratch image, run as a non-root user. Build from the repo root:
#   docker build -f docker/sdk-bell/qumo.Dockerfile -t sdk-bell-qumo .
FROM python:3.12-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "mojo==1.2.0" "max-mojo-libs==26.7.0" "max-core==26.7.0"
WORKDIR /qumo
COPY qumo ./qumo
COPY cli ./cli
RUN mkdir build && mojo build -I . -o build/qumo cli/main.mojo
RUN mkdir -p /rootfs/usr/local/bin /rootfs/usr/local/lib /rootfs/tmp /rootfs/app && chmod 1777 /rootfs/tmp \
 && cp build/qumo /rootfs/usr/local/bin/qumo \
 && cd /usr/local/lib/python3.12/site-packages/modular/lib \
 && cp libKGENCompilerRTShared.so libAsyncRTMojoBindings.so libMSupportGlobals.so libAsyncRTRuntimeGlobals.so /rootfs/usr/local/lib/ \
 && (ldd /qumo/build/qumo; ldd /rootfs/usr/local/lib/*.so) 2>/dev/null \
    | awk '$3 ~ /^\// {print $3} $1 ~ /^\/lib/ {print $1}' | sort -u | grep -v '/modular/lib/' \
    | while read l; do cp -L --parents "$l" /rootfs; done
COPY docker/sdk-bell/bell.qasm /rootfs/app/bell.qasm
RUN chmod -R a+rX /rootfs

FROM scratch
COPY --from=build /rootfs/ /
ENV PATH=/usr/local/bin LD_LIBRARY_PATH=/usr/local/lib
USER 65532:65532
WORKDIR /app
CMD ["qumo", "transpile", "bell.qasm", "--target", "line:2/rz,sx,x,cx", "--level", "2", "--qasm"]
