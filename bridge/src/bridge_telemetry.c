/*
 * bridge_telemetry.c — ADB/USB-C heartbeat between Beast Canyon and S22 Ultra
 *
 * Architecture:
 *   - Desktop opens a persistent TCP connection to localhost:PORT
 *     (ADB forwards this to the phone daemon via USB-C)
 *   - A dedicated thread sends PING every interval_ms
 *   - If PONG doesn't arrive within timeout_ms, sidecar_responsive → false
 *   - On state change (alive→dead or dead→alive), fires the callback
 *   - On disconnect, desktop reclaims speculative draft tasks locally
 */

#include "bridge_telemetry.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")

typedef SOCKET sock_t;
#define SOCK_INVALID INVALID_SOCKET
#define sock_close(s) closesocket(s)

#else
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <errno.h>
#include <pthread.h>
#include <sys/time.h>

typedef int sock_t;
#define SOCK_INVALID (-1)
#define sock_close(s) close(s)
#endif

/* ── Wire protocol ────────────────────────────────────────────────── */

#define BRIDGE_MAGIC_PING  0x474E5042  /* "BPNG" little-endian */
#define BRIDGE_MAGIC_PONG  0x4B4F5042  /* "BPOK" little-endian */

typedef struct __attribute__((packed)) {
    uint32_t magic;
} bridge_ping_t;

typedef struct __attribute__((packed)) {
    uint32_t magic;
    uint64_t phone_timestamp_us;  /* phone's monotonic clock */
} bridge_pong_t;


/* ── Internal context ─────────────────────────────────────────────── */

struct bridge_ctx {
    /* Config snapshot */
    uint16_t port;
    uint32_t interval_ms;
    uint32_t timeout_ms;

    /* State (accessed atomically) */
    volatile bool alive;
    volatile uint64_t last_rtt_us;
    volatile bool running;

    /* Callback */
    bridge_state_cb on_change;
    void           *userdata;

    /* Thread */
#ifdef _WIN32
    HANDLE thread;
#else
    pthread_t thread;
#endif

    /* Socket (owned by heartbeat thread) */
    sock_t sock;
};


/* ── Timing helpers ───────────────────────────────────────────────── */

static uint64_t now_us(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (uint64_t)(cnt.QuadPart * 1000000ULL / freq.QuadPart);
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000ULL + tv.tv_usec;
#endif
}

static void sleep_ms(uint32_t ms) {
#ifdef _WIN32
    Sleep(ms);
#else
    usleep(ms * 1000);
#endif
}


/* ── Socket helpers ───────────────────────────────────────────────── */

static sock_t connect_to_bridge(uint16_t port, uint32_t timeout_ms) {
#ifdef _WIN32
    sock_t s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s == SOCK_INVALID) return SOCK_INVALID;

    /* Set send/recv timeout */
    DWORD tv = timeout_ms;
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char *)&tv, sizeof(tv));
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, (const char *)&tv, sizeof(tv));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

    if (connect(s, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        sock_close(s);
        return SOCK_INVALID;
    }
    return s;
#else
    sock_t s = socket(AF_INET, SOCK_STREAM, 0);
    if (s < 0) return SOCK_INVALID;

    struct timeval tv;
    tv.tv_sec  = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

    if (connect(s, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        sock_close(s);
        return SOCK_INVALID;
    }
    return s;
#endif
}

static bool do_ping(sock_t s, uint32_t timeout_ms, uint64_t *rtt_out) {
    bridge_ping_t ping = { .magic = BRIDGE_MAGIC_PING };
    bridge_pong_t pong;

    uint64_t t0 = now_us();

    /* Send PING */
    int sent = send(s, (const char *)&ping, sizeof(ping), 0);
    if (sent != sizeof(ping)) return false;

    /* Receive PONG */
    int total = 0;
    while (total < (int)sizeof(pong)) {
        int n = recv(s, ((char *)&pong) + total, sizeof(pong) - total, 0);
        if (n <= 0) return false;
        total += n;
    }

    uint64_t t1 = now_us();

    if (pong.magic != BRIDGE_MAGIC_PONG) return false;

    *rtt_out = t1 - t0;
    (void)timeout_ms;
    return true;
}


/* ── Heartbeat thread ─────────────────────────────────────────────── */

static
#ifdef _WIN32
DWORD WINAPI
#else
void *
#endif
heartbeat_loop(void *arg) {
    bridge_ctx_t *ctx = (bridge_ctx_t *)arg;

    while (ctx->running) {
        /* (Re)connect if needed */
        if (ctx->sock == SOCK_INVALID) {
            ctx->sock = connect_to_bridge(ctx->port, ctx->timeout_ms);
            if (ctx->sock == SOCK_INVALID) {
                /* Can't connect — mark dead */
                if (ctx->alive) {
                    ctx->alive = false;
                    fprintf(stderr, "[bridge] Sidecar LOST — reclaiming speculative tasks\n");
                    if (ctx->on_change) ctx->on_change(false, ctx->userdata);
                }
                sleep_ms(ctx->interval_ms);
                continue;
            }
            fprintf(stderr, "[bridge] Connected to sidecar on port %u\n", ctx->port);
        }

        /* Send ping */
        uint64_t rtt = 0;
        bool ok = do_ping(ctx->sock, ctx->timeout_ms, &rtt);

        if (ok) {
            ctx->last_rtt_us = rtt;
            if (!ctx->alive) {
                ctx->alive = true;
                fprintf(stderr, "[bridge] Sidecar ALIVE (rtt=%llu us)\n",
                        (unsigned long long)rtt);
                if (ctx->on_change) ctx->on_change(true, ctx->userdata);
            }
        } else {
            /* Ping failed — tear down socket, mark dead */
            sock_close(ctx->sock);
            ctx->sock = SOCK_INVALID;
            if (ctx->alive) {
                ctx->alive = false;
                fprintf(stderr, "[bridge] Sidecar LOST — reclaiming speculative tasks\n");
                if (ctx->on_change) ctx->on_change(false, ctx->userdata);
            }
        }

        sleep_ms(ctx->interval_ms);
    }

    if (ctx->sock != SOCK_INVALID) {
        sock_close(ctx->sock);
        ctx->sock = SOCK_INVALID;
    }

#ifdef _WIN32
    return 0;
#else
    return NULL;
#endif
}


/* ── Public API ───────────────────────────────────────────────────── */

bridge_ctx_t *bridge_start(const burnhard_config_t *cfg,
                           bridge_state_cb on_change,
                           void *userdata)
{
#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2,2), &wsa);
#endif

    bridge_ctx_t *ctx = (bridge_ctx_t *)calloc(1, sizeof(*ctx));
    if (!ctx) return NULL;

    ctx->port        = cfg->bridge_port;
    ctx->interval_ms = cfg->heartbeat_interval_ms;
    ctx->timeout_ms  = cfg->heartbeat_timeout_ms;
    ctx->on_change   = on_change;
    ctx->userdata    = userdata;
    ctx->sock        = SOCK_INVALID;
    ctx->alive       = false;
    ctx->running     = true;

    /* First: set up ADB forwarding */
    if (bridge_adb_forward(ctx->port) != 0) {
        fprintf(stderr, "[bridge] Warning: adb forward failed (is ADB running?)\n");
        /* Non-fatal — maybe forwarding is already set up */
    }

#ifdef _WIN32
    ctx->thread = CreateThread(NULL, 0, heartbeat_loop, ctx, 0, NULL);
    if (!ctx->thread) { free(ctx); return NULL; }
#else
    if (pthread_create(&ctx->thread, NULL, heartbeat_loop, ctx) != 0) {
        free(ctx);
        return NULL;
    }
#endif

    fprintf(stderr, "[bridge] Heartbeat started (port=%u, interval=%ums, timeout=%ums)\n",
            ctx->port, ctx->interval_ms, ctx->timeout_ms);
    return ctx;
}

bool bridge_is_alive(const bridge_ctx_t *ctx) {
    return ctx ? ctx->alive : false;
}

uint64_t bridge_last_rtt_us(const bridge_ctx_t *ctx) {
    return ctx ? ctx->last_rtt_us : 0;
}

void bridge_stop(bridge_ctx_t *ctx) {
    if (!ctx) return;

    ctx->running = false;

#ifdef _WIN32
    WaitForSingleObject(ctx->thread, 5000);
    CloseHandle(ctx->thread);
    WSACleanup();
#else
    pthread_join(ctx->thread, NULL);
#endif

    bridge_adb_unforward(ctx->port);
    fprintf(stderr, "[bridge] Stopped.\n");
    free(ctx);
}


/* ── ADB helpers ──────────────────────────────────────────────────── */

int bridge_adb_forward(uint16_t port) {
    char cmd[128];
    snprintf(cmd, sizeof(cmd), "adb forward tcp:%u tcp:%u", port, port);
    int rc = system(cmd);
    if (rc == 0) {
        fprintf(stderr, "[bridge] ADB forward tcp:%u → tcp:%u OK\n", port, port);
    }
    return rc;
}

int bridge_adb_unforward(uint16_t port) {
    char cmd[128];
    snprintf(cmd, sizeof(cmd), "adb forward --remove tcp:%u", port);
    return system(cmd);
}
