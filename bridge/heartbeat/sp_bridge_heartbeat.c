// Shannon-Prime Bridge: heartbeat — implementation. See sp_bridge_heartbeat.h.

#include "sp_bridge_heartbeat.h"
#include "../protocol/sp_bridge_proto.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
typedef HANDLE sp_thread_t;
typedef volatile LONG sp_atomic_int;
typedef volatile LONG64 sp_atomic_u64;
#  define sp_atomic_load_int(p)      InterlockedCompareExchange((p), 0, 0)
#  define sp_atomic_store_int(p, v)  InterlockedExchange((p), (v))
#  define sp_atomic_inc_u64(p)       InterlockedIncrement64((p))
#  define sp_atomic_load_u64(p)      InterlockedCompareExchange64((p), 0, 0)
#else
#  include <pthread.h>
#  include <time.h>
#  include <unistd.h>
typedef pthread_t sp_thread_t;
typedef volatile int      sp_atomic_int;
typedef volatile uint64_t sp_atomic_u64;
#  define sp_atomic_load_int(p)      __atomic_load_n((p), __ATOMIC_ACQUIRE)
#  define sp_atomic_store_int(p, v)  __atomic_store_n((p), (v), __ATOMIC_RELEASE)
#  define sp_atomic_inc_u64(p)       __atomic_add_fetch((p), 1, __ATOMIC_ACQ_REL)
#  define sp_atomic_load_u64(p)      __atomic_load_n((p), __ATOMIC_ACQUIRE)
#endif

struct sp_bridge_heartbeat {
    int                       socket_fd;
    uint32_t                  period_ms;
    sp_bridge_state_cb        cb;
    void                     *cb_ctx;

    sp_thread_t               thread;
    sp_atomic_int             state;        // sp_bridge_state_t
    sp_atomic_u64             acks;
    sp_atomic_u64             misses;
    sp_atomic_int             stop_flag;

    int                       miss_streak;
    int                       ack_streak;
    uint64_t                  beat_counter;
};

const char *sp_bridge_state_name(sp_bridge_state_t s) {
    switch (s) {
        case SP_BRIDGE_OFFLINE:  return "OFFLINE";
        case SP_BRIDGE_DEGRADED: return "DEGRADED";
        case SP_BRIDGE_ONLINE:   return "ONLINE";
    }
    return "?";
}

static void sleep_ms(uint32_t ms) {
#ifdef _WIN32
    Sleep(ms);
#else
    struct timespec ts;
    ts.tv_sec  = ms / 1000;
    ts.tv_nsec = (long)(ms % 1000) * 1000000L;
    nanosleep(&ts, NULL);
#endif
}

static bool exchange_one_beat(sp_bridge_heartbeat_t *hb) {
    uint64_t sent = ++hb->beat_counter;
    if (sp_sidecar_send(hb->socket_fd, SP_CMD_BRIDGE_HEARTBEAT,
                        &sent, sizeof(sent)) != 0) {
        return false;
    }
    void    *payload = NULL;
    uint32_t plen    = 0;
    int status = sp_sidecar_recv_msg(hb->socket_fd, &payload, &plen);
    bool ok = false;
    if (status == SP_RESP_OK && plen == sizeof(uint64_t) && payload) {
        uint64_t echoed;
        memcpy(&echoed, payload, sizeof(echoed));
        ok = (echoed == sent);
    }
    if (payload) free(payload);
    return ok;
}

static void publish_state(sp_bridge_heartbeat_t *hb, sp_bridge_state_t next) {
    sp_bridge_state_t prev = (sp_bridge_state_t)sp_atomic_load_int(&hb->state);
    if (prev == next) return;
    sp_atomic_store_int(&hb->state, (int)next);
    fprintf(stderr,
            "[bridge] heartbeat state: %s -> %s (acks=%llu, misses=%llu)\n",
            sp_bridge_state_name(prev),
            sp_bridge_state_name(next),
            (unsigned long long)sp_atomic_load_u64(&hb->acks),
            (unsigned long long)sp_atomic_load_u64(&hb->misses));
    if (hb->cb) hb->cb(next, hb->cb_ctx);
}

#ifdef _WIN32
static DWORD WINAPI heartbeat_loop(LPVOID arg) {
#else
static void *heartbeat_loop(void *arg) {
#endif
    sp_bridge_heartbeat_t *hb = (sp_bridge_heartbeat_t *)arg;
    while (!sp_atomic_load_int(&hb->stop_flag)) {
        bool ok = exchange_one_beat(hb);
        if (ok) {
            sp_atomic_inc_u64(&hb->acks);
            hb->miss_streak = 0;
            sp_bridge_state_t cur =
                (sp_bridge_state_t)sp_atomic_load_int(&hb->state);
            if (cur != SP_BRIDGE_ONLINE) {
                if (++hb->ack_streak >= 3) {
                    publish_state(hb, SP_BRIDGE_ONLINE);
                    hb->ack_streak = 0;
                }
            } else {
                hb->ack_streak = 0;
            }
        } else {
            sp_atomic_inc_u64(&hb->misses);
            hb->ack_streak = 0;
            ++hb->miss_streak;
            sp_bridge_state_t cur =
                (sp_bridge_state_t)sp_atomic_load_int(&hb->state);
            if (cur == SP_BRIDGE_ONLINE && hb->miss_streak >= 2) {
                publish_state(hb, SP_BRIDGE_DEGRADED);
            }
            if (cur != SP_BRIDGE_OFFLINE && hb->miss_streak >= 5) {
                publish_state(hb, SP_BRIDGE_OFFLINE);
            }
        }
        sleep_ms(hb->period_ms);
    }
#ifdef _WIN32
    return 0;
#else
    return NULL;
#endif
}

sp_bridge_heartbeat_t *sp_bridge_heartbeat_start(int socket_fd,
                                                 uint32_t period_ms,
                                                 sp_bridge_state_cb cb,
                                                 void *cb_ctx) {
    if (socket_fd < 0) return NULL;
    if (period_ms == 0) period_ms = 1000;
    sp_bridge_heartbeat_t *hb = (sp_bridge_heartbeat_t *)calloc(1, sizeof(*hb));
    if (!hb) return NULL;
    hb->socket_fd = socket_fd;
    hb->period_ms = period_ms;
    hb->cb        = cb;
    hb->cb_ctx    = cb_ctx;
    sp_atomic_store_int(&hb->state, SP_BRIDGE_ONLINE);

#ifdef _WIN32
    hb->thread = CreateThread(NULL, 0, heartbeat_loop, hb, 0, NULL);
    if (!hb->thread) { free(hb); return NULL; }
#else
    if (pthread_create(&hb->thread, NULL, heartbeat_loop, hb) != 0) {
        free(hb);
        return NULL;
    }
#endif
    return hb;
}

sp_bridge_state_t sp_bridge_heartbeat_state(const sp_bridge_heartbeat_t *hb) {
    if (!hb) return SP_BRIDGE_OFFLINE;
    return (sp_bridge_state_t)sp_atomic_load_int(
        (sp_atomic_int *)&hb->state);
}

uint64_t sp_bridge_heartbeat_acks(const sp_bridge_heartbeat_t *hb) {
    if (!hb) return 0;
    return sp_atomic_load_u64((sp_atomic_u64 *)&hb->acks);
}

uint64_t sp_bridge_heartbeat_misses(const sp_bridge_heartbeat_t *hb) {
    if (!hb) return 0;
    return sp_atomic_load_u64((sp_atomic_u64 *)&hb->misses);
}

void sp_bridge_heartbeat_stop(sp_bridge_heartbeat_t *hb) {
    if (!hb) return;
    sp_atomic_store_int(&hb->stop_flag, 1);
#ifdef _WIN32
    WaitForSingleObject(hb->thread, INFINITE);
    CloseHandle(hb->thread);
#else
    pthread_join(hb->thread, NULL);
#endif
    free(hb);
}
