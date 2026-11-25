/* measure CMD [ARGS...] [--then-- CMD2 ...]: run the commands one after the other, then print one METRIC line with wall time, CPU time (user + system) and peak
   resident memory (ru_maxrss) of CMD and everything it spawned. Built static and bind-mounted into a container,
   so it is not part of any image. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static double ms(struct timeval v) { return v.tv_sec * 1e3 + v.tv_usec / 1e3; }

int main(int argc, char **argv) {
    if (argc < 2) return 2;
    double t0 = now(), user = 0, sys = 0, rss = 0;
    int rc = 0, i = 1;
    while (i < argc) {
        int j = i;
        while (j < argc && strcmp(argv[j], "--then--") != 0) j++;
        argv[j] = NULL; /* end of this command */
        pid_t pid = fork();
        if (pid == 0) { execvp(argv[i], argv + i); perror(argv[i]); _exit(127); }
        int st; struct rusage ru;
        wait4(pid, &st, 0, &ru);
        user += ms(ru.ru_utime); sys += ms(ru.ru_stime);
        if (ru.ru_maxrss / 1024.0 > rss) rss = ru.ru_maxrss / 1024.0;
        rc = WIFEXITED(st) ? WEXITSTATUS(st) : 128;
        if (rc) break;
        i = j + 1;
    }
    double t1 = now();
    fprintf(stderr, "METRIC rc=%d wall_ms=%.1f cpu_ms=%.1f user_ms=%.1f sys_ms=%.1f rss_mib=%.1f\n", rc, t1 - t0, user + sys, user, sys, rss);
    return 0;
}
