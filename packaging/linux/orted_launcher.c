/* Open MPI execs this helper with the AppImage's LD_LIBRARY_PATH inherited.
 * A shell wrapper would itself load the host's ELF interpreter and bundled
 * libc before reaching its first command. Keep this trampoline static and
 * start the real daemon with the matching bundled interpreter explicitly. */
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char **argv) {
    char executable[PATH_MAX + 1];
    const ssize_t length = readlink("/proc/self/exe", executable, PATH_MAX);
    if (length < 0 || length >= PATH_MAX) {
        fputs("orted launcher: cannot locate application resources\n", stderr);
        return 127;
    }
    executable[length] = '\0';
    char *separator = strrchr(executable, '/');
    if (!separator)
        return 127;
    *separator = '\0'; /* usr/bin */
    separator = strrchr(executable, '/');
    if (!separator)
        return 127;
    *separator = '\0'; /* usr */

    char loader[PATH_MAX], libraries[PATH_MAX], daemon[PATH_MAX];
    if (snprintf(loader, sizeof(loader), "%s/lib/ld-linux-x86-64.so.2", executable) >=
            (int)sizeof(loader) ||
        snprintf(libraries, sizeof(libraries), "%s/lib", executable) >=
            (int)sizeof(libraries) ||
        snprintf(daemon, sizeof(daemon), "%s/bin/orted.real", executable) >=
            (int)sizeof(daemon)) {
        fputs("orted launcher: application path is too long\n", stderr);
        return 127;
    }
    char **arguments = calloc((size_t)argc + 5, sizeof(*arguments));
    if (!arguments) {
        fputs("orted launcher: cannot allocate arguments\n", stderr);
        return 127;
    }
    arguments[0] = loader;
    arguments[1] = "--inhibit-cache";
    arguments[2] = "--library-path";
    arguments[3] = libraries;
    arguments[4] = daemon;
    for (int index = 1; index < argc; ++index)
        arguments[index + 4] = argv[index];
    execv(loader, arguments);
    fprintf(stderr, "orted launcher: cannot start bundled daemon: %s\n", strerror(errno));
    free(arguments);
    return 127;
}
