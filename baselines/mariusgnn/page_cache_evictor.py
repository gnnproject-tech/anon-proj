import sys, os, glob, time

interval = float(sys.argv[1])
pats = sys.argv[2:]
n = 0
while True:
    for pat in pats:
        for f in glob.glob(pat):
            try:
                fd = os.open(f, os.O_RDONLY)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                os.close(fd)
                n += 1
            except OSError:
                pass
    time.sleep(interval)
