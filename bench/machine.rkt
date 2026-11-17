#lang racket/base

;; Host facts the benchmark records. The CPU name prefers Linux
;; /proc/cpuinfo and falls back to the macOS sysctl brand string, so a
;; run on either machine names the processor.

(require racket/future
         racket/port
         racket/runtime-path
         racket/string)

(provide cpu-name-from
         host-cpu-name
         host-physical-cpu
         host-ncpu
         git-sha-or-unknown
         host-git-sha
         format-machine-line)

(define-runtime-path repo-root "..")

(define (nonempty text)
  (and (string? text)
       (let ([trimmed (string-trim text)])
         (and (not (string=? trimmed "")) trimmed))))

;; proc-name wins when Linux printed one. Otherwise the macOS brand
;; string is the name. An empty reading from either source is no name.
(define (cpu-name-from proc-name sysctl-name)
  (or (nonempty proc-name) (nonempty sysctl-name)))

;; macOS keeps sysctl in /usr/sbin, which is missing from some PATH values.
(define (sysctl-executable)
  (or (find-executable-path "sysctl")
      (and (file-exists? "/usr/sbin/sysctl") "/usr/sbin/sysctl")))

(define (sysctl-n key)
  (with-handlers ([exn:fail? (lambda (_) #f)])
    (define exe (sysctl-executable))
    (and exe
         (let-values ([(proc out in err)
                       (subprocess #f #f #f exe "-n" key)])
           (close-output-port in)
           (define text (port->string out))
           (close-input-port out)
           (close-input-port err)
           (subprocess-wait proc)
           (and (zero? (subprocess-status proc))
                (nonempty text))))))

(define (cpuinfo-field line key)
  (and (string-prefix? line key)
       (let ([parts (string-split line ":" #:trim? #f)])
         (and (= (length parts) 2) (nonempty (cadr parts))))))

;; x86 Linux prints "model name". aarch64 usually does not: it may print
;; "Model" or "Hardware", and otherwise the architecture is the name.
(define (cpu-from-proc)
  (with-handlers ([exn:fail? (lambda (_) #f)])
    (and (file-exists? "/proc/cpuinfo")
         (call-with-input-file "/proc/cpuinfo"
           (lambda (in)
             (for/fold ([found #f]) ([line (in-lines in)] #:when (not found))
               (or (cpuinfo-field line "model name")
                   (cpuinfo-field line "Model")
                   (cpuinfo-field line "Hardware"))))))))

(define (host-cpu-name)
  (or (cpu-name-from (cpu-from-proc) (sysctl-n "machdep.cpu.brand_string"))
      (symbol->string (system-type 'arch))))

;; Performance-core count on Apple Silicon. Absent on Linux and on Intel Macs.
(define (host-physical-cpu)
  (sysctl-n "hw.perflevel0.physicalcpu"))

(define (host-ncpu)
  (sysctl-n "hw.ncpu"))

;; A missing git is the string "unknown". Returning #f becomes JSON false.
(define (git-sha-or-unknown exe ok? text)
  (if (and exe ok?)
      (or (nonempty text) "unknown")
      "unknown"))

(define (host-git-sha)
  (with-handlers ([exn:fail? (lambda (_) "unknown")])
    (define git (find-executable-path "git"))
    (define root (path->string repo-root))
    (if (not git)
        "unknown"
        (let-values ([(proc out in err)
                      (subprocess #f #f #f git "-C" root "rev-parse" "HEAD")])
          (close-output-port in)
          (define text (port->string out))
          (close-input-port err)
          (close-input-port out)
          (subprocess-wait proc)
          (git-sha-or-unknown git (zero? (subprocess-status proc)) text)))))

(define (format-machine-line)
  (format "machine: os=~a arch=~a vm=~a processors=~a cpu=~a physical-cpu=~a ncpu=~a"
          (system-type 'os)
          (system-type 'arch)
          (system-type 'vm)
          (processor-count)
          (host-cpu-name)
          (or (host-physical-cpu) "unavailable")
          (or (host-ncpu) "unavailable")))
