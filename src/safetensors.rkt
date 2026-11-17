#lang racket/base

;; Safetensors: little-endian header length, JSON header, then F32 payload.

(require json
         racket/flonum
         racket/port
         racket/unsafe/ops
         "errors.rkt"
         "tensor.rkt")

(provide read-safetensors
         safetensors-entry-name)

(define (read-exact-integer n)
  (cond
    [(exact-nonnegative-integer? n) n]
    [(and (real? n) (integer? n) (>= n 0)) (inexact->exact n)]
    [else #f]))

;; Decode a checked byte range; the F32 payload may be unaligned.
(define (f32-flvector bs start n)
  (define out (make-flvector n))
  (for ([i (in-range n)])
    (define base (unsafe-fx+ start (unsafe-fx* i 4)))
    (unsafe-flvector-set! out i
                          (floating-point-bytes->real bs #f base (unsafe-fx+ base 4))))
  out)

(define (safetensors-entry-name key)
  (if (symbol? key) (symbol->string key) key))

;; Skip mask buffers before decoding. Accept unaligned or space-padded headers.
(define (read-safetensors path #:skip [skip (lambda (name) #f)])
  (define bs
    (with-handlers ([exn:fail:gpt2? (lambda (e) (raise e))]
                    [exn:fail? (lambda (e)
                                 (raise-gpt2 (format "could not read ~a: ~a" path (exn-message e))))])
      (call-with-input-file path port->bytes #:mode 'binary)))
  (when (< (bytes-length bs) 8)
    (raise-gpt2 (format "safetensors file is too small: ~a" path)))
  (define header-len (integer-bytes->integer bs #f #f 0 8))
  (define header-end (+ 8 header-len))
  (when (or (< header-len 2) (> header-end (bytes-length bs)))
    (raise-gpt2 (format "safetensors header length is out of range: ~a" path)))
  (define header
    (with-handlers ([exn:fail:gpt2? (lambda (e) (raise e))]
                    [exn:fail? (lambda (e)
                                 (raise-gpt2 (format "safetensors header is not JSON: ~a" (exn-message e))))])
      (read-json (open-input-bytes (subbytes bs 8 header-end)))))
  (unless (hash? header)
    (raise-gpt2 (format "safetensors header is not an object: ~a" path)))
  (define data-base header-end)
  (for/fold ([table (hash)])
            ([(key info) (in-hash header)])
    (define name (safetensors-entry-name key))
    (cond
      [(or (equal? name "__metadata__") (skip name)) table]
      [else
       (unless (hash? info)
         (raise-gpt2 (format "safetensors entry is not an object: ~a" name)))
       (define dtype (hash-ref info 'dtype #f))
       (unless (equal? dtype "F32")
         (raise-gpt2 (format "safetensors dtype ~a is not F32: ~a" dtype name)))
       (define shape (hash-ref info 'shape #f))
       (define offsets (hash-ref info 'data_offsets #f))
       (unless (and (list? shape) (pair? shape) (andmap read-exact-integer shape))
         (raise-gpt2 (format "safetensors shape is invalid: ~a" name)))
       (unless (and (list? offsets) (= (length offsets) 2))
         (raise-gpt2 (format "safetensors offsets are invalid: ~a" name)))
       (define start (read-exact-integer (car offsets)))
       (define end (read-exact-integer (cadr offsets)))
       (unless (and start end (<= 0 start end) (<= (+ data-base end) (bytes-length bs)))
         (raise-gpt2 (format "safetensors offsets are out of range: ~a" name)))
       (define dims (map read-exact-integer shape))
       (define count (apply * dims))
       (define nbytes (- end start))
       (unless (= nbytes (* 4 count))
         (raise-gpt2 (format "safetensors byte length does not match shape: ~a" name)))
       (hash-set table name
                 (make-tensor dims (f32-flvector bs (+ data-base start) count)))])))
