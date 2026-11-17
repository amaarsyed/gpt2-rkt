#lang racket/base

;; Minimal safetensors reader: an 8-byte little-endian header length,
;; a JSON header, then raw little-endian F32 bytes. Other dtypes are
;; rejected. load-model uses this reader for both GPT-2 and the tiny model.

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

;; Read n little-endian F32 values from bs at start. The caller has
;; already checked that the range is inside bs. The official payload is
;; not 4-byte aligned, and this reads it in place.
(define (f32-flvector bs start n)
  (define out (make-flvector n))
  (for ([i (in-range n)])
    (define base (unsafe-fx+ start (unsafe-fx* i 4)))
    (unsafe-flvector-set! out i
                          (floating-point-bytes->real bs #f base (unsafe-fx+ base 4))))
  out)

(define (safetensors-entry-name key)
  (if (symbol? key) (symbol->string key) key))

;; `#:skip` drops a tensor before its bytes are decoded. The GPT-2
;; loader uses it for the `attn.bias` mask buffers. A header need not
;; be a multiple of 8: the official file's length is 14283, so the
;; float payload is not 4-byte aligned, and `read-json` still accepts
;; a header padded with spaces.
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
