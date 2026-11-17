#lang racket/base

;; Minimal safetensors reader: an 8-byte little-endian header length,
;; a JSON header, then raw little-endian F32 bytes. Other dtypes are
;; rejected. This is the format load-model reads, so the real GPT-2
;; file can use the same function later.

(require json
         racket/flonum
         racket/port
         "errors.rkt"
         "tensor.rkt")

(provide read-safetensors)

(define (read-exact-integer n)
  (cond
    [(exact-nonnegative-integer? n) n]
    [(and (real? n) (integer? n) (>= n 0)) (inexact->exact n)]
    [else #f]))

(define (f32-flvector raw)
  (define nbytes (bytes-length raw))
  (unless (zero? (remainder nbytes 4))
    (raise-gpt2 (format "safetensors F32 data length is not a multiple of 4: ~a" nbytes)))
  (define n (quotient nbytes 4))
  (define out (make-flvector n))
  (for ([i (in-range n)])
    (define base (* i 4))
    (flvector-set! out i
                   (real->double-flonum
                    (floating-point-bytes->real raw #f base (+ base 4)))))
  out)

(define (read-safetensors path)
  (define bs
    (with-handlers ([exn:fail? (lambda (e)
                                 (raise-gpt2 (format "could not read ~a: ~a" path (exn-message e))))])
      (call-with-input-file path port->bytes #:mode 'binary)))
  (when (< (bytes-length bs) 8)
    (raise-gpt2 (format "safetensors file is too small: ~a" path)))
  (define header-len (integer-bytes->integer bs #f #f 0 8))
  (define header-end (+ 8 header-len))
  (when (or (< header-len 2) (> header-end (bytes-length bs)))
    (raise-gpt2 (format "safetensors header length is out of range: ~a" path)))
  (define header
    (with-handlers ([exn:fail? (lambda (e)
                                 (raise-gpt2 (format "safetensors header is not JSON: ~a" (exn-message e))))])
      (read-json (open-input-bytes (subbytes bs 8 header-end)))))
  (unless (hash? header)
    (raise-gpt2 "safetensors header is not an object"))
  (define data-base header-end)
  (for/hash ([(key info) (in-hash header)]
             #:unless (equal? (format "~a" key) "__metadata__"))
    (unless (hash? info)
      (raise-gpt2 (format "safetensors entry is not an object: ~a" key)))
    (define dtype (hash-ref info 'dtype #f))
    (unless (equal? dtype "F32")
      (raise-gpt2 (format "safetensors dtype ~a is not supported" dtype)))
    (define shape (hash-ref info 'shape #f))
    (define offsets (hash-ref info 'data_offsets #f))
    (unless (and (list? shape) (andmap read-exact-integer shape))
      (raise-gpt2 (format "safetensors shape is invalid: ~a" key)))
    (unless (and (list? offsets) (= (length offsets) 2))
      (raise-gpt2 (format "safetensors offsets are invalid: ~a" key)))
    (define start (read-exact-integer (car offsets)))
    (define end (read-exact-integer (cadr offsets)))
    (unless (and start end (<= 0 start end) (<= (+ data-base end) (bytes-length bs)))
      (raise-gpt2 (format "safetensors offsets are out of range: ~a" key)))
    (define dims (map read-exact-integer shape))
    (define raw (subbytes bs (+ data-base start) (+ data-base end)))
    (define nums (f32-flvector raw))
    (define name (if (symbol? key) (symbol->string key) key))
    (values name (make-tensor dims nums))))
