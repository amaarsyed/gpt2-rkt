#lang racket/base

;; GPT-2's bytes_to_unicode map. Printable bytes stay themselves.
;; The remaining bytes, in order, map to code points starting at U+0100
;; so the BPE never sees a raw space or control character.

(provide bpe-byte-count
         bpe-byte->char
         bpe-char->byte)

(define bpe-byte-count 256)

(define (code-list from to)
  (for/list ([i (in-range from to)])
    i))

(define byte->char-vec
  (let* ([initial (append (code-list (char->integer #\!) (add1 (char->integer #\~)))
                          (code-list #xA1 (add1 #xAC))
                          (code-list #xAE (add1 #xFF)))]
         [seen (make-vector 256 #f)]
         [vec (make-vector 256)]
         [n 0])
    (for ([b (in-list initial)])
      (vector-set! seen b #t)
      (vector-set! vec b (integer->char b)))
    (for ([b (in-range 256)])
      (unless (vector-ref seen b)
        (vector-set! vec b (integer->char (+ 256 n)))
        (set! n (add1 n))))
    vec))

(define char->byte-hash
  (for/hash ([b (in-range 256)])
    (values (vector-ref byte->char-vec b) b)))

(define (bpe-byte->char b)
  (unless (and (exact-integer? b) (<= 0 b 255))
    (raise-argument-error 'bpe-byte->char "byte?" b))
  (vector-ref byte->char-vec b))

(define (bpe-char->byte c)
  (unless (char? c)
    (raise-argument-error 'bpe-char->byte "char?" c))
  (hash-ref char->byte-hash c
            (lambda ()
              (raise-arguments-error 'bpe-char->byte
                                     "character is not in the byte map"
                                     "char" c))))
