#lang racket/base

;; Incremental UTF-8. A token may end in the middle of a character.
;; Those bytes stay buffered until the sequence is complete. Only a
;; finish replaces a trailing fragment with U+FFFD.

(provide make-utf8-stream
         utf8-stream-add
         utf8-stream-finish)

(struct utf8-stream (pending emitted)
  #:mutable
  #:constructor-name raw-utf8-stream)

(define (make-utf8-stream)
  (raw-utf8-stream #"" ""))

;; Bytes that cannot start or continue a character at the end of the
;; buffer are complete. An incomplete lead sequence is the remainder.
(define (split-incomplete bs)
  (define n (bytes-length bs))
  (if (zero? n)
      (values #"" #"")
      (let loop ([i (sub1 n)] [cont 0])
        (cond
          [(< i 0) (values #"" bs)]
          [else
           (define b (bytes-ref bs i))
           (cond
             [(<= b #x7F) (values bs #"")]
             [(<= #x80 b #xBF) (loop (sub1 i) (add1 cont))]
             [else
              (define need
                (cond
                  [(<= b #xDF) 2]
                  [(<= b #xEF) 3]
                  [(<= b #xF7) 4]
                  [else 1]))
              (if (< (add1 cont) need)
                  (values (subbytes bs 0 i) (subbytes bs i))
                  (values bs #""))])]))))

(define (utf8-stream-add stream bs)
  (unless (bytes? bs)
    (raise-argument-error 'utf8-stream-add "bytes?" bs))
  (define-values (complete pending)
    (split-incomplete (bytes-append (utf8-stream-pending stream) bs)))
  (define text (if (zero? (bytes-length complete))
                   ""
                   (bytes->string/utf-8 complete #\uFFFD)))
  (set-utf8-stream-pending! stream pending)
  (set-utf8-stream-emitted! stream (string-append (utf8-stream-emitted stream) text))
  text)

(define (utf8-stream-finish stream)
  (define pending (utf8-stream-pending stream))
  (define text (if (zero? (bytes-length pending))
                   ""
                   (bytes->string/utf-8 pending #\uFFFD)))
  (set-utf8-stream-pending! stream #"")
  (set-utf8-stream-emitted! stream (string-append (utf8-stream-emitted stream) text))
  text)

(define (utf8-stream-text stream)
  (utf8-stream-emitted stream))

(provide utf8-stream-text)
