#lang racket/base

;; Row-major tensors on a flat flvector. Dims are an explicit list.
;; Tensors are treated as immutable; tensor-reshape shares the flvector,
;; and mutating tensor-data also mutates any reshape of it.
;; Residual connections are tensor-add of the sublayer onto the stream.

(require racket/fixnum
         racket/flonum
         racket/string)

(provide make-tensor
         tensor?
         tensor-dims
         tensor-data
         tensor-ref
         tensor-add
         slice-rows
         slice-cols
         split-last
         tensor-reshape
         tensor-row-groups
         flvector-dot
         dims->string
         matrix-shape
         paste-cols!)

(struct tensor (dims data)
  #:constructor-name raw-tensor
  #:transparent)

(define (dims->string dims)
  (string-join (map number->string dims) " " #:before-first "(" #:after-last ")"))

(define (valid-dims? dims)
  (and (list? dims)
       (pair? dims)
       (andmap exact-nonnegative-integer? dims)))

(define (dim-product dims)
  (let loop ([dims dims] [n 1])
    (if (null? dims)
        n
        (loop (cdr dims) (* n (car dims))))))

(define (make-tensor dims data)
  (unless (valid-dims? dims)
    (raise-arguments-error 'make-tensor
                           "dims must be a non-empty list of nonnegative integers"
                           "dims" dims))
  (unless (flvector? data)
    (raise-argument-error 'make-tensor "flvector?" data))
  (define need (dim-product dims))
  (unless (= need (flvector-length data))
    (raise-arguments-error 'make-tensor
                           "data length does not match dims"
                           "dims" (dims->string dims)
                           "data length" (flvector-length data)
                           "expected length" need))
  (raw-tensor dims data))

(define (check-tensor who t)
  (unless (tensor? t)
    (raise-argument-error who "tensor?" t)))

(define (matrix-shape who t)
  (check-tensor who t)
  (define dims (tensor-dims t))
  (unless (= (length dims) 2)
    (raise-arguments-error who
                           "expected a rank-2 tensor"
                           "dims" (dims->string dims)))
  (values (car dims) (cadr dims)))

(define (last-dim dims)
  (let loop ([dims dims])
    (if (null? (cdr dims))
        (car dims)
        (loop (cdr dims)))))

(define (tensor-row-groups dims)
  (define width (last-dim dims))
  (define total (dim-product dims))
  (values width (if (zero? width) 0 (quotient total width))))

(define (butlast dims)
  (if (null? (cdr dims))
      '()
      (cons (car dims) (butlast (cdr dims)))))

(define (tensor-ref t . idxs)
  (check-tensor 'tensor-ref t)
  (define dims (tensor-dims t))
  (unless (= (length idxs) (length dims))
    (raise-arguments-error 'tensor-ref
                           "index rank does not match tensor"
                           "dims" (dims->string dims)
                           "indices" idxs))
  (define data (tensor-data t))
  (let loop ([dims dims] [idxs idxs] [offset 0])
    (cond
      [(null? dims) (flvector-ref data offset)]
      [else
       (define dim (car dims))
       (define idx (car idxs))
       (unless (and (exact-integer? idx) (<= 0 idx) (< idx dim))
         (raise-arguments-error 'tensor-ref
                                "index out of range"
                                "index" idx
                                "dim" dim))
       (loop (cdr dims) (cdr idxs) (+ (* offset dim) idx))])))

(define (flvector-dot a b)
  (unless (flvector? a)
    (raise-argument-error 'flvector-dot "flvector?" 0 a b))
  (unless (flvector? b)
    (raise-argument-error 'flvector-dot "flvector?" 1 a b))
  (define n (flvector-length a))
  (unless (fx= n (flvector-length b))
    (raise-arguments-error 'flvector-dot
                           "lengths differ"
                           "first length" n
                           "second length" (flvector-length b)))
  (let loop ([i 0] [acc 0.0])
    (if (fx= i n)
        acc
        (loop (fx+ i 1)
              (fl+ acc (fl* (flvector-ref a i) (flvector-ref b i)))))))

(define (tensor-add a b)
  (check-tensor 'tensor-add a)
  (check-tensor 'tensor-add b)
  (define da (tensor-dims a))
  (define db (tensor-dims b))
  (unless (equal? da db)
    (raise-arguments-error 'tensor-add
                           "shapes differ"
                           "left" (dims->string da)
                           "right" (dims->string db)))
  (define xa (tensor-data a))
  (define xb (tensor-data b))
  (define n (flvector-length xa))
  (define out (make-flvector n))
  (let loop ([i 0])
    (unless (fx= i n)
      (flvector-set! out i (fl+ (flvector-ref xa i) (flvector-ref xb i)))
      (loop (fx+ i 1))))
  (raw-tensor da out))

(define (check-slice who start end len)
  (unless (and (exact-integer? start) (exact-integer? end))
    (raise-arguments-error who
                           "start and end must be exact integers"
                           "start" start
                           "end" end))
  (unless (<= 0 start end len)
    (raise-arguments-error who
                           "slice bounds out of range"
                           "start" start
                           "end" end
                           "length" len)))

(define (slice-rows t start end)
  (check-tensor 'slice-rows t)
  (define dims (tensor-dims t))
  (unless (= (length dims) 2)
    (raise-arguments-error 'slice-rows
                           "expected a rank-2 tensor"
                           "dims" (dims->string dims)))
  (define rows (car dims))
  (define cols (cadr dims))
  (check-slice 'slice-rows start end rows)
  (define n (- end start))
  (define src (tensor-data t))
  (define base (* start cols))
  (define count (* n cols))
  (define out (flvector-copy src base (+ base count)))
  (raw-tensor (list n cols) out))

(define (slice-cols t start end #:rows [row-limit #f])
  (check-tensor 'slice-cols t)
  (define dims (tensor-dims t))
  (unless (= (length dims) 2)
    (raise-arguments-error 'slice-cols
                           "expected a rank-2 tensor"
                           "dims" (dims->string dims)))
  (define all-rows (car dims))
  (define rows (or row-limit all-rows))
  (unless (and (exact-nonnegative-integer? rows) (<= rows all-rows))
    (raise-arguments-error 'slice-cols
                           "row limit is past the tensor"
                           "rows" rows
                           "tensor rows" all-rows))
  (define cols (cadr dims))
  (check-slice 'slice-cols start end cols)
  (define width (- end start))
  (define out (make-flvector (* rows width)))
  (define src (tensor-data t))
  (let loop-i ([i 0])
    (unless (fx= i rows)
      (let loop-j ([j 0])
        (unless (fx= j width)
          (flvector-set! out (fx+ (fx* i width) j)
                         (flvector-ref src (fx+ (fx+ (fx* i cols) start) j)))
          (loop-j (fx+ j 1))))
      (loop-i (fx+ i 1))))
  (raw-tensor (list rows width) out))

;; Inverse of slice-cols: copy src's columns into dest starting at col0.
(define (paste-cols! dest dest-cols src col0)
  (define rows (car (tensor-dims src)))
  (define width (cadr (tensor-dims src)))
  (define from (tensor-data src))
  (define to (tensor-data dest))
  (for ([i (in-range rows)])
    (define src-base (* i width))
    (define dest-base (+ (* i dest-cols) col0))
    (for ([j (in-range width)])
      (flvector-set! to (+ dest-base j) (flvector-ref from (+ src-base j))))))

(define (split-last t parts)
  (check-tensor 'split-last t)
  (unless (exact-positive-integer? parts)
    (raise-argument-error 'split-last "exact-positive-integer?" parts))
  (define dims (tensor-dims t))
  (define width (last-dim dims))
  (unless (zero? (remainder width parts))
    (raise-arguments-error 'split-last
                           "last dimension is not divisible by the part count"
                           "dims" (dims->string dims)
                           "parts" parts))
  (define part (quotient width parts))
  (define prefix (butlast dims))
  (define groups (dim-product prefix))
  (define src (tensor-data t))
  (for/list ([p (in-range parts)])
    (define out (make-flvector (* groups part)))
    (define src-off (* p part))
    (let loop-i ([i 0])
      (unless (fx= i groups)
        (define dst-base (fx* i part))
        (define src-base (fx+ (fx* i width) src-off))
        (let loop-j ([j 0])
          (unless (fx= j part)
            (flvector-set! out (fx+ dst-base j)
                           (flvector-ref src (fx+ src-base j)))
            (loop-j (fx+ j 1))))
        (loop-i (fx+ i 1))))
    (raw-tensor (append prefix (list part)) out)))

(define (tensor-reshape t dims)
  (check-tensor 'tensor-reshape t)
  (unless (valid-dims? dims)
    (raise-arguments-error 'tensor-reshape
                           "dims must be a non-empty list of nonnegative integers"
                           "dims" dims))
  (define from (tensor-dims t))
  (define need (dim-product dims))
  (unless (= need (flvector-length (tensor-data t)))
    (raise-arguments-error 'tensor-reshape
                           "element count does not match"
                           "from" (dims->string from)
                           "to" (dims->string dims)))
  (raw-tensor dims (tensor-data t)))
