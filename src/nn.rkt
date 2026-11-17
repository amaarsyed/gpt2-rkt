#lang racket/base

;; GPT-2 numeric kernels. Storage is row-major flvectors.
;; matmul is HF Conv1D: y[n, out] = x[n, in] · W[in, out] + b[out].
;; The inner loops are i-k-j so each step reads a contiguous W row.
;; matmul-transposed-b is h[n, in] · Wᵀ with W stored as [out, in]
;; (tied wte). Its k loop dots two contiguous rows.

(require racket/fixnum
         racket/flonum
         racket/future
         "tensor.rkt")

(provide matmul
         matmul-transposed-b
         softmax
         layer-norm
         layernorm-eps
         gelu)

(define layernorm-eps 1e-5)

;; Below this row count the future setup costs more than the arithmetic.
(define parallel-row-threshold 32)

;; tanh(x) = (e^{2x}-1)/(e^{2x}+1), saturated where the flonum result is ±1.
(define (fltanh x)
  (cond
    [(fl> x 20.0) 1.0]
    [(fl< x -20.0) -1.0]
    [else
     (define e (flexp (fl* 2.0 x)))
     (fl/ (fl- e 1.0) (fl+ e 1.0))]))

;; sqrt(2/π), decimal from an independent evaluation of that quantity.
(define gelu-k 0.7978845608028654)
(define gelu-cubic 0.044715)

(define (flgelu x)
  (define x2 (fl* x x))
  (define x3 (fl* x2 x))
  (define inner (fl* gelu-k (fl+ x (fl* gelu-cubic x3))))
  (fl* 0.5 (fl* x (fl+ 1.0 (fltanh inner)))))

(define (require-matrix who t)
  (unless (tensor? t)
    (raise-argument-error who "tensor?" t))
  (define dims (tensor-dims t))
  (unless (fx= (length dims) 2)
    (raise-arguments-error who
                           "expected a rank-2 tensor"
                           "dims" (dims->string dims)))
  (values (car dims) (cadr dims) (tensor-data t)))

(define (for-row-blocks rows proc)
  (define cpus (processor-count))
  (cond
    [(or (fx< rows parallel-row-threshold) (fx< cpus 2))
     (proc 0 rows)]
    [else
     (define futures
       (for/list ([cpu (in-range cpus)])
         (define start (quotient (fx* cpu rows) cpus))
         (define end (quotient (fx* (fx+ cpu 1) rows) cpus))
         (future (lambda () (proc start end)))))
     (for-each touch futures)]))

;; y[i, j] = b[j] + sum_k x[i, k] * W[k, j]
(define (fill-matmul-rows! y x W b in out start end)
  (let loop-i ([i start])
    (unless (fx= i end)
      (define ybase (fx* i out))
      (define xbase (fx* i in))
      (let loop-j ([j 0])
        (unless (fx= j out)
          (flvector-set! y (fx+ ybase j) (flvector-ref b j))
          (loop-j (fx+ j 1))))
      (let loop-k ([k 0])
        (unless (fx= k in)
          (define xik (flvector-ref x (fx+ xbase k)))
          (define wbase (fx* k out))
          (let loop-j ([j 0])
            (unless (fx= j out)
              (define yj (fx+ ybase j))
              (flvector-set! y yj
                             (fl+ (flvector-ref y yj)
                                  (fl* xik (flvector-ref W (fx+ wbase j)))))
              (loop-j (fx+ j 1))))
          (loop-k (fx+ k 1))))
      (loop-i (fx+ i 1)))))

(define (matmul x W b)
  (define-values (n in xdata) (require-matrix 'matmul x))
  (define-values (w-in out wdata) (require-matrix 'matmul W))
  (unless (fx= in w-in)
    (raise-arguments-error 'matmul
                           "x columns do not match W rows"
                           "x" (dims->string (tensor-dims x))
                           "W" (dims->string (tensor-dims W))))
  (unless (tensor? b)
    (raise-argument-error 'matmul "tensor?" b))
  (define bdims (tensor-dims b))
  (unless (and (fx= (length bdims) 1) (fx= (car bdims) out))
    (raise-arguments-error 'matmul
                           "bias length does not match W columns"
                           "bias" (dims->string bdims)
                           "W" (dims->string (tensor-dims W))))
  (define y (make-flvector (fx* n out)))
  (define bdata (tensor-data b))
  (for-row-blocks n
    (lambda (start end)
      (fill-matmul-rows! y xdata wdata bdata in out start end)))
  (make-tensor (list n out) y))

;; y[i, j] = sum_k h[i, k] * W[j, k], W stored [out, in].
(define (fill-transposed-rows! y h W in out start end)
  (let loop-i ([i start])
    (unless (fx= i end)
      (define ybase (fx* i out))
      (define hbase (fx* i in))
      (let loop-j ([j 0])
        (unless (fx= j out)
          (define wbase (fx* j in))
          (define acc
            (let loop-k ([k 0] [sum 0.0])
              (if (fx= k in)
                  sum
                  (loop-k (fx+ k 1)
                          (fl+ sum
                               (fl* (flvector-ref h (fx+ hbase k))
                                    (flvector-ref W (fx+ wbase k))))))))
          (flvector-set! y (fx+ ybase j) acc)
          (loop-j (fx+ j 1))))
      (loop-i (fx+ i 1)))))

(define (matmul-transposed-b h W)
  (define-values (n in hdata) (require-matrix 'matmul-transposed-b h))
  (define-values (out w-in wdata) (require-matrix 'matmul-transposed-b W))
  (unless (fx= in w-in)
    (raise-arguments-error 'matmul-transposed-b
                           "h columns do not match W columns"
                           "h" (dims->string (tensor-dims h))
                           "W" (dims->string (tensor-dims W))))
  (define y (make-flvector (fx* n out)))
  (for-row-blocks n
    (lambda (start end)
      (fill-transposed-rows! y hdata wdata in out start end)))
  (make-tensor (list n out) y))

(define (row-groups dims)
  (define width
    (let loop ([dims dims])
      (if (null? (cdr dims))
          (car dims)
          (loop (cdr dims)))))
  (define total
    (let loop ([dims dims] [n 1])
      (if (null? dims)
          n
          (loop (cdr dims) (* n (car dims))))))
  (values width (if (zero? width) 0 (quotient total width))))

(define (softmax t)
  (unless (tensor? t)
    (raise-argument-error 'softmax "tensor?" t))
  (define dims (tensor-dims t))
  (define-values (width rows) (row-groups dims))
  (when (fx= width 0)
    (raise-arguments-error 'softmax
                           "last dimension must be positive"
                           "dims" (dims->string dims)))
  (define src (tensor-data t))
  (define out (make-flvector (flvector-length src)))
  (let loop-i ([i 0])
    (unless (fx= i rows)
      (define base (fx* i width))
      (define maxv
        (let loop-j ([j 0] [m -inf.0])
          (if (fx= j width)
              m
              (loop-j (fx+ j 1) (flmax m (flvector-ref src (fx+ base j)))))))
      (cond
        [(fl= maxv -inf.0)
         (let loop-j ([j 0])
           (unless (fx= j width)
             (flvector-set! out (fx+ base j) 0.0)
             (loop-j (fx+ j 1))))]
        ;; +inf - +inf is NaN, so a row that contains +inf cannot be
        ;; shifted. The infinite positions share the mass and the rest are 0.
        [(fl= maxv +inf.0)
         (define count
           (let loop-j ([j 0] [n 0])
             (if (fx= j width)
                 n
                 (loop-j (fx+ j 1)
                         (if (fl= (flvector-ref src (fx+ base j)) +inf.0)
                             (fx+ n 1)
                             n)))))
         (define share (fl/ 1.0 (fx->fl count)))
         (let loop-j ([j 0])
           (unless (fx= j width)
             (define idx (fx+ base j))
             (flvector-set! out idx
                            (if (fl= (flvector-ref src idx) +inf.0)
                                share
                                0.0))
             (loop-j (fx+ j 1))))]
        [else
         (define sum
           (let loop-j ([j 0] [s 0.0])
             (if (fx= j width)
                 s
                 (let* ([idx (fx+ base j)]
                        [e (flexp (fl- (flvector-ref src idx) maxv))])
                   (flvector-set! out idx e)
                   (loop-j (fx+ j 1) (fl+ s e))))))
         (let loop-j ([j 0])
           (unless (fx= j width)
             (define idx (fx+ base j))
             (flvector-set! out idx (fl/ (flvector-ref out idx) sum))
             (loop-j (fx+ j 1))))])
      (loop-i (fx+ i 1))))
  (make-tensor dims out))

(define (layer-norm x gain bias #:eps [eps layernorm-eps])
  (unless (tensor? x)
    (raise-argument-error 'layer-norm "tensor?" x))
  (unless (and (flonum? eps) (fl> eps 0.0))
    (raise-argument-error 'layer-norm "positive flonum?" eps))
  (define dims (tensor-dims x))
  (define-values (width rows) (row-groups dims))
  (when (fx= width 0)
    (raise-arguments-error 'layer-norm
                           "last dimension must be positive"
                           "dims" (dims->string dims)))
  (define (require-vector who v)
    (unless (tensor? v)
      (raise-argument-error 'layer-norm "tensor?" v))
    (define vdims (tensor-dims v))
    (unless (and (fx= (length vdims) 1) (fx= (car vdims) width))
      (raise-arguments-error 'layer-norm
                             (format "~a length does not match the last dimension" who)
                             who (dims->string vdims)
                             "last dimension" width))
    (tensor-data v))
  (define g (require-vector "gain" gain))
  (define b (require-vector "bias" bias))
  (define src (tensor-data x))
  (define out (make-flvector (flvector-length src)))
  (define inv-n (fl/ 1.0 (fx->fl width)))
  (let loop-i ([i 0])
    (unless (fx= i rows)
      (define base (fx* i width))
      (define mean
        (let loop-j ([j 0] [sum 0.0])
          (if (fx= j width)
              (fl* sum inv-n)
              (loop-j (fx+ j 1) (fl+ sum (flvector-ref src (fx+ base j)))))))
      (define var
        (let loop-j ([j 0] [sum 0.0])
          (if (fx= j width)
              (fl* sum inv-n)
              (let ([d (fl- (flvector-ref src (fx+ base j)) mean)])
                (loop-j (fx+ j 1) (fl+ sum (fl* d d)))))))
      (define inv-std (fl/ 1.0 (flsqrt (fl+ var eps))))
      (let loop-j ([j 0])
        (unless (fx= j width)
          (define z (fl* (fl- (flvector-ref src (fx+ base j)) mean) inv-std))
          (flvector-set! out (fx+ base j)
                         (fl+ (fl* z (flvector-ref g j)) (flvector-ref b j)))
          (loop-j (fx+ j 1))))
      (loop-i (fx+ i 1))))
  (make-tensor dims out))

(define (gelu t)
  (unless (tensor? t)
    (raise-argument-error 'gelu "tensor?" t))
  (define src (tensor-data t))
  (define n (flvector-length src))
  (define out (make-flvector n))
  (let loop ([i 0])
    (unless (fx= i n)
      (flvector-set! out i (flgelu (flvector-ref src i)))
      (loop (fx+ i 1))))
  (make-tensor (tensor-dims t) out))
