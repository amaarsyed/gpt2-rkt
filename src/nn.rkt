#lang racket/base

;; GPT-2 numeric kernels. Storage is row-major flvectors.
;; matmul is HF Conv1D: y[n, out] = x[n, in] · W[in, out] + b[out].
;; The inner loops are i-k-j so each step reads a contiguous W row.
;; A multiply below parallel-work-threshold flops runs in one thread, so
;; a small per-head attention matmul does not spawn futures. Larger
;; problems split across rows, or across output columns when the row
;; count is short, which still covers a single-token logits multiply.
;; matmul-transposed-b is h[n, in] · Wᵀ with W stored as [out, in]
;; (tied wte). Its k loop dots two contiguous rows.

(require racket/fixnum
         racket/flonum
         racket/future
         racket/unsafe/ops
         "tensor.rkt")

(provide matmul
         matmul-gelu
         matmul-add
         matmul-column-block
         matmul-transposed-b
         matmul-transposed-block
         softmax
         layer-norm
         layernorm-eps
         current-decode-workers
         current-prefill-workers
         gelu)

(define layernorm-eps 1e-5)

;; Below this many flops (rows × inner × cols) the future setup costs
;; more than the arithmetic. Per-head attention stays under it. The
;; logits multiply and the MLP stay over it.
(define parallel-work-threshold 100000)

(define (worker-budget n)
  (unless (or (not n) (exact-positive-integer? n))
    (raise-argument-error 'worker-budget "#f or positive integer" n))
  n)

;; #f uses the host count. Budgets apply to matrix products, not heads.
(define current-decode-workers (make-parameter #f worker-budget))
(define current-prefill-workers (make-parameter #f worker-budget))
;; Once a problem is large enough to parallelize, split rows when there
;; are at least this many, and output columns otherwise.
(define parallel-axis-threshold 32)
;; One block per processor avoids dozens of tiny decode futures.
(define parallel-block-multiple 1)

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
  (define-values (rows cols) (matrix-shape who t))
  (values rows cols (tensor-data t)))

(define (for-parallel-blocks total cpus proc)
  (cond
    [(or (fx< cpus 2) (fx< total parallel-axis-threshold))
     (proc 0 total)]
    [else
     (define blocks (fx* parallel-block-multiple cpus))
     (define jobs
       (for/list ([b (in-range blocks)])
         (define start (quotient (fx* b total) blocks))
         (define end (quotient (fx* (fx+ b 1) total) blocks))
         (and (fx< start end)
              (future (lambda () (proc start end))))))
     (for ([job (in-list jobs)] #:when job)
       (touch job))]))

;; All four matrix products use the same row/column split.
(define (for-matmul-blocks n in out proc)
  (define budget (if (= n 1) (current-decode-workers) (current-prefill-workers)))
  (define cpus (min (or budget (processor-count)) (processor-count)))
  (cond
    [(or (< cpus 2) (< (* n in out) parallel-work-threshold))
     (proc 0 n 0 out)]
    [(fx>= n parallel-axis-threshold)
     (for-parallel-blocks n cpus
       (lambda (i0 i1) (proc i0 i1 0 out)))]
    [else
     (for-parallel-blocks out cpus
       (lambda (j0 j1) (proc 0 n j0 j1)))]))

;; y[i, j] = b[j] + sum_k x[i, k] * W[k, col0 + j].
;; W's rows are w-stride apart. A contiguous W has w-stride = out and col0 = 0.
(define (fill-matmul! y x W b in out i0 i1 j0 j1 w-stride col0)
  (let loop-i ([i i0])
    (unless (unsafe-fx= i i1)
      (define ybase (unsafe-fx* i out))
      (define xbase (unsafe-fx* i in))
      (let loop-j ([j j0])
        (define j4 (unsafe-fx+ j 4))
        (cond
          [(unsafe-fx<= j4 j1)
           (unsafe-flvector-set! y (unsafe-fx+ ybase j) (unsafe-flvector-ref b j))
           (unsafe-flvector-set! y (unsafe-fx+ ybase (unsafe-fx+ j 1))
                                 (unsafe-flvector-ref b (unsafe-fx+ j 1)))
           (unsafe-flvector-set! y (unsafe-fx+ ybase (unsafe-fx+ j 2))
                                 (unsafe-flvector-ref b (unsafe-fx+ j 2)))
           (unsafe-flvector-set! y (unsafe-fx+ ybase (unsafe-fx+ j 3))
                                 (unsafe-flvector-ref b (unsafe-fx+ j 3)))
           (loop-j j4)]
          [(unsafe-fx< j j1)
           (unsafe-flvector-set! y (unsafe-fx+ ybase j) (unsafe-flvector-ref b j))
           (loop-j (unsafe-fx+ j 1))]))
      (let loop-k ([k 0])
        (unless (unsafe-fx= k in)
          (define xik (unsafe-flvector-ref x (unsafe-fx+ xbase k)))
          (define wbase (unsafe-fx+ col0 (unsafe-fx* k w-stride)))
          (let loop-j ([j j0])
            (define j4 (unsafe-fx+ j 4))
            (cond
              [(unsafe-fx<= j4 j1)
               (define y0 (unsafe-fx+ ybase j))
               (define y1 (unsafe-fx+ y0 1))
               (define y2 (unsafe-fx+ y0 2))
               (define y3 (unsafe-fx+ y0 3))
               (unsafe-flvector-set!
                y y0
                (unsafe-fl+ (unsafe-flvector-ref y y0)
                            (unsafe-fl* xik (unsafe-flvector-ref W (unsafe-fx+ wbase j)))))
               (unsafe-flvector-set!
                y y1
                (unsafe-fl+ (unsafe-flvector-ref y y1)
                            (unsafe-fl* xik
                                        (unsafe-flvector-ref W (unsafe-fx+ wbase (unsafe-fx+ j 1))))))
               (unsafe-flvector-set!
                y y2
                (unsafe-fl+ (unsafe-flvector-ref y y2)
                            (unsafe-fl* xik
                                        (unsafe-flvector-ref W (unsafe-fx+ wbase (unsafe-fx+ j 2))))))
               (unsafe-flvector-set!
                y y3
                (unsafe-fl+ (unsafe-flvector-ref y y3)
                            (unsafe-fl* xik
                                        (unsafe-flvector-ref W (unsafe-fx+ wbase (unsafe-fx+ j 3))))))
               (loop-j j4)]
              [(unsafe-fx< j j1)
               (define yj (unsafe-fx+ ybase j))
               (unsafe-flvector-set!
                y yj
                (unsafe-fl+ (unsafe-flvector-ref y yj)
                            (unsafe-fl* xik (unsafe-flvector-ref W (unsafe-fx+ wbase j)))))
               (loop-j (unsafe-fx+ j 1))]))
          (loop-k (unsafe-fx+ k 1))))
      (loop-i (unsafe-fx+ i 1)))))

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
  (for-matmul-blocks n in out
    (lambda (i0 i1 j0 j1)
      (fill-matmul! y xdata wdata bdata in out i0 i1 j0 j1 out 0)))
  (make-tensor (list n out) y))

;; These products own their output, so the following pass can reuse it.
(define (matmul-gelu x W b)
  (define out (matmul x W b))
  (define data (tensor-data out))
  (for ([i (in-range (flvector-length data))])
    (flvector-set! data i (flgelu (flvector-ref data i))))
  out)

(define (matmul-add x W b residual)
  (define out (matmul x W b))
  (unless (and (tensor? residual) (equal? (tensor-dims out) (tensor-dims residual)))
    (raise-argument-error 'matmul-add "residual with the output shape" residual))
  (define data (tensor-data out))
  (define r (tensor-data residual))
  (for ([i (in-range (flvector-length data))])
    (flvector-set! data i (fl+ (flvector-ref r i) (flvector-ref data i))))
  out)

;; The same product, reading columns [col, col+width) of a wider W and
;; only the leading (tensor columns of x) rows. Attention uses this so a
;; head does not copy its key or value prefix out of the cache.
(define (matmul-column-block x W b col width)
  (define-values (n in xdata) (require-matrix 'matmul-column-block x))
  (define-values (w-rows w-cols wdata) (require-matrix 'matmul-column-block W))
  (unless (and (exact-nonnegative-integer? col)
               (exact-positive-integer? width)
               (<= (+ col width) w-cols)
               (<= in w-rows))
    (raise-arguments-error 'matmul-column-block
                           "column block is outside W"
                           "x" (dims->string (tensor-dims x))
                           "W" (dims->string (tensor-dims W))
                           "col" col
                           "width" width))
  (unless (tensor? b)
    (raise-argument-error 'matmul-column-block "tensor?" b))
  (define bdims (tensor-dims b))
  (unless (and (fx= (length bdims) 1) (fx= (car bdims) width))
    (raise-arguments-error 'matmul-column-block
                           "bias length does not match the column block"
                           "bias" (dims->string bdims)
                           "width" width))
  (define y (make-flvector (fx* n width)))
  (define bdata (tensor-data b))
  (for-matmul-blocks n in width
    (lambda (i0 i1 j0 j1)
      (fill-matmul! y xdata wdata bdata in width i0 i1 j0 j1 w-cols col)))
  (make-tensor (list n width) y))

(define (dot-row h hbase W wbase in)
  (let loop-k ([k 0] [sum 0.0])
    (if (unsafe-fx= k in)
        sum
        (loop-k (unsafe-fx+ k 1)
                (unsafe-fl+ sum
                            (unsafe-fl* (unsafe-flvector-ref h (unsafe-fx+ hbase k))
                                        (unsafe-flvector-ref W (unsafe-fx+ wbase k))))))))

;; y[i, j] = sum_k h[i, k] * W[j, col0 + k].
;; W rows are row-stride apart. A contiguous W has row-stride = in and col0 = 0.
(define (fill-transposed! y h W in out row-stride col0 i0 i1 j0 j1)
  (let loop-i ([i i0])
    (unless (unsafe-fx= i i1)
      (define ybase (unsafe-fx* i out))
      (define hbase (unsafe-fx* i in))
      (let loop-j ([j j0])
        (define j4 (unsafe-fx+ j 4))
        (cond
          [(unsafe-fx<= j4 j1)
           (unsafe-flvector-set! y (unsafe-fx+ ybase j)
                                 (dot-row h hbase W (unsafe-fx+ col0 (unsafe-fx* j row-stride)) in))
           (unsafe-flvector-set! y (unsafe-fx+ ybase (unsafe-fx+ j 1))
                                 (dot-row h hbase W (unsafe-fx+ col0 (unsafe-fx* (unsafe-fx+ j 1) row-stride)) in))
           (unsafe-flvector-set! y (unsafe-fx+ ybase (unsafe-fx+ j 2))
                                 (dot-row h hbase W (unsafe-fx+ col0 (unsafe-fx* (unsafe-fx+ j 2) row-stride)) in))
           (unsafe-flvector-set! y (unsafe-fx+ ybase (unsafe-fx+ j 3))
                                 (dot-row h hbase W (unsafe-fx+ col0 (unsafe-fx* (unsafe-fx+ j 3) row-stride)) in))
           (loop-j j4)]
          [(unsafe-fx< j j1)
           (unsafe-flvector-set! y (unsafe-fx+ ybase j)
                                 (dot-row h hbase W (unsafe-fx+ col0 (unsafe-fx* j row-stride)) in))
           (loop-j (unsafe-fx+ j 1))]))
      (loop-i (unsafe-fx+ i 1)))))

(define (matmul-transposed-b h W)
  (define-values (n in hdata) (require-matrix 'matmul-transposed-b h))
  (define-values (out w-in wdata) (require-matrix 'matmul-transposed-b W))
  (unless (fx= in w-in)
    (raise-arguments-error 'matmul-transposed-b
                           "h columns do not match W columns"
                           "h" (dims->string (tensor-dims h))
                           "W" (dims->string (tensor-dims W))))
  (define y (make-flvector (fx* n out)))
  (for-matmul-blocks n in out
    (lambda (i0 i1 j0 j1)
      (fill-transposed! y hdata wdata in out in 0 i0 i1 j0 j1)))
  (make-tensor (list n out) y))

;; h[n, width] · K[0:rows, col:col+width)^T, K stored [capacity, stride].
(define (matmul-transposed-block h W rows col)
  (define-values (n in hdata) (require-matrix 'matmul-transposed-block h))
  (define-values (w-rows w-cols wdata) (require-matrix 'matmul-transposed-block W))
  (unless (and (exact-positive-integer? rows)
               (<= rows w-rows)
               (exact-nonnegative-integer? col)
               (<= (+ col in) w-cols))
    (raise-arguments-error 'matmul-transposed-block
                           "column block is outside W"
                           "h" (dims->string (tensor-dims h))
                           "W" (dims->string (tensor-dims W))
                           "rows" rows
                           "col" col))
  (define y (make-flvector (fx* n rows)))
  (for-matmul-blocks n in rows
    (lambda (i0 i1 j0 j1)
      (fill-transposed! y hdata wdata in rows w-cols col i0 i1 j0 j1)))
  (make-tensor (list n rows) y))

(define (softmax t)
  (unless (tensor? t)
    (raise-argument-error 'softmax "tensor?" t))
  (define dims (tensor-dims t))
  (define-values (width rows) (tensor-row-groups dims))
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
  (define-values (width rows) (tensor-row-groups dims))
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
