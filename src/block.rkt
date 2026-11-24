#lang racket/base

;; One pre-LN GPT-2 block: h = x + attn(ln_1(x)); out = h + mlp(ln_2(h)).
;; The MLP is c_fc [width, inner], tanh GELU, c_proj [inner, width].

(require racket/flonum
         "attention.rkt"
         "nn.rkt"
         "source-map.rkt"
         "tensor.rkt"
         "trace.rkt")

(provide gpt2-mlp
         gpt2-mlp-hidden
         make-block-params
         block-params-from-tensors
         block-params?
         block-params-n-head
         block-trace?
         block-trace-ln1
         block-trace-attn
         block-trace-residual1
         block-trace-ln2
         block-trace-mlp-hidden
         block-trace-out
         gpt2-block-parts
         gpt2-block
         gpt2-block-cached
         gpt2-blocks)

(struct block-params (n-head
                      ln1-gain ln1-bias
                      c-attn-w c-attn-b
                      attn-proj-w attn-proj-b
                      ln2-gain ln2-bias
                      c-fc-w c-fc-b
                      mlp-proj-w mlp-proj-b)
  #:constructor-name raw-block-params
  #:transparent)

;; Keywords avoid same-shape gain/bias swaps; tensor names use HF suffixes.
(define (make-block-params #:n-head n-head
                           #:ln1-gain ln1-gain
                           #:ln1-bias ln1-bias
                           #:c-attn-w c-attn-w
                           #:c-attn-b c-attn-b
                           #:attn-proj-w attn-proj-w
                           #:attn-proj-b attn-proj-b
                           #:ln2-gain ln2-gain
                           #:ln2-bias ln2-bias
                           #:c-fc-w c-fc-w
                           #:c-fc-b c-fc-b
                           #:mlp-proj-w mlp-proj-w
                           #:mlp-proj-b mlp-proj-b)
  (raw-block-params n-head
                    ln1-gain ln1-bias
                    c-attn-w c-attn-b
                    attn-proj-w attn-proj-b
                    ln2-gain ln2-bias
                    c-fc-w c-fc-b
                    mlp-proj-w mlp-proj-b))

(define (block-params-from-tensors #:n-head n-head tensors)
  (unless (hash? tensors)
    (raise-argument-error 'block-params-from-tensors "hash?" tensors))
  (define (get name)
    (hash-ref tensors name
              (lambda ()
                (raise-arguments-error 'block-params-from-tensors
                                       "missing tensor"
                                       "name" name))))
  (make-block-params #:n-head n-head
                     #:ln1-gain (get "ln_1.weight")
                     #:ln1-bias (get "ln_1.bias")
                     #:c-attn-w (get "attn.c_attn.weight")
                     #:c-attn-b (get "attn.c_attn.bias")
                     #:attn-proj-w (get "attn.c_proj.weight")
                     #:attn-proj-b (get "attn.c_proj.bias")
                     #:ln2-gain (get "ln_2.weight")
                     #:ln2-bias (get "ln_2.bias")
                     #:c-fc-w (get "mlp.c_fc.weight")
                     #:c-fc-b (get "mlp.c_fc.bias")
                     #:mlp-proj-w (get "mlp.c_proj.weight")
                     #:mlp-proj-b (get "mlp.c_proj.bias")))

(struct block-trace (ln1 attn residual1 ln2 mlp-hidden out)
  #:transparent)

(define (gpt2-mlp-hidden x c-fc-w c-fc-b)
  (matmul-gelu x c-fc-w c-fc-b))

(define (gpt2-mlp x c-fc-w c-fc-b c-proj-w c-proj-b)
  (matmul (gpt2-mlp-hidden x c-fc-w c-fc-b) c-proj-w c-proj-b))

(define (gpt2-block-parts x params)
  (unless (block-params? params)
    (raise-argument-error 'gpt2-block "block-params?" params))
  (define ln1 (layer-norm x
                          (block-params-ln1-gain params)
                          (block-params-ln1-bias params)))
  (define attn (gpt2-attention ln1
                               (block-params-c-attn-w params)
                               (block-params-c-attn-b params)
                               (block-params-attn-proj-w params)
                               (block-params-attn-proj-b params)
                               (block-params-n-head params)))
  (define residual1 (tensor-add x attn))
  (define ln2 (layer-norm residual1
                          (block-params-ln2-gain params)
                          (block-params-ln2-bias params)))
  (define hidden (gpt2-mlp-hidden ln2
                                  (block-params-c-fc-w params)
                                  (block-params-c-fc-b params)))
  (define mlp (matmul hidden
                      (block-params-mlp-proj-w params)
                      (block-params-mlp-proj-b params)))
  (block-trace ln1 attn residual1 ln2 hidden (tensor-add residual1 mlp)))

(define (gpt2-block x params)
  (block-trace-out (gpt2-block-parts x params)))

(define (write-qkv! k v row mixed width)
  (define from (tensor-data mixed))
  (define kd (tensor-data k))
  (define vd (tensor-data v))
  (for ([i (in-range (car (tensor-dims mixed)))])
    (define src (+ (* i 3 width) width))
    (define dest (* (+ row i) width))
    (for ([j (in-range width)])
      (flvector-set! kd (+ dest j) (flvector-ref from (+ src j)))
      (flvector-set! vd (+ dest j) (flvector-ref from (+ src width j))))))

;; past-k and past-v are preallocated buffers. The new rows are written
;; at #:filled, and attend sees only the filled prefix.
(define (gpt2-block-cached x params past-k past-v #:filled filled #:workspace [workspace #f])
  (unless (block-params? params)
    (raise-argument-error 'gpt2-block-cached "block-params?" params))
  (unless (and (tensor? past-k) (tensor? past-v))
    (raise-arguments-error 'gpt2-block-cached
                           "k and v caches must be tensors"
                           "k" past-k
                           "v" past-v))
  (unless (exact-nonnegative-integer? filled)
    (raise-argument-error 'gpt2-block-cached "exact-nonnegative-integer?" filled))
  (define obs (current-trace-observer))
  (define ln1
    (if-trace obs 'block-ln1
      (stage block-ln1 layer-norm
        (layer-norm x
                    (block-params-ln1-gain params)
                    (block-params-ln1-bias params)))
      (lambda (y t0) (trace-activation! obs 'block-ln1 y t0))))
  (define mixed
    (if-trace obs 'block-qkv
      (stage block-qkv matmul
        (matmul ln1
                (block-params-c-attn-w params)
                (block-params-c-attn-b params)))
      (lambda (y t0) (trace-activation! obs 'block-qkv y t0))))
  (define-values (nq mixed-width) (matrix-shape 'gpt2-block-cached mixed))
  (define-values (cap width) (matrix-shape 'gpt2-block-cached past-k))
  (unless (and (= mixed-width (* 3 width)) (equal? (tensor-dims past-k) (tensor-dims past-v)))
    (raise-arguments-error 'gpt2-block-cached "cache shapes do not match the projected key/value width"
                           "mixed" (tensor-dims mixed) "k" (tensor-dims past-k) "v" (tensor-dims past-v)))
  (unless (<= (+ filled nq) cap)
    (raise-arguments-error 'gpt2-block-cached
                           "the cache is shorter than the next rows"
                           "filled" filled
                           "new" nq
                           "capacity" cap))
  (if (eq? obs #f)
      (stage kv-write write-qkv!
        (write-qkv! past-k past-v filled mixed width))
      (let ([t0 (current-inexact-monotonic-milliseconds)])
        (write-qkv! past-k past-v filled mixed width)
        (trace-kv! obs (kv-write-record filled nq cap width) t0)))
  (define kv-rows (+ filled nq))
  (define heads
    (if (eq? obs #f)
        (attend mixed past-k past-v (block-params-n-head params)
                #:kv-rows kv-rows #:workspace workspace #:query-width width)
        (let ([t0 (current-inexact-monotonic-milliseconds)])
          (define n-head (block-params-n-head params))
          (define slots (make-vector n-head '()))
          (define y
            (parameterize ([current-trace-stage 'attend])
              (attend mixed past-k past-v n-head
                      #:kv-rows kv-rows #:workspace workspace #:query-width width
                      #:weights-out slots)))
          (trace-attention! obs slots)
          (trace-activation! obs 'attend y t0)
          y)))
  (define residual1
    (if-trace obs 'block-proj-res
      (stage block-proj-res matmul-add
        (matmul-add heads
                    (block-params-attn-proj-w params)
                    (block-params-attn-proj-b params)
                    x))
      (lambda (y t0) (trace-activation! obs 'block-proj-res y t0))))
  (define ln2
    (if-trace obs 'block-ln2
      (stage block-ln2 layer-norm
        (layer-norm residual1
                    (block-params-ln2-gain params)
                    (block-params-ln2-bias params)))
      (lambda (y t0) (trace-activation! obs 'block-ln2 y t0))))
  (define hidden
    (if-trace obs 'mlp-gelu
      (stage mlp-gelu gpt2-mlp-hidden
        (gpt2-mlp-hidden ln2
                         (block-params-c-fc-w params)
                         (block-params-c-fc-b params)))
      (lambda (y t0) (trace-activation! obs 'mlp-gelu y t0))))
  (define out
    (if (eq? obs #f)
        (stage block-mlp-res matmul-add
          (matmul-add hidden
                      (block-params-mlp-proj-w params)
                      (block-params-mlp-proj-b params)
                      residual1))
        (let ([t0 (current-inexact-monotonic-milliseconds)]
              [cap (box #f)])
          (define y
            (parameterize ([current-trace-stage 'block-mlp-res]
                           [current-trace-capture cap])
              (matmul-add hidden
                          (block-params-mlp-proj-w params)
                          (block-params-mlp-proj-b params)
                          residual1)))
          (trace-activation! obs 'mlp-cproj (unbox cap) #f)
          (trace-activation! obs 'block-mlp-res y t0)
          y)))
  (values out past-k past-v))

;; Layout of one KV write. elementBytes is the IEEE-754 width of an
;; flvector slot. The write stores into the existing buffers; it does
;; not allocate a key or value copy.
(define (kv-write-record filled nq cap width)
  (define element-bytes 8)
  (hasheq 'filledBefore filled
          'filledAfter (+ filled nq)
          'capacity cap
          'width width
          'rowsWritten nq
          'elementBytes element-bytes
          'strideBytes (* width element-bytes)
          'bytesWritten (* nq width element-bytes 2)
          'kByteOffset (* filled width element-bytes)
          'vByteOffset (* filled width element-bytes)
          'srcStrideElements (* 3 width)
          'dstStrideElements width
          'zeroCopy #t
          'layout "row-major flvector; distinct K and V buffers; head columns are read in place"))

(define (gpt2-blocks x params-list)
  (unless (list? params-list)
    (raise-argument-error 'gpt2-blocks "list?" params-list))
  (for/fold ([hidden x]) ([params (in-list params-list)])
    (gpt2-block hidden params)))
